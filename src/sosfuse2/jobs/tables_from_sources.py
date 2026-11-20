#!/usr/bin/env python3
"""Batch: stored fused sources -> table.npz + meta.json + previews + zstd pack.

Scans stored sources and skips outputs that pass output_complete. Live workers and locked states are skipped.
Uses radiance_table.make_table (same rays, same kernel, same post-processing as
run_state) and sosfuse2.outputs.pack_state.pack_state.
The meta signature is this batch's own hash (state, backend, grid, code_hashes), not
run_state.signed_contract: run_state refuses such an out_name
(E_OUTPUT_IDENTITY), so give these tables their own work root.
"""

import argparse, fcntl, hashlib, json, os, re, sys, tarfile, time, zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
import zlib

if __name__ == '__main__':
    sys.dont_write_bytecode = True
from sosfuse2.foundation import paths

THREAD_KEYS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS',
)


ARCHIVE_ERRORS = (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile, zlib.error, tarfile.TarError)


def check_pack(out, pack, meta, restore=None):
    from .run_state import verify_pack
    from ..outputs.radiance_table import exports_first_order

    first_sha = None
    if exports_first_order(meta) and restore is None:
        with (out / 'first_order.npz').open('rb') as stream:
            first_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    try:
        verify_pack(pack, meta['signature'], export_first_order=exports_first_order(meta), first_order_sha=first_sha,
                    loose_table=out / 'table.npz', restore_first=restore)
    except ARCHIVE_ERRORS as exc:
        raise RuntimeError(f'E_PACK_VERIFY: {type(exc).__name__}: {exc}') from exc


def check_identity(meta, stream, state, args, restored, source_sha):
    import numpy as np
    from ..atmosphere.components import canonical_state, check_same_state, optics_state, physics
    from ..atmosphere.spectral import same_settings
    from ..foundation.content_identity import state_identity
    from ..outputs import radiance_table as tf
    from .run_state import signed_contract

    stream.seek(0)
    with np.load(stream) as archive:
        contract = json.loads(str(archive['source_contract']))
    if not isinstance(contract, dict) or not isinstance(contract.get('state'), dict):
        raise ValueError('source state missing')
    cfg = contract['settings']
    tol = tf.check_source_tol(contract, cfg)
    mode = 'reduced-validation' if args.reduced else 'full'
    if (state_identity(contract['state']) != canonical_state(state) or contract['backend'] != args.backend
            or contract['polarization_model'] != 'depolarizing' or meta.get('backend') != args.backend
            or meta.get('mode') != mode or meta.get('anderson_tol') != tol or meta.get('id') != state['id']
            or meta.get('parameters') != state.get('parameters')):
        raise ValueError('state or settings differ')
    check_same_state(meta, dict(physics=physics(), optics_state=optics_state(state)))
    method = meta.get('table_method')
    producer = method.get('runner') if isinstance(method, dict) else None
    if producer == 'run_gpu_state_v3':
        signature, _ = signed_contract(state, cfg, args.backend, args.reduced, 'depolarizing', meta.get('derived'))
        if meta.get('polarization_model') != 'depolarizing':
            raise ValueError('polarization differs')
    elif producer == 'tables_from_sources':
        if meta.get('source_sha256') != source_sha:
            raise ValueError('source digest differs or is missing')
        wl, hs, el, az = tf.axes(args.reduced) if restored is None else tf.restored_axes(restored)
        signature, _ = signature_of(state, args.backend, args.reduced, wl, hs, len(el), len(az), restored)
    else:
        raise ValueError('unknown table producer')
    if meta.get('signature') != signature or not isinstance(meta.get('settings'), dict) or not same_settings(meta['settings'], cfg):
        raise ValueError('signature or settings differ')


def compare_first_order(archived, rebuilt):
    import numpy as np
    from ..outputs.radiance_table import TABLE_AXES, read_table

    archived_axes, reference = read_table(archived)
    rebuilt_axes, actual = read_table(rebuilt)
    different = [name for name, a, b in zip(TABLE_AXES, archived_axes, rebuilt_axes) if not np.array_equal(a, b)]
    if different:
        raise ValueError(f'E_PACK_TABLE_MISMATCH: {rebuilt.name}: axes differ ({", ".join(different)})')

    solar_axis = TABLE_AXES.index('solar_elevation_deg') if 'solar_elevation_deg' in TABLE_AXES else None
    reduce_axes = None if solar_axis is None else tuple(i for i in range(reference.ndim) if i != solar_axis)
    reference = reference.astype('f8')
    delta = np.max(np.abs(actual - reference), axis=reduce_axes)
    magnitude = np.max(np.abs(reference), axis=reduce_axes)
    normalized = np.divide(delta, magnitude, out=np.where(delta == 0, 0.0, np.inf), where=magnitude != 0)
    maximum = normalized.max()
    if maximum > 1e-3:
        location = 'across the table'
        if solar_axis is not None:
            location = f'at solar elevation {archived_axes[solar_axis][normalized.argmax()]:g} deg'
        raise ValueError(
            f'E_PACK_TABLE_MISMATCH: {rebuilt.name}: max normalized difference {maximum:.9g} {location} exceeds 1e-3'
        )


def output_complete(out, pack, source=None, state=None, args=None):
    from ..outputs import radiance_table as tf

    if not (out / 'table.npz').is_file() or not pack.is_file():
        return False
    try:
        meta = json.loads((out / 'meta.json').read_text())
        if not isinstance(meta, dict) or meta.get('complete') is not True or not tf.first_order_complete(out, meta):
            return False
        if source is not None:
            if state is None or args is None:
                return False
            with source.open('rb') as stream:
                source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                stream.seek(0)
                restored = tf.restored_settings(stream, state, args.reduced, args.backend)
                check_identity(meta, stream, state, args, restored, source_sha)
        tf.read_table(out / 'table.npz')
        check_pack(out, pack, meta)
    except (RuntimeError, *ARCHIVE_ERRORS):
        return False
    return True


def live_workers():
    """(work_root, id) pairs of running state scripts or modules (Linux /proc)."""
    owned = set()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        args = [x.decode(errors='replace') for x in args if x]
        if not any(re.search(r'(run_gpu_state(_v[23])?\.py|sosfuse2\.run_gpu_state_v3|sosfuse2\.jobs\.run_state|(?:^|/)jobs/run_state\.py)$', x) for x in args):
            continue
        try:
            ident = args[args.index('--id') + 1]
            root = Path(args[args.index('--work-root') + 1]).resolve()
        except (ValueError, IndexError):
            continue
        owned.add((str(root), ident))
    return owned


def signature_of(state, backend, reduced, wl, hs, nr, nc, settings=None):
    """Signature of a batch table: this module's own `contract` dict hashed with the package code hashes."""
    from sosfuse2.jobs.run_state import package_hashes
    from sosfuse2.jobs.run_state import effective_tol
    from sosfuse2.atmosphere.components import physics, optics_state, canonical_state

    hashes = package_hashes()
    contract = dict(
        physics=physics(),
        optics_state=optics_state(state),
        state=canonical_state(state),
        backend=backend,
        reduced=reduced,
        wavelengths_nm=wl.tolist(),
        solar_elevation_deg=hs.tolist(),
        view_shape=[nr, nc],
        tolerance=effective_tol(settings or {}),
        code_hashes=hashes,
    )
    if settings is not None:
        contract['settings'] = settings
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(), hashes


def candidates(roots, ids):
    rows = []
    for root in roots:
        for source in sorted(root.glob('tmp/*/source.npz')):
            ident = source.parent.name
            if ids and ident not in ids:
                continue
            if not re.fullmatch('[A-Za-z0-9_-]+', ident):
                continue
            rows.append((root, ident, source))
    return rows


def process(root, ident, source, state, a, log):
    from .run_state import output_identity_message

    try:
        with source.open('rb') as stream:
            return process_source(root, ident, source, stream, state, a, log)
    except (EOFError, zipfile.BadZipFile, zlib.error) as exc:
        reason = f'unreadable archive {source}: {type(exc).__name__}: {exc}'
        raise ValueError(output_identity_message(root / 'out' / ident, reason)) from exc
    except (KeyError, TypeError) as exc:
        raise ValueError(f'E_OUTPUT_PROCESS: {type(exc).__name__}: {exc}') from exc


def process_source(root, ident, source, stream, state, a, log):
    import numpy as np
    from sosfuse2.outputs import radiance_table as tf
    from sosfuse2.outputs.pack_state import pack_state
    from sosfuse2.jobs.run_state import CONTRACT_VERSION, effective_tol, output_identity_message
    from sosfuse2.atmosphere.components import physics, optics_state, production_only

    production_only()
    out = root / 'out' / ident
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    stream.seek(0)
    meta_path = out / 'meta.json'
    try:
        settings = tf.restored_settings(stream, state, a.reduced, a.backend)
    except ValueError as exc:
        if meta_path.exists() and not str(exc).startswith('E_SOURCE_CONTRACT: unreadable source archive'):
            raise ValueError(output_identity_message(out, str(exc))) from exc
        raise
    optical_meta = None
    wl, hs, el, az = tf.axes(a.reduced) if settings is None else tf.restored_axes(settings)
    signature, hashes = signature_of(state, a.backend, a.reduced, wl, hs, len(el), len(az), settings)
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f'E_OUTPUT_META: cannot read {meta_path.name}: {type(exc).__name__}') from exc
        if not isinstance(old, dict):
            raise ValueError('E_OUTPUT_META: expected a metadata object')
        try:
            check_identity(old, stream, state, a, settings, source_sha)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(output_identity_message(out, str(exc))) from exc
        pack = root / 'pack' / f'{ident}.tar.zst'
        total = out / 'table.npz'
        if total.exists():
            tf.read_table(total)
        if pack.exists():
            if not total.is_file():
                raise ValueError(output_identity_message(out, 'loose table missing'))
            first = out / 'first_order.npz'
            if tf.exports_first_order(old) and not first.exists():
                with TemporaryDirectory(dir=out) as directory:
                    scratch = Path(directory)
                    archived = scratch / 'archived.npz'
                    with archived.open('wb') as tmp:
                        check_pack(out, pack, old, tmp)
                    if old.get('complete') is not True or not tf.first_order_complete(out, old, first_path=archived):
                        raise ValueError('E_PACK_VERIFY: invalid archived first-order table')
                    stream.seek(0)
                    # Replay in scratch so a failed comparison cannot change the published tables.
                    tf.restored_table(stream, state, settings, root / 'tmp' / ident / 'prepare',
                                      a.backend, a.group, a.batch, log, first_order_out=scratch)
                    compare_first_order(archived, scratch / first.name)
                    archived.chmod(0o644)
                    os.replace(archived, first)
            else:
                check_pack(out, pack, old)
            if old.get('complete') is not True or not tf.first_order_complete(out, old):
                raise ValueError('E_PACK_VERIFY: incomplete loose output')
            return 'verified-existing'
        if (out / 'first_order.npz').is_file() and old.get('complete') is True and not tf.first_order_complete(out, old):
            raise ValueError('E_PACK_VERIFY: invalid loose first-order table')
        if old.get('complete') is True and total.is_file() and tf.first_order_complete(out, old):
            pack_state(out, root / 'pack', 'zstd')
            return 'packed-existing'
    elif any(p.exists() for p in (out / 'table.npz', out / 'first_order.npz', root / 'pack' / f'{ident}.tar.zst')):
        raise ValueError(output_identity_message(out, 'existing output has no metadata'))
    stream.seek(0)
    if settings is None:
        t = time.monotonic()
        s, scope, settings = tf.build_solver(state, root / 'tmp' / ident / 'prepare', a.reduced, a.backend)
        create_seconds = time.monotonic() - t
        field = tf.load_source(stream, s, wl, settings=settings, backend=a.backend)
        stream.seek(0)
        with np.load(stream) as archive:
            settings = json.loads(str(archive['source_contract']))['settings']
        t = time.monotonic()
        table, seconds = tf.make_table(s, field, hs, el, az, a.group, a.batch, log)
        table_seconds = time.monotonic() - t
        from ..atmosphere.stratosphere import computed_metadata

        from ..atmosphere.density import metadata as density_metadata

        optical_meta = dict(computed_metadata(state, getattr(s, 'strato', None)), density=density_metadata(s))
    else:
        table, seconds, create_seconds, table_seconds, optical_meta = tf.restored_table(
            stream, state, settings, root / 'tmp' / ident / 'prepare', a.backend, a.group, a.batch, log,
            first_order_out=out, table_path=out / 'table.npz',
        )
    if not settings.get('export_first_order'):
        tf.mark_incomplete(out / 'table.npz')
    tf.save_table(out / 'table.npz', table, wl, hs, el, az)
    meta = dict(
        complete=True,
        contract=CONTRACT_VERSION,
        physics=physics(),
        optics_state=optics_state(state),
        signature=signature,
        id=state['id'],
        parameters=state['parameters'],
        stratosphere=state.get('stratosphere'),
        mode='reduced-validation' if a.reduced else 'full',
        shape=list(table.shape),
        dtype='float32',
        settings=settings,
        anderson_tol=effective_tol(settings),
        source_solve=None,
        source_file=str(source.relative_to(root)),
        source_sha256=source_sha,
        source_seconds=None,
        query_seconds=seconds,
        table_seconds=table_seconds,
        solver_create_seconds=create_seconds,
        elapsed_seconds=time.monotonic() - started,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        backend=a.backend,
        threads=a.threads,
        unsupported_parameters=['stratosphere_tau550'],
        folding='mirror azimuth around solar plane; cell centers 0.5..179.5 in full mode',
        units=tf.UNITS,
        convergence_scope=f'source and 35-box residual <={100 * effective_tol(settings):g}% (stored source; solve stats in the worker log)',
        table_method=dict(runner='tables_from_sources', table='table_fast.make_table', group=a.group, batch=a.batch),
        code_hashes=hashes,
    )
    if optical_meta is not None:
        meta.update(optical_meta, unsupported_parameters=[])
    tmp = meta_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, meta_path)
    pack_state(out, root / 'pack', 'zstd', started=started)
    return f'done create={create_seconds:.1f}s table={table_seconds:.1f}s total={time.monotonic() - started:.1f}s'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--work-roots', type=Path, nargs='+', required=True)
    p.add_argument('--states', type=Path, default=paths.ASSETS / 'states_v2.json')
    p.add_argument('--ids', nargs='*')
    p.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES for this process')
    p.add_argument('--backend', choices=['cupy', 'numpy'], default='cupy')
    p.add_argument('--reduced', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--group', type=int, default=32)
    p.add_argument('--batch', type=int, default=25)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.group < 1:
        p.error('E_TABLE_GROUP: group must be positive')
    if a.backend == 'numpy' and not a.reduced:
        p.error('numpy requires --reduced')
    if not 1 <= a.threads <= 4:
        p.error('threads 1..4')
    for key in THREAD_KEYS:
        os.environ[key] = str(a.threads)
    if a.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    from .run_state import read_states

    states = {s['id']: s for s in read_states(a.states, True) if s.get('runnable')}
    roots = [r.resolve() for r in a.work_roots]
    live = live_workers() if Path('/proc').is_dir() else set()
    log = lambda m: print(m, flush=True)
    failed = False
    for root, ident, source in candidates(roots, set(a.ids or [])):
        out = root / 'out' / ident
        pack = root / 'pack' / f'{ident}.tar.zst'
        if ident in states and output_complete(out, pack, source, states[ident], a):
            continue
        if (str(root), ident) in live:
            log(f'SKIP {root.name}/{ident}: live worker owns it')
            continue
        if ident not in states:
            log(f'SKIP {root.name}/{ident}: not a runnable state')
            continue
        if a.dry_run:
            log(f'WOULD {root.name}/{ident}')
            continue
        out.mkdir(parents=True, exist_ok=True)
        with (out / '.tables_from_sources.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log(f'SKIP {root.name}/{ident}: locked by another batch')
                continue
            # run_state owns out/<id> through .worker.lock, and the live_workers() sample above is taken once and
            # not on every host, so the batch takes the same lock for the whole of process() and looks again once it holds it.
            with (out / '.worker.lock').open('a') as owner:
                try:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    log(f'SKIP {root.name}/{ident}: a worker owns the output (.worker.lock)')
                    continue
                if output_complete(out, pack, source, states[ident], a):
                    log(f'SKIP {root.name}/{ident}: finished while waiting for the locks')
                    continue
                try:
                    log(f'TABLES {root.name}/{ident} {process(root, ident, source, states[ident], a, log)}')
                except (OSError, ValueError, RuntimeError) as e:
                    failed = True
                    log(f'FAIL {root.name}/{ident}: {e}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
