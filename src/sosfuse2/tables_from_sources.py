#!/usr/bin/env python3
"""Batch: stored fused sources -> table.npz + meta.json + previews + zstd pack.

Scans <work-root>/tmp/<id>/source.npz; processes states whose
<work-root>/out/<id>/table.npz is missing and that no live run_gpu_state_v3
worker (script or `-m sosfuse2.run_gpu_state_v3`) currently owns. Uses
table_fast.make_table (same rays, same kernel, same post-processing as
run_gpu_state_v3) and sosfuse2.host.pack_state.pack_state.
The meta signature is this batch's own hash (state, backend, grid, code_hashes), not
run_gpu_state_v3.signed_contract: run_gpu_state_v3 refuses such an out_name
(E_OUTPUT_IDENTITY), so give these tables their own work root.
"""

import argparse, fcntl, hashlib, json, os, re, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
from sosfuse2 import paths

THREAD_KEYS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS',
)


def live_workers():
    """(work_root, id) pairs of running run_gpu_state*.py / `-m sosfuse2.run_gpu_state_v3` processes (Linux /proc)."""
    owned = set()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        args = [x.decode(errors='replace') for x in args if x]
        if not any(re.search(r'(run_gpu_state(_v[23])?\.py|sosfuse2\.run_gpu_state_v3)$', x) for x in args):
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
    from sosfuse2.run_gpu_state_v3 import package_hashes
    from sosfuse2.components import physics, optics_state, canonical_state

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
        tolerance=0.005,
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
    from sosfuse2 import table_fast as tf
    from sosfuse2.host.pack_state import pack_state
    from sosfuse2.run_gpu_state_v3 import CONTRACT_VERSION
    from sosfuse2.components import physics, optics_state, check_same_state, production_only

    production_only()
    out = root / 'out' / ident
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    settings = tf.restored_settings(source, state, a.reduced, a.backend)
    optical_meta = None
    wl, hs, el, az = tf.axes(a.reduced) if settings is None else tf.restored_axes(settings)
    signature, hashes = signature_of(state, a.backend, a.reduced, wl, hs, len(el), len(az), settings)
    meta_path = out / 'meta.json'
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        check_same_state(old, dict(physics=physics(), optics_state=optics_state(state)))
        if old.get('signature') != signature:
            raise ValueError('E_OUTPUT_IDENTITY: batch table signature differs')
        if old.get('complete') and (out / 'table.npz').is_file():
            pack_state(out, root / 'pack', 'zstd')
            return 'packed-existing'
    if settings is None:
        t = time.monotonic()
        s, scope, settings = tf.build_solver(state, root / 'tmp' / ident / 'prepare', a.reduced, a.backend)
        create_seconds = time.monotonic() - t
        field = tf.load_source(source, s, wl, settings=settings, backend=a.backend)
        t = time.monotonic()
        table, seconds = tf.make_table(s, field, hs, el, az, a.group, a.batch, log)
        table_seconds = time.monotonic() - t
    else:
        table, seconds, create_seconds, table_seconds, optical_meta = tf.restored_table(
            source, state, settings, root / 'tmp' / ident / 'prepare', a.backend, a.group, a.batch, log
        )
    tf.save_table(out / 'table.npz', table, wl, hs, el, az)
    meta = dict(
        complete=True,
        contract=CONTRACT_VERSION,
        physics=physics(),
        optics_state=optics_state(state),
        signature=signature,
        id=ident,
        parameters=state['parameters'],
        stratosphere=state['stratosphere'],
        mode='reduced-validation' if a.reduced else 'full',
        shape=list(table.shape),
        dtype='float32',
        settings=settings,
        source_solve=None,
        source_file=str(source.relative_to(root)),
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
        convergence_scope='source and 35-box residual <=0.5% (stored source; solve stats in the worker log)',
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
    p.add_argument('--states', type=Path, default=paths.ASSETS / 'states_v1.json')
    p.add_argument('--ids', nargs='*')
    p.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES for this process')
    p.add_argument('--backend', choices=['cupy', 'numpy'], default='cupy')
    p.add_argument('--reduced', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--group', type=int, default=32)
    p.add_argument('--batch', type=int, default=25)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.backend == 'numpy' and not a.reduced:
        p.error('numpy requires --reduced')
    if not 1 <= a.threads <= 4:
        p.error('threads 1..4')
    for key in THREAD_KEYS:
        os.environ[key] = str(a.threads)
    if a.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    states = {s['id']: s for s in json.loads(a.states.read_text())['states'] if s.get('runnable')}
    roots = [r.resolve() for r in a.work_roots]
    live = live_workers() if Path('/proc').is_dir() else set()
    log = lambda m: print(m, flush=True)
    failed = False
    for root, ident, source in candidates(roots, set(a.ids or [])):
        out = root / 'out' / ident
        if (out / 'table.npz').is_file() and (root / 'pack' / f'{ident}.tar.zst').is_file():
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
            # run_gpu_state_v3 owns out/<id> through .worker.lock, and the live_workers() sample above is taken once and
            # not on every host, so the batch takes the same lock for the whole of process() and looks again once it holds it.
            with (out / '.worker.lock').open('a') as owner:
                try:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    log(f'SKIP {root.name}/{ident}: a worker owns the output (.worker.lock)')
                    continue
                if (out / 'table.npz').is_file() and (root / 'pack' / f'{ident}.tar.zst').is_file():
                    log(f'SKIP {root.name}/{ident}: finished while waiting for the locks')
                    continue
                try:
                    log(f'TABLES {root.name}/{ident} {process(root, ident, source, states[ident], a, log)}')
                except (ValueError, RuntimeError) as e:
                    failed = True
                    log(f'FAIL {root.name}/{ident}: {e}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
