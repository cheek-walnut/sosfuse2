"""Disposable host sidecars: atomic, checked, and independent of optical state.

Columns 0, 2, 3 and 4 are shared across scale heights. The host aerosol column (col1) is keyed by the aerosol
profile (scale height, troposphere profile) and shared across optics (aod, ssa, g, ...). No solver
outputs are persisted. The single-file envelope prevents a reader seeing mixed data/metadata generations.
"""

import fcntl
import hashlib
import io
import json
import os
import re
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
import numpy as np

from . import identity as source_identity, paths

VERSION = 3
PKG = paths.PKG


ROOT = None  # override of the default cache root (tests patch it); None = work root


def default_root():
    """Default packed-geometry cache root (public split: under the work root, not inside a results/ folder)."""
    return Path(ROOT) if ROOT is not None else paths.work_root() / 'geometry_cache'


class CacheUnavailable(RuntimeError):
    """A filesystem failure requires the caller's uncached calculation."""


def unavailable(code, path, error):
    reason = code + ': ' + str(path)
    warnings.warn(reason, RuntimeWarning)
    raise CacheUnavailable(reason) from error


def housekeeping(folder, token):
    """Remove abandoned temp files only; report other generations unchanged."""
    cutoff = time.time() - 3600
    for path in folder.glob('*.tmp'):
        try:
            if path.is_symlink() or path.stat().st_mtime > cutoff:
                continue
            with path.open('rb') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if os.fstat(f.fileno()).st_mtime <= cutoff:
                    path.unlink()
        except FileNotFoundError:
            pass  # Another builder cleaned or published it.
    stale = []
    for path in folder.iterdir():
        if re.fullmatch(r'[0-9a-f]{64}\..+', path.name) and not path.name.startswith(token + '.'):
            try:
                if path.is_file() and not path.is_symlink():
                    stale.append(path.stat().st_size)
            except FileNotFoundError:
                pass
    if stale:
        warnings.warn(f'HOST_CACHE_STALE n={len(stale)} bytes={sum(stale)}', RuntimeWarning)


def enabled():
    return os.environ.get('SOSFUSE2_HOST_CACHE', '1') != '0'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=hashlib.sha256(a.tobytes()).hexdigest())


def context(s, optics):
    from .geometry_contract import identity
    from .density import DEFAULTS, column_identity

    common, signature = identity(s)
    # Include the complete common identity, not only its shortened directory name. Profile: effective table bytes,
    # grid N/step, mask, scale, normalization and origin; options/asset/mask alone do not separate injected tables.
    # Additional sources cover sun_parallel rows, ground_parallel cells, density corrections, column workers and methods,
    # ray_columns' GL walk, density profiles and the solver_sky incoming quadrature. Common-identity sources are in common.
    key = dict(
        version=VERSION,
        common=common,
        options={k: optics['meta'][k] for k in DEFAULTS},
        asset_sha256=optics['meta'].get('asset_sha256'),
        mask=optics['mask'],
        profile=column_identity(optics),
        selected_columns=[0, 2, 3, 4],
        sources={
            'sosfuse2/' + name: source_identity.file_sha(PKG / name)
            for name in (
                'host_cache.py',
                'density.py',
                'ray_columns.py',
                'solver_sky.py',
                'sun_parallel.py',
                'ground_parallel.py',
                'column_table.py',
                'column_parallel.py',
            )
        },
    )
    root = Path(getattr(s, 'host_cache_root', None) or default_root())
    return root / ('endpoints_' + signature) / 'host_cache', key


def cached(folder, key, build, compressed=False):
    """One builder across processes; retry reads after acquiring the writer lock.

    SHA/size checks cover the NPZ payload; array descriptors check decoded shape
    and dtype as well. Keys cover semantic inputs. A failed build publishes nothing.
    """
    folder = Path(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        unavailable('E_HOST_CACHE_UNWRITABLE', folder, error)
    token = digest(key)
    path = folder / (token + '.hcache')

    def read():
        with path.open('rb') as f:
            header = json.loads(f.readline(1024 * 1024))
            data = f.read()
        if header['key'] != key or len(data) != header['size'] or hashlib.sha256(data).hexdigest() != header['sha256']:
            raise ValueError('envelope')
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files}
        if {name: dict(shape=list(a.shape), dtype=a.dtype.str) for name, a in arrays.items()} != header['arrays']:
            raise ValueError('arrays')
        return arrays

    failures = (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile)
    try:
        return read()
    except PermissionError as error:
        unavailable('E_HOST_CACHE_PERMISSION', path, error)
    except failures:
        pass
    try:
        lock = (folder / (token + '.lock')).open('a')
    except OSError as error:
        unavailable('E_HOST_CACHE_UNWRITABLE', folder, error)
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError as error:
            unavailable('E_HOST_CACHE_UNWRITABLE', folder, error)
        # A concurrent builder may have published while we waited.
        for attempt in range(2):
            try:
                return read()
            except PermissionError as error:
                unavailable('E_HOST_CACHE_PERMISSION', path, error)
            except FileNotFoundError:
                break
            except failures:
                if attempt == 0:
                    time.sleep(0.01)
        if path.exists():
            warnings.warn('E_HOST_CACHE_CORRUPT: ' + path.name, RuntimeWarning)
        try:
            housekeeping(folder, token)
        except OSError as error:
            unavailable('E_HOST_CACHE_UNWRITABLE', folder, error)
        arrays = {name: np.ascontiguousarray(a) for name, a in build().items()}
        buf = io.BytesIO()
        (np.savez_compressed if compressed else np.savez)(buf, **arrays)
        data = buf.getvalue()
        header = dict(
            key=key,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            arrays={name: dict(shape=list(a.shape), dtype=a.dtype.str) for name, a in arrays.items()},
        )
        try:
            fd, name = tempfile.mkstemp(prefix=token + '.', suffix='.tmp', dir=folder)
        except OSError as error:
            unavailable('E_HOST_CACHE_UNWRITABLE', folder, error)
        try:
            with os.fdopen(fd, 'wb') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(header, sort_keys=True).encode() + b'\n')
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                os.chmod(name, 0o644)
                os.replace(name, path)
        except OSError as error:
            unavailable('E_HOST_CACHE_UNWRITABLE', folder, error)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
            except OSError as error:
                warnings.warn('E_HOST_CACHE_UNWRITABLE: ' + str(folder), RuntimeWarning)
        return arrays


def sun_geometry(old, scale, folder, height, shared=None):
    # The SunGeometry file carries an identity (code, components, aerosol profile, scale; solver_columns.SunGeometry.identity) and a
    # mismatch rebuilds it, so the folders below are for reuse, not safety: they keep states from replacing one file in turn
    # when a caller reuses a prepare folder for a different state; scale 1 keeps its old folder.
    # shared = the job's geometry cache: one table per (table source, aerosol profile, scale, height) for every state.
    # Unwritable shared root: warn and keep the per-state folder.
    from .sun_parallel import prebuild

    local = folder
    if shared is not None:
        # Table identity: the code that computes it (old.total_columns is compiled from solver_columns.py), its
        # dependencies and radius/top; scale and height key the folders below.
        here = Path(__file__).resolve()
        # tropo_profile.py: the aerosol_profile tag hashes the profile config, not the code that evaluates it.
        files = sorted(
            {
                str(here.parent / 'solver_columns.py'),
                str(here.parent / 'accepted/engine.py'),
                str(here.with_name('sun_parallel.py')),
                str(here.with_name('column_parallel.py')),
                str(here.with_name('tropo_profile.py')),
                str(here.with_name('ray_columns.py')),
                # solver_columns.profiles reads density.no2_profile, which interpolates this asset.
                str(here.with_name('density.py')),
                str(here.parent / 'assets/no2_afglus_profile.json'),
            }
        )
        source = hashlib.sha256(json.dumps(dict(R=float(old.R), Z=float(old.Z))).encode())
        for name in files:
            p = Path(name)
            digest = source_identity.asset_sha(p) if p.name in source_identity.ASSET_NOTES else source_identity.file_sha(p)
            source.update(digest.encode())
        folder = Path(shared) / 'host_sun' / source.hexdigest()[:20]
        # tropo_profile.AER_FOLDER(cache) = cache/aerosol_profile/<tag>: the table depends on that profile too.
        if local.parent.name == 'aerosol_profile':
            folder = folder / 'aerosol_profile' / local.name
    for root in (folder, local) if shared is not None else (folder,):
        target = root / 'host_sun' if root is local else root
        if scale != 1:
            target = target / f'scale{scale}'
        target = target / float(height).hex()
        try:
            target.mkdir(parents=True, exist_ok=True)
            with (target / '.build.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                seconds = prebuild(old, scale, target)
                sun = old.SunGeometry(scale, target)
        except OSError as error:
            # Shared root: also a write that fails after the lock opened (prebuild's mkstemp/replace).
            if root is local:
                raise
            reason = (
                f'{type(error).__name__} errno={error.errno} {os.strerror(error.errno)}'
                if error.errno
                else type(error).__name__
            )
            warnings.warn(f'HOST_SUN_SHARED_UNAVAILABLE: {target}: {reason}', RuntimeWarning)
            continue
        folder = target
        break
    if (len(sun.h), len(sun.x)) != (80 * scale + 1, 400 * scale + 1):
        raise RuntimeError(
            f'E_SUN_TABLE_SCALE: {folder} holds a {len(sun.h)}x{len(sun.x)} table, scale {scale} expected'
        )
    if seconds is not None:
        sun.build_seconds += seconds
    return sun


def corrected_sun(s, optics, method=None):
    from .density import sun_method

    method = sun_method() if method is None else method  # density.attach passes its one resolution
    sun = s.geo.sun
    folder, key = context(s, optics)
    key.update(
        kind='sun02',
        h=array_hash(sun.h),
        x=array_hash(sun.x),
        lengths=[len(sun.h), len(sun.x)],
        base02=array_hash(sun.col[..., [0, 2, 3, 4]]),
        sun_method=method,
    )

    def build():
        from .sun_parallel import rows

        grid = rows(optics['old'], sun.h, sun.x, 6371.0, 32, sun.col, optics, method)
        return dict(columns=grid[..., [0, 2, 3, 4]])

    result = sun.col.copy()
    result[..., [0, 2, 3, 4]] = cached(folder, key, build)['columns']
    return result


def endpoint_columns(s, host, optics, folder=None, signature=None, prepared=None):
    from .density import correct_columns

    default, key = context(s, optics)
    folder = default if folder is None else Path(folder) / 'host_cache'
    key.update(
        kind='endpoint02',
        endpoint_signature=signature,
        lengths=[len(host['starts']), len(host['dirs'])],
        inputs={k: array_hash(host[k]) for k in ('starts', 'cell_off', 'dirs')},
        base02=array_hash(host['end_col'][:, [0, 2, 3, 4]]),
    )

    def build():
        if prepared is not None:
            return dict(columns=prepared)
        starts = host['starts']
        co = host['cell_off']
        dirs = host['dirs']
        out = np.empty((len(dirs), 4), 'f4')
        from . import column_parallel as cp
        from .column_table import method

        if method() != 'gl24' or (cp.workers() > 1 and len(dirs) >= cp.MIN_RAYS):
            # Opt-in (SOSFUSE2_COLUMN_WORKERS): the same 2048-ray correct_columns calls on the same end_col rows,
            # so columns 0/2 equal the loop below byte for byte; False = run that loop.
            full = np.array(host['end_col'], 'f4')
            if cp.parallel_columns(full, 'density', starts, co, dirs, optics['old'], optics):
                return dict(columns=np.ascontiguousarray(full[:, [0, 2, 3, 4]]))
        for lo in range(0, len(dirs), 2048):
            hi = min(lo + 2048, len(dirs))
            index = np.searchsorted(co, np.arange(lo, hi), side='right') - 1
            value = correct_columns(starts[index], dirs[lo:hi], host['end_col'][lo:hi], optics['old'], optics, 24)
            out[lo:hi] = value[:, [0, 2, 3, 4]]
        return dict(columns=out)

    try:
        return cached(folder, key, build)['columns']
    except CacheUnavailable:
        return build()['columns']


def aerosol_identity(s):
    """What old.total_columns(...)[:,1] reads besides geometry: the aerosol profile.

    Optics scale the column later through s.ext and are left out (same split as the strato attach key,
    stratosphere.ATTACH_OPTICS_PARAMS). The troposphere blocks patch the engine source per state.
    """
    from .stratosphere import ATTACH_OPTICS_PARAMS
    from .tropo_profile import LEGACY_BLOCKS

    state = s.fuse_state
    return dict(
        params={k: v for k, v in state.get('parameters', {}).items() if k not in ATTACH_OPTICS_PARAMS},
        blocks={k: state.get(k) for k in LEGACY_BLOCKS},
        sources={
            'sosfuse2/' + name: source_identity.file_sha(PKG / name)
            for name in ('ground_parallel.py', 'tropo_profile.py')
        },
    )


def prepare_ground(s, scope, optics, method=None):
    """Cache geometry, retaining full per-cell BLAS shapes and accumulation order.

    The geometric candidate mask only avoids columns for provably zero-weight
    rays. The accepted value>0 take mask is evaluated afresh for every state.
    """
    from .density import sun_method

    method = sun_method() if method is None else method
    b = scope.b
    old = b.sh.old
    g = s.geo
    folder, key = context(s, optics)
    key.update(
        kind='ground', lengths=[g.ns, len(g.h), len(g.chi)], sun_h=array_hash(g.sun.h), sun_x=array_hash(g.sun.x)
    )
    aerosol_key = dict(
        key, kind='ground_aerosol01', aerosol=aerosol_identity(s)
    )  # col1 does not read the PROFILE method
    key.update(sun_method=method)

    def build():
        from .ground_parallel import cells

        rows = {k: [] for k in ('loc', 'w', 'q', 'r', 'columns', 'candidate')}
        off = [0]
        for values in cells(s, scope, 'host', optics, method=method):
            for name, value in zip(rows, values):
                rows[name].append(value)
            off.append(off[-1] + len(values[0]))
        return dict(off=np.array(off, 'i8'), **{k: np.concatenate(v) for k, v in rows.items()})

    z = cached(folder, key, build, compressed=True)
    # Compute only the candidate aerosol rays; cache them per aerosol profile and share across optics.
    built = []
    started = time.monotonic()

    def build_aerosol():
        built.append(True)
        ps = []
        ds = []
        for cell in range(g.ns):
            lo, hi = z['off'][cell : cell + 2]
            loc = z['loc'][lo:hi][z['candidate'][lo:hi]]
            ih, ic = divmod(cell, len(g.chi))
            p, up, ex = old.probe.frame(np.array(b.R + g.h[ih]), g.chi[ic])
            ey = np.cross(up, ex)
            d = loc[:, 0, None] * ex + loc[:, 1, None] * ey + loc[:, 2, None] * up
            ps.append(np.broadcast_to(p, d.shape))
            ds.append(d)
        from .ground_parallel import aerosol

        return dict(columns=aerosol(scope, g.sun, np.concatenate(ps), np.concatenate(ds)))

    col1 = np.zeros(len(z['loc']))
    col1[z['candidate']] = cached(folder, aerosol_key, build_aerosol)['columns']
    from .column_parallel import event_fields

    print(
        json.dumps(
            dict(
                event='PHASE',
                phase='host_cache',
                **event_fields(),
                kind='ground_aerosol',
                result='miss' if built else 'hit',
                seconds=time.monotonic() - started,
            )
        ),
        flush=True,
    )
    ds = []
    vs = []
    off = [0]
    s.gm = np.zeros((g.ns, 4, s.W), 'f4')
    for cell in range(g.ns):
        lo, hi = z['off'][cell : cell + 2]
        loc = z['loc'][lo:hi]
        w = z['w'][lo:hi]
        q = z['q'][lo:hi]
        r = z['r'][lo:hi]
        col = np.empty((hi - lo, 5))
        col[:, [0, 2, 3, 4]] = z['columns'][lo:hi]
        col[:, 1] = col1[lo:hi]
        sc, lit = g.sun.lookup(q, np.maximum(b.R, r))
        ch = np.arccos(np.clip(q[:, 2] / r, -1, 1))
        value = (
            b.sh.NORMAL['albedo']
            / np.pi
            * np.maximum(0, np.cos(ch))[:, None]
            * np.exp(-(sc + col) @ s.ext)
            * lit[:, None]
            * (r < b.R + 1e-5)[:, None]
            * w[:, None]
        )
        take = np.any(value > 0, axis=1)
        ds.append(loc[take])
        vs.append(value[take])
        off.append(off[-1] + take.sum())
        factor = np.stack(
            [(1 - loc[:, 0] ** 2) / 2, (1 - loc[:, 1] ** 2) / 2, (1 - loc[:, 2] ** 2) / 2, -loc[:, 0] * loc[:, 2] / 2],
            -1,
        )
        s.gm[cell] = (factor.T @ value) * (2 * scope.C * s.ext[0])[None, :]
    s.goff = np.array(off, 'i8')
    s.gdir = np.ascontiguousarray(np.concatenate(ds), 'f4')
    s.gval = np.ascontiguousarray(np.concatenate(vs), 'f4')
