"""Disposable, content-addressed column arrays; no geometry/cache schema changes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import warnings
import zipfile
import numpy as np


def array_hash(value):
    # Same descriptor as host_cache.array_hash, without its full .tobytes() copy.
    a = np.ascontiguousarray(value)
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                sha256=hashlib.sha256(memoryview(a.reshape(-1).view('u1'))).hexdigest())


def cache_key(kernel, bundle, c, kind):
    """Key: immutable Bundle starts/dirs/cell_off bytes+dtype+shape (memoized);
    strato center/sigma/norm exact float repr; density end_col bytes+dtype+shape,
    asset SHA256, mask and optical table bytes; GL=24, chunk=2048, format version;
    integrator, geometry_segments, probe boundaries/GL and engine source SHA256.
    Tau, aod and albedo are excluded. A changed array requires a fresh Bundle.
    """
    from .stratosphere import columns
    from .density import correct_columns
    hashes = bundle.__dict__.setdefault('_column_array_hashes', {})
    names = ('starts', 'dirs', 'cell_off') + (('end_col',) if kind == 'density' else ())
    for name in names:
        if name not in hashes: hashes[name] = array_hash(kernel.host(bundle.arrays[name]))
    optics = c['density_optics'] if kind == 'density' else None
    old = optics['old'] if optics is not None else c['strato_old']
    functions = (correct_columns if optics is not None else columns,
                 old.geometry_segments, old.probe.boundaries, old.probe.gl)
    sources = {str(i): hashlib.sha256(Path(f.__code__.co_filename).read_bytes()).hexdigest()
               for i, f in enumerate(functions)}
    sources['joint_columns'] = hashlib.sha256(Path(__file__).with_name('ray_columns.py').read_bytes()).hexdigest()
    sources['engine'] = hashlib.sha256((Path(__file__).resolve().parent/'accepted/engine.py').read_bytes()).hexdigest()
    from .density import profile_identity   # ②b: grid N/step, normalization, origin with the table bytes
    parameters = (dict(asset=optics['meta'].get('asset_sha256'), mask=int(optics['mask']),
                       table=array_hash(optics['table']), profile=profile_identity(optics)['sha256']) if optics is not None else
                  {k: repr(float(c['strato_profile'][k])) for k in ('center', 'sigma', 'norm')})
    from .column_table import identity as column_identity
    return hashlib.sha256(json.dumps(dict(version=1, method=column_identity(), kind=kind, rays={k:hashes[k] for k in names},
        parameters=parameters, gl=24, chunk=2048, sources=sources), sort_keys=True).encode()).hexdigest()


def job_root():
    """SOSFUSE2_COLUMN_CACHE_JOB (run_gpu_state_v3: its --work-root); unset = no first-use comparison."""
    text = os.environ.get('SOSFUSE2_COLUMN_CACHE_JOB', '').strip()
    return Path(text) if text else None


def direct(build, shape):
    value = np.ascontiguousarray(build(), dtype='f4')
    if value.shape != shape or not np.isfinite(value).all(): raise ValueError('E_COLUMN_CACHE_BUILD')
    return value


def first_use(job, key, payload, value, build, shape, reference=None):
    """Called only under cached_columns' per-key flock, including the marker publication.

    One direct calculation per (key, payload SHA) and job, after strato-kernel-loop cycle17
    host_cache.checked_columns. Markers are job-local evidence about that exact payload, never cache
    content. A mismatch returns the direct bytes and disables the payload for the rest of the job;
    the shared file is left untouched for diagnosis. Returns (array, verify, result or None, reason).
    """
    marker = Path(job)/'.column_cache_checks'/(key+'.'+payload+'.json')
    status = None
    if reference is None:
        try: status = json.loads(marker.read_text()).get('status')
        except (FileNotFoundError, ValueError, TypeError, AttributeError): pass
    if status == 'match':
        return value, 'compare_skip', None, None
    if status == 'mismatch':
        return direct(build, shape), 'disabled', 'disabled', 'E_COLUMN_CACHE_MISMATCH'
    fresh = direct(build, shape) if reference is None else reference
    equal = value.shape == fresh.shape and value.dtype == fresh.dtype and value.tobytes() == fresh.tobytes()
    marker.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=marker.parent, suffix='.tmp', delete=False) as tmp:
        json.dump(dict(status='match' if equal else 'mismatch'), tmp)
    try: os.replace(tmp.name, marker)
    finally: Path(tmp.name).unlink(missing_ok=True)
    verify = 'first_compare' if reference is None else 'reload_compare'
    if equal: return value, verify, None, None
    warnings.warn('E_COLUMN_CACHE_MISMATCH: '+key[:16], RuntimeWarning)
    return fresh, verify, 'invalid', 'E_COLUMN_CACHE_MISMATCH'


def cached_columns(kernel, bundle, c, kind, build):
    started = time.monotonic(); key = ''; result = 'disabled'; reason = None; verify = None; payload = ''
    shape = (bundle.rays,) if kind == 'strato' else bundle.arrays['end_col'].shape
    folder = os.environ.get('SOSFUSE2_COLUMN_CACHE'); job = job_root()
    try:
        if not folder: return build()
        result = 'miss'
        key = cache_key(kernel, bundle, c, kind)
        root = Path(folder); root.mkdir(parents=True, exist_ok=True)
        path = root/(key+'.npy'); checksum = root/(key+'.sha256')
        with (root/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = 'miss'; loaded = None
            if path.exists():
                try:
                    value = np.load(path, allow_pickle=False)
                    if not isinstance(value, np.ndarray):
                        value.close(); raise ValueError('E_COLUMN_CACHE_FORMAT')
                    if value.shape != shape or value.dtype != np.dtype('f4') or not np.isfinite(value).all():
                        raise ValueError('E_COLUMN_CACHE_ARRAY')
                    payload = hashlib.sha256(path.read_bytes()).hexdigest()
                    if payload != checksum.read_text():
                        raise ValueError('E_COLUMN_CACHE_CHECKSUM')
                    loaded = value
                except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                    result = 'invalid'; reason = str(exc) if str(exc).startswith('E_COLUMN_CACHE_') else 'E_COLUMN_CACHE_READ'
            if loaded is not None:
                result = 'hit'
                if job is None: return loaded
                # Outside the read try: a build or marker failure is not a corrupt cache file.
                value, verify, override, why = first_use(job, key, payload, loaded, build, shape)
                result = override or result; reason = why or reason
                return value
            value = direct(build, shape)
            # The writer lock covers both publications; a crash between them is
            # detected as a checksum mismatch/missing sidecar on the next read.
            temps = []
            try:
                with tempfile.NamedTemporaryFile(dir=root, suffix='.tmp', delete=False) as tmp:
                    temps.append(Path(tmp.name)); np.save(tmp, value, allow_pickle=False)
                digest = hashlib.sha256(temps[0].read_bytes()).hexdigest()
                with tempfile.NamedTemporaryFile(mode='w', dir=root, suffix='.tmp', delete=False) as side:
                    temps.append(Path(side.name)); side.write(digest)
                os.replace(temps[1], checksum); os.replace(temps[0], path)
            finally:
                for tmp in temps: tmp.unlink(missing_ok=True)
            payload = digest
            if job is not None:
                # Reload what was published and compare it once with the direct bytes just built.
                try:
                    reloaded = np.load(path, allow_pickle=False)
                    if not isinstance(reloaded, np.ndarray): reloaded.close(); raise ValueError('E_COLUMN_CACHE_FORMAT')
                except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                    # Keep an earlier read reason (e.g. E_COLUMN_CACHE_CHECKSUM) next to this one.
                    verify = 'reload_failed'; result = 'invalid'
                    reason = 'E_COLUMN_CACHE_RELOAD' if reason is None else reason+';E_COLUMN_CACHE_RELOAD'
                    return value
                _, verify, override, why = first_use(job, key, payload, reloaded, build, shape, reference=value)
                result = override or result; reason = why or reason
            return value
    except OSError as exc:
        if not folder: raise
        reason = 'E_COLUMN_CACHE_IO'
        raise RuntimeError(reason) from exc
    finally:
        print(json.dumps(dict(event='PHASE', phase='column_cache', id=os.environ.get('SOSFUSE2_STATE_ID', 'COLUMN_CACHE'),
            pid=os.getpid(), timestamp_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
            kind=kind, result=result, key=key[:16], rays=bundle.rays,
            seconds=time.monotonic()-started, reason=reason, verify=verify, payload=payload[:16])), flush=True)
