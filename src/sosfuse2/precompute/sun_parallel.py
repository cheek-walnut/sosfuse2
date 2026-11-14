"""Independent solar-table rows; preserve the accepted per-row arithmetic."""

import json
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import column_parallel as cp
from .column_parallel import default_workers, event_fields, freeze, thaw

ENV = 'SOSFUSE2_SUN_WORKERS'
_STATE = {}


def workers():
    text = os.environ.get(ENV, '').strip()
    if not text:
        return default_workers()
    try:
        value = int(text)
        if value > 0:
            return value
    except ValueError:
        pass
    raise ValueError('E_SUN_WORKERS: expected positive integer')


def _init(frozen, h, x, radius, n, optics, method, sources):
    _STATE['source_drift'] = cp.source_drift(sources)
    if _STATE['source_drift'] is not None:
        return
    _STATE.update(old=thaw(frozen), h=h, x=x, radius=radius, n=n, optics=optics, method=method)


def _row(old, h, x, radius, n, base, optics, method='gl'):
    r = radius + h
    mg = -np.sqrt(max(0, 1 - (radius / r) ** 2))
    mu = mg + (1 - mg) * x**2
    mu[0] = mg + 1e-9
    p = np.column_stack([np.zeros(len(mu)), np.zeros(len(mu)), np.full(len(mu), r)])
    d = np.column_stack([np.sqrt(np.maximum(1 - mu * mu, 0)), np.zeros(len(mu)), mu])
    from ..rays.ray_columns import integrate, replace

    if optics is not None and method == 'exact':
        # density.sun_method exact (T3): same rays and probe end; only the selected PROFILE columns of base change.
        from ..atmosphere.density import exact_profile_columns

        return replace(base, exact_profile_columns(p, d, old.probe.boundaries(p, d)[:, -1], optics), optics)
    # Endpoint totals do not consume geometry_segments' optical prefixes.
    value, corrected, _ = integrate(p, d, old, n, legacy=optics is None, optics=optics)
    return value if optics is None else replace(base, corrected, optics)


def _run(task):
    cp.check_source(_STATE["source_drift"])
    i, base = task
    w = _STATE
    return i, _row(w['old'], w['h'][i], w['x'], w['radius'], w['n'], base, w['optics'], w['method'])


def rows(old, h, x, radius, n, base=None, optics=None, method=None):
    """On failure discard all partial output and rerun every row serially.

    method: density.sun_method() value for the corrected (optics) rows, resolved here once when omitted and shipped
    to every worker and the serial fallback; ignored for the base rows (optics None)."""
    from ..atmosphere.density import SUN_METHODS, sun_method

    method = 'gl' if optics is None else sun_method() if method is None else method
    if method not in SUN_METHODS:
        raise ValueError('E_SUN_METHOD: expected gl or exact')
    count = workers()
    out = np.empty((len(h), len(x), 5)) if base is None else base.copy()
    if count > 1:
        started = time.monotonic()
        result, reason = 'parallel', None
        try:
            frozen = freeze(old, ['profiles', 'probe'])
            parameter = None if optics is None else {k: v for k, v in optics.items() if k != 'old'}
            with ProcessPoolExecutor(
                min(count, len(h)),
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_init,
                initargs=(frozen, h, x, radius, n, parameter, method, cp.SOURCE_HASHES),
            ) as pool:
                tasks = ((i, None if base is None else base[i]) for i in range(len(h)))
                for i, value in pool.map(_run, tasks):
                    out[i] = value
        except Exception as exc:
            result = 'fallback'
            reason = str(exc).split(':')[0] if str(exc).startswith('E_') else type(exc).__name__
        print(
            json.dumps(
                dict(
                    event='PHASE',
                    phase='sun_parallel',
                    **event_fields(),
                    kind='base' if optics is None else 'density',
                    workers=count,
                    result=result,
                    reason=reason,
                    seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )
        if result == 'parallel':
            return out
    for i, height in enumerate(h):
        out[i] = _row(old, height, x, radius, n, None if base is None else base[i], optics, method)
    return out


def prebuild(old, scale, folder):
    """Publish the same NPZ arrays before the unchanged SunGeometry reads them."""
    path = folder / 'sun_columns.npz'
    ident = old.SunGeometry.identity(scale)
    if path.exists():
        stored = old.SunGeometry.stored_identity(path)
        if stored == ident:
            return
        import warnings  # a table without (or with another) identity is rebuilt, never reused (solver_columns.SunGeometry)

        warnings.warn(
            f"SUN_TABLE_STALE: {path}: {'no identity' if stored is None else 'identity differs'}; rebuilding",
            RuntimeWarning,
        )
    # Also with one worker (rows() then runs serially): the accepted SunGeometry writes its NPZ in place, so an
    # interrupted serial build would leave a partial table at the final path.
    started = time.perf_counter()
    h = np.linspace(0, np.sqrt(old.Z), 80 * scale + 1) ** 2
    x = np.linspace(0, 1, 400 * scale + 1)
    col = rows(old, h, x, old.R, 32 * scale)
    fd, name = tempfile.mkstemp(prefix='sun_columns.', suffix='.tmp', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, h=h, x=x, col=col, identity=np.array(ident))
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return time.perf_counter() - started
