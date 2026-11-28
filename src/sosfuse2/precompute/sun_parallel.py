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


_DEFAULT_ROW = _row


def _native_rows(old, h, x, radius, n, out, optics, method):
    """Integrate blocks of independent solar rays only for the built-in row calculation."""
    from ..atmosphere import density
    from ..rays import column_table, ray_columns, shell_segments

    if (_row is not _DEFAULT_ROW or method != 'gl' or not shell_segments._standard_numpy(np)
            or np is not ray_columns.np or np is not density.np or np is not column_table.np
            or ray_columns.integrate is not ray_columns._DEFAULT_INTEGRATE
            or ray_columns.Domain is not shell_segments._DOMAIN
            or ray_columns.segments is not ray_columns._DEFAULT_SEGMENTS
            or type(ray_columns.MOLECULAR_COLUMNS) is not tuple or ray_columns.NCOMP != 5
            or not shell_segments._standard_probe(old)
            or not shell_segments._standard_domain(getattr(old, 'domain', shell_segments.Domain()))
            or type(radius) not in (int, float, np.float64) or not np.isfinite(radius) or radius <= 0
            or type(n) not in (int, np.int64) or n <= 0
            or any(type(v) is not np.ndarray or v.dtype != np.float64 or v.ndim != 1
                   or not len(v) or not np.isfinite(v).all() for v in (h, x))
            or type(out) is not np.ndarray or out.shape != (len(h), len(x), 5)
            or np.any(h < 0) or np.any(x < 0) or np.any(x > 1)):
        return False
    if optics is None:
        if not shell_segments._standard_binding(getattr(old, 'profiles', None), shell_segments._PROFILES):
            return False
    elif (ray_columns.replace is not ray_columns._DEFAULT_REPLACE or not column_table._standard_minor_optics(optics)
          or density.exact_profile_columns is not density._DEFAULT_EXACT_PROFILE_COLUMNS
          or type(optics.get('mask')) is not int or not optics['mask'] & 1 or optics['table'].dtype != np.float64):
        return False
    lib = ray_columns._host_loops()
    if lib is None:
        return False
    h, x = (np.require(v, dtype='f8', requirements=['C', 'A']) for v in (h, x))
    density._host_buffer(x, (len(x),))
    # NumPy scalar power and vector multiplication can differ at the grazing boundary.
    grazing = np.array([-np.sqrt(max(0, 1 - (radius / (radius + height)) ** 2)) for height in h])
    # The probe can bind a domain independently of the extra-shell domain.
    domain = getattr(old, 'domain', shell_segments.Domain())
    bound = old.probe.boundaries
    boundary_domain = (None if bound is shell_segments.boundaries else bound.keywords.get('domain'))
    boundary_domain = boundary_domain or shell_segments.Domain()
    probe_radius = shell_segments.R
    extra_r2 = np.array([(6371.0 + height)**2 for height in domain.extra_shells], 'f8')
    density._host_buffer(extra_r2, (len(extra_r2),))
    # Bound the temporary ray, shell and column arrays independently of the height count.
    block = max(1, 16384 // len(x))
    for start in range(0, len(h), block):
        heights = h[start:start + block]
        shape = (len(heights) * len(x), 3)
        p, d = np.empty(shape), np.empty(shape)
        edge_shape = (shape[0], 5 + 2 * len(extra_r2))
        edges = np.empty(edge_shape)
        lib.sun_rays_c(len(heights), density._host_buffer(heights, (len(heights),)), len(x), x, radius,
                       density._host_buffer(grazing[start:start + len(heights)], heights.shape),
                       probe_radius * probe_radius, (probe_radius + boundary_domain.top_km)**2, len(extra_r2), extra_r2,
                       density._host_buffer(p, shape, writable=True), density._host_buffer(d, shape, writable=True),
                       density._host_buffer(edges, edge_shape, writable=True))
        base, corrected, _ = ray_columns.integrate(p, d, old, n, legacy=optics is None, optics=optics, edges=edges)
        target = out[start:start + len(heights)]
        if optics is not None:
            base = ray_columns.replace(target.reshape(-1, 5), corrected, optics)
        target[:] = base.reshape(target.shape)
    return True


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
    if _row is _DEFAULT_ROW:
        if any(type(v) is np.ndarray and v.ndim != 1 for v in (h, x)):
            raise ValueError('E_SUN_GRID: heights and solar coordinates must be vectors')
        if type(base) is np.ndarray and base.shape[:2] != (len(h), len(x)):
            raise ValueError('E_SUN_GRID: base column dimensions must match the solar coordinates')
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
    plain_base = (base is None and optics is None) or type(base) is np.ndarray
    if plain_base and _native_rows(old, h, x, radius, n, out, optics, method):
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
        import warnings  # a table without (or with another) identity is rebuilt, never reused (shell_segments.SunGeometry)

        warnings.warn(
            f"SUN_TABLE_STALE: {path}: {'no identity' if stored is None else 'identity differs'}; rebuilding",
            RuntimeWarning,
        )
    # Also with one worker (rows() then runs serially): the accepted SunGeometry writes its NPZ in place, so an
    # interrupted serial build would leave a partial table at the final path.
    started = time.perf_counter()
    from ..rays.shell_segments import sun_grid

    h, x = sun_grid(scale, old.Z)
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
