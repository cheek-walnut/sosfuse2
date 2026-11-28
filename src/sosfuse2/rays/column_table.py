"""Spherical endpoint columns (main rays), f64 until the existing f4 boundary.

SOSFUSE2_COLUMN_METHOD=table-v1 (default) | gl24 (Gauss-Legendre path) | exact. Only main-ray callers
select this method; solar and first-ground quadratures are unchanged.
Tables are independent of optical coefficients and are cached per profile.
"""

from collections import OrderedDict
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np
from ..foundation import content_identity as source_identity, dependencies, paths
from ..foundation.top import TOP_KM, COLUMN_TOP_MARGIN_KM, Domain
from ..atmosphere.components import MOLECULAR_COLUMNS

R, Z = 6371.0, TOP_KM
NA, NX = 1024, 1024
MAPPING = 'sqrt-mu-sqrt-height-1'
BOUNDARY = f'stable-tangent-short1m-corner005-ht02-top{Z - COLUMN_TOP_MARGIN_KM:g}-1'
# A cell can use four legacy profiles, four corrected profiles, and stratosphere.
_MEMO_MAX_PROFILES = 9
_MEMO = OrderedDict()


def method():
    value = os.environ.get('SOSFUSE2_COLUMN_METHOD', 'table-v1').strip()
    if value not in ('gl24', 'table-v1', 'exact'):
        raise ValueError('E_COLUMN_METHOD: expected gl24, table-v1 or exact')
    return value


@lru_cache(None)
def source_hash():
    h = hashlib.sha256()
    for name in dependencies.files('column_integral'):
        path = paths.PKG / name
        h.update(path.name.encode())
        h.update(source_identity.file_sha(path).encode())
    return h.hexdigest()


def identity():
    from ..atmosphere.density import host_native_source_hash

    sources = bytes.fromhex(source_hash()) + bytes.fromhex(host_native_source_hash())
    return dict(method=method(), mapping=MAPPING, boundary=BOUNDARY, na=NA, nx=NX,
                source=hashlib.sha256(sources).hexdigest())


def descriptor(kind, **parameters):
    return dict(kind=kind, **parameters)


def aerosol(old):
    explicit = getattr(old, 'aerosol_descriptor', None)
    if explicit is not None:
        return explicit
    raise ValueError('E_COLUMN_AER_DESCRIPTOR: missing aerosol_descriptor')


def molecular(optics, k):
    spec = _molecular(optics, k)
    if optics is not None and 'domain' in optics:
        from ..atmosphere.density import domain_of

        domain = domain_of(optics)
        spec.update(domain=domain.identity(), end_km=domain.air_end if k == 0 else domain.lower_end)
    return spec


def _molecular(optics, k):
    if k == 2:
        from ..atmosphere.density import LEGACY, build

        if optics is None:
            optics = build(dict(density=dict(LEGACY)))
        from ..atmosphere.density import profile_identity

        ident = profile_identity(optics)
        return descriptor(
            'usstd76',
            component='no2',
            values=optics['table'][:, 2].tolist(),
            grid=dict(n=ident['n'], step_km=ident['step_km'], top_km=ident['top_km'], interpolation='linear'),
            **optics['meta']['no2'],
        )
    if k == 3:
        air = molecular(optics, 0)
        if air['kind'] == 'exp':
            return descriptor('exp', height=4.0)
        surface = air['values'][0]
        return dict(
            air,
            kind='air_squared',
            component='o4',
            values=(np.asarray(air['values']) / surface).tolist(),
            air_surface=surface,
            interpolation='square of linear air',
        )
    if optics is None or not optics['mask'] & (1 << k):
        return descriptor('exp', height=8.0) if k == 0 else descriptor('triangle')
    # 'usstd76' = piecewise-linear PROFILE column; grid and policy from the effective table, so the
    # table key separates node counts, origins (asset/external) and normalizations as well as values.
    from ..atmosphere.density import profile_identity

    ident = profile_identity(optics)
    return descriptor(
        'usstd76',
        asset=optics.get('meta', {}).get('asset_sha256'),
        component=k,
        values=np.asarray(optics['table'][:, k]).tolist(),
        grid=dict(n=ident['n'], step_km=ident['step_km'], top_km=ident['top_km'], interpolation='linear'),
        origin=ident['origin'],
        normalization='effective air values' if k == 0 else 'effective ozone values',
    )


def strato(profile):
    result = descriptor('gaussian', **{k: float(profile[k]) for k in ('center', 'sigma', 'norm')})
    result.update({k: profile[k] for k in ('end_km', 'domain') if k in profile})
    return result


def values(h, spec):
    result = _values(h, spec)
    return np.where(np.asarray(h) > spec['end_km'], 0, result) if 'end_km' in spec else result


def _values(h, spec):
    kind = spec['kind']
    if kind == 'exp':
        return np.exp(-np.maximum(h, 0) / spec['height'])
    if kind == 'triangle':
        return np.maximum(0, 1 - np.abs(h - 25) / 15)
    if kind in ('usstd76', 'air_squared'):
        from ..atmosphere.density import grid

        g = spec.get('grid', {})
        q = np.interp(h, grid(len(spec['values']), step=g.get('step_km'), top_km=g.get('top_km')), spec['values'])
        return q * q if kind == 'air_squared' else q
    if kind == 'gaussian':
        return np.exp(-0.5 * ((h - spec['center']) / spec['sigma']) ** 2) / spec['norm']
    if kind == 'two_layer':
        from ..atmosphere.tropo_profile import profile

        return profile(h, spec['vector'])
    raise ValueError('E_COLUMN_PROFILE: ' + kind)


def _plain_profile(spec):
    """Array and scalar providers can have observable per-row conversion semantics."""
    if type(spec) is not dict:
        return False
    scalars = (float, int, np.float64, np.float32, np.int64, np.int32)
    if any(type(spec[k]) not in scalars for k in ('end_km', 'height', 'center', 'sigma', 'norm') if k in spec):
        return False
    for key in ('grid', 'domain'):
        if key in spec and type(spec[key]) is not dict:
            return False
    for key in ('values', 'vector'):
        if key not in spec:
            continue
        value = spec[key]
        if type(value) is np.ndarray and value.ndim == 1 and value.dtype.kind in 'fiu':
            continue
        if type(value) not in (list, tuple) or any(type(x) not in scalars for x in value):
            return False
    return True


def build_table(spec, ground):
    ztop = spec.get('domain', {}).get('top_km', Z)
    a = np.linspace(0, 1, NA)
    x = np.linspace(0, 1, NX)
    u, w = np.polynomial.legendre.leggauss(8)
    u = (u + 1) / 2
    w = w / 2
    rho = R * np.sqrt(1 - a**4) if ground else R + ztop * a * a
    hlo = np.zeros(NA) if ground else rho - R
    G = np.zeros((NA, NX), 'f8')
    from . import shell_segments
    from ..atmosphere import density, tropo_profile

    # A callback can consume rows in sequence, or depend on the 2-D node array it receives.
    blocked = (values is _DEFAULT_VALUES and _values is _DEFAULT_PROFILE_VALUES
               and density._standard_grid() and tropo_profile.profile is tropo_profile._TWO_LAYER_PROFILE
               and shell_segments._standard_numpy(np) and np is density.np is tropo_profile.np
               and np.polynomial.legendre.leggauss is shell_segments._LEGGAUSS and _plain_profile(spec))
    if (values is _DEFAULT_VALUES and _values is _DEFAULT_PROFILE_VALUES and density._standard_grid()
            and type(spec) is dict and spec.get('kind') in ('usstd76', 'air_squared')):
        data, grid = spec['values'], spec.get('grid', {})
        if type(grid) is dict and type(data) in (np.ndarray, tuple, list):
            if type(data) is np.ndarray and data.ndim != 1 or grid.get('n', len(data)) != len(data):
                raise ValueError('E_PROFILE_GRID: expected one value per profile node')
    if blocked and _fill_table(G, spec, ground, a, x, u, w, rho, hlo, ztop):
        return G
    for i in range(NA):
        vmax = np.sqrt(max(ztop - hlo[i], 0))
        dv = vmax / (NX - 1)
        v = x[:-1, None] * vmax + dv * u
        widths = dv
        if spec.get('end_km', ztop) < ztop:
            support = np.sqrt(max(spec['end_km'] - hlo[i], 0))
            left = np.minimum(x[:-1] * vmax, support)
            right = np.minimum(x[1:] * vmax, support)
            widths = (right - left)[:, None]
            v = left[:, None] + widths * u
        h = hlo[i] + v * v
        r = R + h
        if ground:
            dtdv = 2 * r * v / np.sqrt(v * v * (r + R) + R * R * a[i] ** 4)
        else:
            dtdv = 2 * r / np.sqrt(r + rho[i])
        G[i, 1:] = np.cumsum(np.sum(values(h, spec) * dtdv * (w * widths), axis=1))
    return G


def _fill_table(out, spec, ground, a, x, u, w, rho, hlo, ztop):
    """Integrate plain profiles, using the same grid and GL8 rule as the row loop."""
    from ..atmosphere import density
    from .ray_columns import _host_loops

    end = spec.get('end_km', np.inf)
    if (len(a) < 2 or len(x) < 2 or u.shape != (8,) or w.shape != (8,)
            or not np.isfinite(ztop) or ztop <= 0 or not np.isfinite(R) or R <= 0
            or np.isnan(end) or end <= 0):
        return False
    loops = _host_loops()
    if loops is None:
        return False
    kinds = ('exp', 'triangle', 'usstd76', 'air_squared', 'gaussian', 'two_layer')
    name = spec['kind']
    if name not in kinds:
        return False
    params = vals = np.empty(0)
    step = 1.
    if name == 'exp':
        params = np.array([spec['height']], 'f8')
        if params[0] <= 0:
            return False
    elif name == 'gaussian':
        params = np.array([spec[k] for k in ('center', 'sigma', 'norm')], 'f8')
        if params[1] <= 0 or params[2] <= 0:
            return False
    elif name == 'two_layer':
        params = np.require(spec['vector'], dtype='f8', requirements=['C', 'A'])
    elif name in ('usstd76', 'air_squared'):
        vals = np.require(spec['values'], dtype='f8', requirements=['C', 'A'])
        grid = spec.get('grid', {})
        z = density.grid(len(vals), step=grid.get('step_km'), top_km=grid.get('top_km'))
        step = z[1]
    kind = kinds.index(name)
    density._host_buffer(vals, (len(vals),))
    density._host_buffer(params, ((1, 0, 0, 0, 3, 7)[kind],))
    impact2 = np.array([R * R * angle**4 for angle in a])
    density._host_buffer(x, (len(x),))
    for array in (u, w):
        density._host_buffer(array, (8,))
    for array in (impact2, rho, hlo):
        density._host_buffer(array, (len(a),))
    density._host_buffer(out, (len(a), len(x)), writable=True)
    workspace_shape = (2, len(x) - 1, 8) if ground else (0,)
    workspace = density._host_buffer(np.empty(workspace_shape), workspace_shape, writable=True)
    loops.build_table_c(len(a), len(x), ground, R, ztop, end, kind, params, vals, len(vals), step,
                        x, u, w, impact2, rho, hlo, workspace, out)
    return True


def table_key(spec):
    return dict(**identity(), profile=spec)


def tables(spec):
    from ..precompute import host_cache

    key = table_key(spec)
    token = host_cache.digest(key)
    root = os.environ.get('SOSFUSE2_COLUMN_TABLE_CACHE') or os.environ.get('SOSFUSE2_COLUMN_CACHE')
    memo_key = (root, token)
    if memo_key in _MEMO:
        _MEMO.move_to_end(memo_key)
        return _MEMO[memo_key]

    def build():
        return dict(ground=build_table(spec, True), tangent=build_table(spec, False))

    if root:
        try:
            result = host_cache.cached(Path(root) / 'profile_tables', key, build)
        except host_cache.CacheUnavailable:
            result = build()  # host_cache emits the stable failure reason.
    else:
        result = build()
    for a in result.values():
        if a.shape != (NA, NX) or a.dtype != np.dtype('f8') or not np.isfinite(a).all():
            raise ValueError('E_COLUMN_TABLE_ARRAY')
        a.flags.writeable = False
    _MEMO[memo_key] = result
    while len(_MEMO) > _MEMO_MAX_PROFILES:
        _MEMO.popitem(last=False)
    return result


def cubic_weights(t, n):
    s = np.clip(t, 0, 1) * (n - 1)
    i = np.clip(np.floor(s).astype('i8') - 1, 0, n - 4)
    q = s - i
    return i, np.stack(
        [
            -(q - 1) * (q - 2) * (q - 3) / 6,
            q * (q - 2) * (q - 3) / 2,
            -q * (q - 1) * (q - 3) / 2,
            q * (q - 1) * (q - 2) / 6,
        ],
        -1,
    )


def lookup(G, a, x):
    ia, wa = cubic_weights(a, len(G))
    ix, wx = cubic_weights(x, G.shape[1])
    out = np.zeros(len(a))
    for j in range(4):
        for k in range(4):
            out += wa[:, j] * wx[:, k] * G[ia + j, ix + k]
    return out


def coordinates(p, d, end, top_km=Z):
    p = np.broadcast_to(p, d.shape)
    sa = np.sum(p * d, axis=-1)
    sb = sa + end
    rho = np.sqrt(np.maximum(np.sum(p * p, axis=-1) - sa * sa, 0))
    ground = rho < R
    ht = rho - R
    ra = np.hypot(rho, sa)
    rb = np.hypot(rho, sb)
    a = np.where(ground, np.maximum(1 - (rho / R) ** 2, 0) ** 0.25, np.sqrt(np.maximum(ht, 0) / top_km))
    # No subtraction r-rho in tangent coordinates, including the ht=0.2 seam.
    dena = np.sqrt(np.maximum((ra + rho) * (top_km - ht), np.finfo(float).tiny))
    denb = np.sqrt(np.maximum((rb + rho) * (top_km - ht), np.finfo(float).tiny))
    xa = np.where(ground, np.sqrt(np.maximum(ra - R, 0) / top_km), np.abs(sa) / dena)
    xb = np.where(ground, np.sqrt(np.maximum(rb - R, 0) / top_km), np.abs(sb) / denb)
    corner = ground & (a < 0.05) & (np.minimum(xa, xb) < 0.05)
    tangent = ~ground & (ht < 0.2) & (np.minimum(xa, xb) < 0.05)
    direct = corner | tangent | (~ground & (ht > top_km - COLUMN_TOP_MARGIN_KM))
    short = end < 1e-3
    return dict(sa=sa, sb=sb, rho=rho, ground=ground, a=a, xa=xa, xb=xb, direct=direct & ~short, short=short)


def _linear_knots(col, spec, step):
    if 'end_km' in spec:
        return col[:round(spec['end_km'] / step) + 1]
    return col


def pieces(spec):
    if spec['kind'] == 'triangle':
        return [(R + 10, R + 25, -(R + 10) / 15, 1 / 15), (R + 25, R + 40, (R + 40) / 15, -1 / 15)]
    from ..atmosphere.density import STEP_OF_N

    col = np.asarray(spec['values'])
    step = spec['grid']['step_km'] if 'grid' in spec else STEP_OF_N[len(col)]
    col = _linear_knots(col, spec, step)
    slope = np.diff(col) / step
    lo = R + np.arange(len(col) - 1) * step
    return zip(lo, lo + step, col[:-1] - slope * lo, slope)


_DEFAULT_PIECES = pieces
_DEFAULT_VALUES = values
_DEFAULT_PROFILE_VALUES = _values


def linear_columns(spec, rho, sa, sb):
    """Closed shell integrals, including both sides of the tangent."""

    from . import ray_columns, shell_segments
    from ..atmosphere.density import _host_buffer

    if (pieces is _DEFAULT_PIECES and _plain_profile(spec) and shell_segments._standard_numpy(np)
            and _ray_vectors(rho, sa, sb)):
        lib = ray_columns._host_loops()
        if lib is not None:
            coeff = np.asarray(list(pieces(spec)), 'f8').reshape(-1, 4)
            if not np.isfinite(coeff).all():
                lib = None
        if lib is not None:
            out = np.zeros(len(rho))
            arrays = [_host_buffer(v, rho.shape) for v in _buffers(rho, sa, sb)]
            lib.linear_columns_c(len(rho), len(coeff), _host_buffer(coeff, (len(coeff), 4)), *arrays,
                                 _host_buffer(out, rho.shape, writable=True))
            return out

    def primitive(s, r):
        return 0.5 * (s * np.hypot(r, s) + r * r * np.arcsinh(s / np.maximum(r, 1e-100)))

    out = np.zeros(len(rho))
    for lo, hi, alpha, beta in pieces(spec):
        ca = np.sqrt(np.maximum((lo - rho) * (lo + rho), 0))
        cb = np.sqrt(np.maximum((hi - rho) * (hi + rho), 0))
        for u, v in ((ca, cb), (-cb, -ca)):
            a = np.maximum(sa, u)
            b = np.minimum(sb, v)
            m = b > a
            if m.any():
                out[m] += alpha * (b[m] - a[m]) + beta * (primitive(b[m], rho[m]) - primitive(a[m], rho[m]))
    return np.maximum(out, 0)


def _buffers(*arrays):
    return [np.require(v, dtype='f8', requirements=['C', 'A']) for v in arrays]


def _ray_vectors(rho, sa, sb):
    return (type(rho) is np.ndarray and rho.ndim == 1
            and all(type(v) is np.ndarray and v.shape == rho.shape and v.dtype == np.float64
                    and np.isfinite(v).all() for v in (rho, sa, sb)))


@lru_cache(maxsize=1)
def _default_gl8():
    return np.polynomial.legendre.leggauss(8)


def _shell_rule():
    from . import shell_segments as shell

    if np.polynomial.legendre.leggauss is shell._LEGGAUSS and shell._standard_numpy(np):
        return _default_gl8()
    return np.polynomial.legendre.leggauss(8)


def _squared_walk(loops, spec, z, u, w, rho, sa, sb, linear=None):
    from ..atmosphere.density import _host_buffer

    col = np.asarray(spec['values'])
    if (z.ndim != 1 or rho.ndim != 1 or col.ndim != 1 or col.dtype != np.float64 or len(z) > len(col) or not np.isfinite(col).all()
            or u.shape != (8,) or w.shape != (8,)
            or (len(z) > 1 and (z[0] != 0 or z[1] <= 0 or not np.all(np.diff(z) == z[1])))):
        return None
    out = np.zeros(len(rho))
    aux = np.zeros_like(out) if linear is not None else np.empty(0)
    alpha = beta = np.empty(0)
    if linear is not None:
        if (linear['kind'] != 'usstd76' or linear.get('grid') != spec.get('grid')
                or linear.get('end_km') != spec.get('end_km') or len(linear['values']) != len(col)):
            return None
        step = linear['grid']['step_km']
        vals = _linear_knots(np.asarray(linear['values'], 'f8'), linear, step)
        # The rounded linear endpoint can include a shell beyond the last O4 knot.
        if len(vals) != len(z) or not np.isfinite(vals).all():
            return None
        beta = np.diff(vals) / step
        alpha = vals[:-1] - beta * (R + z[:-1])
    arrays = _buffers(z, col, u, w, rho, sa, sb)
    for array, shape in zip(arrays, ((len(z),), (len(col),), (8,), (8,), (len(rho),), (len(rho),), (len(rho),))):
        _host_buffer(array, shape)
    coefficients = (len(z) - 1,) if linear is not None else (0,)
    _host_buffer(alpha, coefficients)
    _host_buffer(beta, coefficients)
    _host_buffer(out, rho.shape, writable=True)
    _host_buffer(aux, rho.shape if linear is not None else (0,), writable=True)
    loops.squared_columns(len(rho), len(z), len(col), *arrays, R, spec.get('end_km', np.inf),
                          out, linear is not None, alpha, beta, aux)
    return out, aux


def direct_columns(spec, p, d, end, n=24):
    """Boundary fallback: exact US1976/triangle; split GL for smooth profiles."""
    p = np.broadcast_to(p, d.shape)
    c = coordinates(p, d, end, spec.get('domain', {}).get('top_km', Z))
    if spec['kind'] in ('usstd76', 'triangle'):
        return linear_columns(spec, c['rho'], c['sa'], c['sb'])
    if spec['kind'] == 'air_squared':
        # Split at every air knot before quadrature: integrate the square of the
        # interpolant, never the interpolant of squared knots. Using local h in
        # the integrand avoids cancellation of Earth-radius-sized primitives.
        from ..atmosphere import density
        from . import ray_columns, shell_segments

        out = np.zeros(len(d))
        u, w = _shell_rule()
        g = spec.get('grid', {})
        z = density.grid(len(spec['values']), step=g.get('step_km'), top_km=g.get('top_km'))
        if 'end_km' in spec:
            z = z[z <= spec['end_km']]
        if (values is _DEFAULT_VALUES and _values is _DEFAULT_PROFILE_VALUES and density._standard_grid()
                and density.np is np and shell_segments._standard_numpy(np)
                and np.polynomial.legendre.leggauss is shell_segments._LEGGAUSS
                and _plain_profile(spec) and _ray_vectors(c['rho'], c['sa'], c['sb'])):
            loops = ray_columns._host_loops()
            if loops is not None:
                result = _squared_walk(loops, spec, z, u, w, c['rho'], c['sa'], c['sb'])
                if result is not None:
                    return result[0]
        for lo, hi in zip(z[:-1], z[1:]):
            ca = np.sqrt(np.maximum((R + lo - c['rho']) * (R + lo + c['rho']), 0))
            cb = np.sqrt(np.maximum((R + hi - c['rho']) * (R + hi + c['rho']), 0))
            for left, right in ((ca, cb), (-cb, -ca)):
                a = np.maximum(c['sa'], left)
                b = np.minimum(c['sb'], right)
                m = b > a
                if not m.any():
                    continue
                s = (a[m, None] + b[m, None]) / 2 + (b - a)[m, None] * u / 2
                rho = c['rho'][m, None]
                h = rho - R + s * s / (np.hypot(rho, s) + rho)
                out[m] += (b - a)[m] * np.sum(values(h, spec) * w, axis=1) / 2
        return out
    cuts = [np.zeros(len(d)), end, np.clip(-c['sa'], 0, end)]
    heights = [10.0, 25.0, 40.0]
    if 'end_km' in spec:
        heights.append(spec['end_km'])
    if spec['kind'] == 'two_layer':
        v = spec['vector']
        heights += [v[0], v[0] + 1 / v[1]]
    for h in heights:
        root = np.sqrt(np.maximum((R + h - c['rho']) * (R + h + c['rho']), 0))
        for sign in (-1, 1):
            cuts.append(np.clip(-c['sa'] + sign * root, 0, end))
    edges = np.sort(np.column_stack(cuts), axis=1)
    u, w = np.polynomial.legendre.leggauss(n)
    u = (u + 1) / 2
    w = w / 2
    out = np.zeros(len(d))
    for k in range(edges.shape[1] - 1):
        a, b = edges[:, k], edges[:, k + 1]
        m = b > a
        if not m.any():
            continue
        t = a[m, None] + (b - a)[m, None] * u
        # r-rho = s²/(r+rho) keeps sub-mm tangent altitude differences.
        s = c['sa'][m, None] + t
        rho = c['rho'][m, None]
        h = (rho - R) + s * s / (np.hypot(rho, s) + rho)
        out[m] += np.sum(values(h, spec) * w, axis=1) * (b - a)[m]
    return np.maximum(out, 0)


def exact_columns(spec, p, d, end):
    """Gate B reference: every ray closed form (US1976/triangle) or split GL64, no table, no midpoint rule."""
    p = np.broadcast_to(p, d.shape)
    if spec['kind'] in ('usstd76', 'triangle'):
        sa = np.sum(p * d, axis=-1)
        rho = np.sqrt(np.maximum(np.sum(p * p, axis=-1) - sa * sa, 0))
        out = linear_columns(spec, rho, sa, sa + end)
    else:
        out = direct_columns(spec, p, d, end, 64)
    if not np.isfinite(out).all():
        raise ValueError('E_COLUMN_TABLE_NONFINITE')
    return np.maximum(out, 0)


# The paired walk specializes only this chain of ordinary component calls.
# A replacement may carry state, so the caller keeps the original per-component loop.
_EXACT_BINDINGS = tuple((name, globals()[name]) for name in (
    'molecular', '_molecular', 'descriptor', 'exact_columns', 'linear_columns', 'pieces', '_linear_knots',
    'direct_columns', 'coordinates', 'values', '_values',
))


def _standard_minor_optics(optics):
    from ..atmosphere import density
    from . import shell_segments as shell

    return (type(optics) is dict and type(optics.get('table')) is np.ndarray
            and type(optics.get('meta')) is dict and type(optics['meta'].get('no2')) is dict
            and all(globals()[name] is fn for name, fn in _EXACT_BINDINGS)
            and density._standard_descriptors() and shell._standard_numpy(np) and np is density.np
            and np.polynomial.legendre.leggauss is shell._LEGGAUSS
            and shell._standard_domain(density.domain_of(optics)))


def _minor_pair(optics, rho, sa, sb, describe, integrate):
    from ..atmosphere import density
    from . import ray_columns

    if (describe is not molecular or integrate is not exact_columns
            or not _standard_minor_optics(optics) or not _ray_vectors(rho, sa, sb)):
        return None
    loops = ray_columns._host_loops()
    if loops is None:
        return None
    linear = describe(optics, 2)
    square = describe(optics, 3)
    if square['kind'] != 'air_squared':
        return None
    g = square['grid']
    u, w = _shell_rule()
    z = density.grid(len(square['values']), step=g['step_km'], top_km=g['top_km'])
    if 'end_km' in square:
        z = z[z <= square['end_km']]
    return _squared_walk(loops, square, z, u, w, rho, sa, sb, linear)


def column(spec, p, d, end, coords=None):
    if method() == 'exact':
        return exact_columns(spec, p, d, end)
    c = coordinates(p, d, end, spec.get('domain', {}).get('top_km', Z)) if coords is None else coords
    out = np.zeros(len(d))
    short = c['short'].copy()
    direct = c['direct'].copy()
    outside = np.zeros(len(d), bool)
    if spec.get('end_km', Z) < spec.get('domain', {}).get('top_km', Z):
        support = R + spec['end_km']
        chord = np.sqrt(np.maximum((support - c['rho']) * (support + c['rho']), 0))
        outside = (c['sb'] <= -chord) | (c['sa'] >= chord) | (c['rho'] >= support)
        near = ((abs(np.hypot(c['rho'], c['sa']) - support) < COLUMN_TOP_MARGIN_KM)
                | (abs(np.hypot(c['rho'], c['sb']) - support) < COLUMN_TOP_MARGIN_KM)
                | (abs(c['rho'] - support) < COLUMN_TOP_MARGIN_KM))
        direct |= near
        direct &= ~outside
        short &= ~(near | outside)
    if short.any():
        s = c['sa'][short] + end[short] / 2
        rho = c['rho'][short]
        h = rho - R + s * s / (np.hypot(rho, s) + rho)
        out[short] = values(h, spec) * end[short]
    if spec['kind'] == 'triangle':
        # The same sub-metre midpoint rule precedes the analytic ozone path.
        m = ~short
        out[m] = linear_columns(spec, c['rho'][m], c['sa'][m], c['sb'][m])
        if not np.isfinite(out).all():
            raise ValueError('E_COLUMN_TABLE_NONFINITE')
        return np.maximum(out, 0)
    if direct.any():
        out[direct] = direct_columns(spec, np.broadcast_to(p, d.shape)[direct], d[direct], end[direct])
    use = ~(short | direct | outside)
    if use.any():
        G = tables(spec)
        for ground, name in ((True, 'ground'), (False, 'tangent')):
            m = use & (c['ground'] == ground)
            if not m.any():
                continue
            ga = lookup(G[name], c['a'][m], c['xa'][m])
            gb = lookup(G[name], c['a'][m], c['xb'][m])
            out[m] = np.where(c['sa'][m] >= 0, gb - ga, np.where(c['sb'][m] <= 0, ga - gb, ga + gb))
    if not np.isfinite(out).all():
        raise ValueError('E_COLUMN_TABLE_NONFINITE')
    return np.maximum(out, 0)


def lower_descriptor(spec, domain):
    if domain.top_km > domain.lower_end:
        return dict(spec, domain=domain.identity(), end_km=domain.lower_end)
    return spec


def integrate(p, d, old, *, legacy=False, optics=None, profile=None, edges=None):
    p = np.broadcast_to(p, d.shape)
    end = old.probe.boundaries(p, d)[:, -1] if edges is None else edges[:, -1]
    domain = getattr(old, 'domain', Domain())
    c = coordinates(p, d, end, domain.top_km)

    base = (
        np.column_stack(
            [
                column(lower_descriptor(spec, domain), p, d, end, c)
                for spec in (
                    molecular(None, 0),
                    aerosol(old),
                    molecular(None, 1),
                    molecular(None, 2),
                    molecular(None, 3),
                )
            ]
        )
        if legacy
        else None
    )
    corrected = (
        np.column_stack([column(molecular(optics, k), p, d, end, c) for k in range(4)]) if optics is not None else None
    )
    st = column(strato(profile), p, d, end, c) if profile is not None else None
    return base, corrected, st


def fill(out, kind, starts, cell_off, dirs, old, parameter):
    """Sidecars from cached endpoints; also handles one worker and tiny bundles."""
    for lo in range(0, len(dirs), 32768):
        hi = min(lo + 32768, len(dirs))
        owner = np.searchsorted(cell_off, np.arange(lo, hi), side='right') - 1
        p = starts[owner]
        d = dirs[lo:hi]
        end = old.probe.boundaries(p, d)[:, -1]
        c = coordinates(p, d, end, getattr(old, 'domain', Domain()).top_km)
        if kind == 'strato':
            out[lo:hi] = column(strato(parameter), p, d, end, c)
        else:
            for k, col in enumerate(MOLECULAR_COLUMNS):
                if k >= 2 or parameter['mask'] & (1 << k):
                    out[lo:hi, col] = column(molecular(parameter, k), p, d, end, c)
    return True
