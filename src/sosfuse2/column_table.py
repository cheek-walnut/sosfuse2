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
from . import identity as source_identity
from .components import MOLECULAR_COLUMNS

R, Z = 6371.0, 80.0
NA, NX = 1024, 1024
MAPPING = 'sqrt-mu-sqrt-height-1'
BOUNDARY = 'stable-tangent-short1m-corner005-ht02-top79-1'
_MEMO = OrderedDict()


def method():
    value = os.environ.get('SOSFUSE2_COLUMN_METHOD', 'table-v1').strip()
    if value not in ('gl24', 'table-v1', 'exact'):
        raise ValueError('E_COLUMN_METHOD: expected gl24, table-v1 or exact')
    return value


@lru_cache(None)
def source_hash():
    h = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name('tropo_profile.py')):
        h.update(path.name.encode())
        h.update(source_identity.file_sha(path).encode())
    return h.hexdigest()


def identity():
    return dict(method=method(), mapping=MAPPING, boundary=BOUNDARY, na=NA, nx=NX, source=source_hash())


def descriptor(kind, **parameters):
    return dict(kind=kind, **parameters)


def aerosol(old):
    explicit = getattr(old, 'aerosol_descriptor', None)
    if explicit is not None:
        return explicit
    raise ValueError('E_COLUMN_AER_DESCRIPTOR: missing aerosol_descriptor')


def molecular(optics, k):
    if k == 2:
        from .density import build

        if optics is None:
            optics = build({})
        from .density import profile_identity

        ident = profile_identity(optics)
        return descriptor(
            'usstd76',
            component='no2',
            values=optics['table'][:, 2].tolist(),
            grid=dict(n=ident['n'], step_km=ident['step_km'], top_km=80.0, interpolation='linear'),
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
    from .density import profile_identity

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
    return descriptor('gaussian', **{k: float(profile[k]) for k in ('center', 'sigma', 'norm')})


def values(h, spec):
    kind = spec['kind']
    if kind == 'exp':
        return np.exp(-np.maximum(h, 0) / spec['height'])
    if kind == 'triangle':
        return np.maximum(0, 1 - np.abs(h - 25) / 15)
    if kind in ('usstd76', 'air_squared'):
        from .density import grid

        q = np.interp(h, grid(len(spec['values'])), spec['values'])
        return q * q if kind == 'air_squared' else q
    if kind == 'gaussian':
        return np.exp(-0.5 * ((h - spec['center']) / spec['sigma']) ** 2) / spec['norm']
    if kind == 'two_layer':
        from .tropo_profile import profile

        return profile(h, spec['vector'])
    raise ValueError('E_COLUMN_PROFILE: ' + kind)


def build_table(spec, ground):
    a = np.linspace(0, 1, NA)
    x = np.linspace(0, 1, NX)
    u, w = np.polynomial.legendre.leggauss(8)
    u = (u + 1) / 2
    w = w / 2
    rho = R * np.sqrt(1 - a**4) if ground else R + Z * a * a
    hlo = np.zeros(NA) if ground else rho - R
    G = np.zeros((NA, NX), 'f8')
    for i in range(NA):
        vmax = np.sqrt(Z - hlo[i])
        dv = vmax / (NX - 1)
        v = x[:-1, None] * vmax + dv * u
        h = hlo[i] + v * v
        r = R + h
        if ground:
            dtdv = 2 * r * v / np.sqrt(v * v * (r + R) + R * R * a[i] ** 4)
        else:
            dtdv = 2 * r / np.sqrt(r + rho[i])
        G[i, 1:] = np.cumsum(np.sum(values(h, spec) * dtdv * (w * dv), axis=1))
    return G


def table_key(spec):
    return dict(**identity(), profile=spec)


def tables(spec):
    from . import host_cache

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
    while len(_MEMO) > 8:
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


def coordinates(p, d, end):
    p = np.broadcast_to(p, d.shape)
    sa = np.sum(p * d, axis=-1)
    sb = sa + end
    rho = np.sqrt(np.maximum(np.sum(p * p, axis=-1) - sa * sa, 0))
    ground = rho < R
    ht = rho - R
    ra = np.hypot(rho, sa)
    rb = np.hypot(rho, sb)
    a = np.where(ground, np.maximum(1 - (rho / R) ** 2, 0) ** 0.25, np.sqrt(np.maximum(ht, 0) / Z))
    # No subtraction r-rho in tangent coordinates, including the ht=0.2 seam.
    dena = np.sqrt(np.maximum((ra + rho) * (Z - ht), np.finfo(float).tiny))
    denb = np.sqrt(np.maximum((rb + rho) * (Z - ht), np.finfo(float).tiny))
    xa = np.where(ground, np.sqrt(np.maximum(ra - R, 0) / Z), np.abs(sa) / dena)
    xb = np.where(ground, np.sqrt(np.maximum(rb - R, 0) / Z), np.abs(sb) / denb)
    corner = ground & (a < 0.05) & (np.minimum(xa, xb) < 0.05)
    tangent = ~ground & (ht < 0.2) & (np.minimum(xa, xb) < 0.05)
    direct = corner | tangent | (~ground & (ht > 79))
    short = end < 1e-3
    return dict(sa=sa, sb=sb, rho=rho, ground=ground, a=a, xa=xa, xb=xb, direct=direct & ~short, short=short)


def pieces(spec):
    if spec['kind'] == 'triangle':
        return [(R + 10, R + 25, -(R + 10) / 15, 1 / 15), (R + 25, R + 40, (R + 40) / 15, -1 / 15)]
    from .density import STEP_OF_N

    col = np.asarray(spec['values'])
    step = STEP_OF_N[len(col)]  # 161/321/641 nodes: N-1 shells of width step
    slope = np.diff(col) / step
    lo = R + np.arange(len(col) - 1) * step
    return zip(lo, lo + step, col[:-1] - slope * lo, slope)


def linear_columns(spec, rho, sa, sb):
    """Closed shell integrals, including both sides of the tangent."""

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


def direct_columns(spec, p, d, end, n=24):
    """Boundary fallback: exact US1976/triangle; split GL for smooth profiles."""
    p = np.broadcast_to(p, d.shape)
    c = coordinates(p, d, end)
    if spec['kind'] in ('usstd76', 'triangle'):
        return linear_columns(spec, c['rho'], c['sa'], c['sb'])
    if spec['kind'] == 'air_squared':
        # Split at every air knot before quadrature: integrate the square of the
        # interpolant, never the interpolant of squared knots. Using local h in
        # the integrand avoids cancellation of Earth-radius-sized primitives.
        from .density import grid

        out = np.zeros(len(d))
        u, w = np.polynomial.legendre.leggauss(8)
        z = grid(len(spec['values']))
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


def column(spec, p, d, end, coords=None):
    if method() == 'exact':
        return exact_columns(spec, p, d, end)
    c = coordinates(p, d, end) if coords is None else coords
    out = np.zeros(len(d))
    short = c['short']
    direct = c['direct']
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
    use = ~(short | direct)
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


def integrate(p, d, old, *, legacy=False, optics=None, profile=None, edges=None):
    p = np.broadcast_to(p, d.shape)
    end = old.probe.boundaries(p, d)[:, -1] if edges is None else edges[:, -1]
    c = coordinates(p, d, end)
    base = (
        np.column_stack(
            [
                column(spec, p, d, end, c)
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
        c = coordinates(p, d, end)
        if kind == 'strato':
            out[lo:hi] = column(strato(parameter), p, d, end, c)
        else:
            for k, col in enumerate(MOLECULAR_COLUMNS):
                if k >= 2 or parameter['mask'] & (1 << k):
                    out[lo:hi, col] = column(molecular(parameter, k), p, d, end, c)
    return True
