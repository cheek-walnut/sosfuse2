"""Column integration and the sun-column table of the CPU reference engine; the aerosol profile is an argument."""

from functools import lru_cache, partial
from types import SimpleNamespace
import json
import time

import numpy as np

from sosfuse2.foundation import content_identity as source_identity, dependencies, paths
from sosfuse2.foundation.top import TOP_KM, SUN_HEIGHT_INTERVALS, Domain

R = 6371.0
Z = TOP_KM


_LEGENDRE = np.polynomial.legendre
_LEGGAUSS = _LEGENDRE.leggauss if getattr(_LEGENDRE.leggauss, '__globals__', None) is vars(_LEGENDRE) else None


@lru_cache(maxsize=32)
def _gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    u, w = (x + 1) / 2, w / 2
    u.flags.writeable = w.flags.writeable = False
    return u, w


def gl(n):
    if np.polynomial.legendre.leggauss is not _LEGGAUSS or not _standard_numpy(np):
        x, w = np.polynomial.legendre.leggauss(n)
        return (x + 1) / 2, w / 2
    u, w = _gl(n)
    return u.copy(), w.copy()


def frame(r, chi):
    up = np.stack([np.sin(chi), np.zeros_like(chi), np.cos(chi)], -1)
    ex = np.stack([-np.cos(chi), np.zeros_like(chi), np.sin(chi)], -1)
    return r[..., None] * up, up, ex


def boundaries(p, d, *, domain=None):
    """Exact shell exit, opaque-ground hit, tangent, and cylindrical shadow hits.
    Sun is +z; only the back half of the Earth cylinder is shadow.
    """
    b = np.sum(p * d, -1)
    pp = np.sum(p * p, -1)
    domain = domain or Domain()
    end = -b + np.sqrt(np.maximum(b * b - pp + (R + domain.top_km) ** 2, 0))
    disc = b * b - pp + R * R
    hit = -b - np.sqrt(np.maximum(disc, 0))
    ground = (disc >= 0) & (b < 0) & (hit >= -1e-7)
    end = np.where(ground, np.maximum(hit, 0), end)
    end = np.maximum(end, 0)
    tangent = np.where((-b > 0) & (-b < end), -b, end)
    aa = d[:, 0] ** 2 + d[:, 1] ** 2
    bb = p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1]
    cc = p[:, 0] ** 2 + p[:, 1] ** 2 - R * R
    ds = bb * bb - aa * cc
    roots = []
    for sign in (-1, 1):
        root = (-bb + sign * np.sqrt(np.maximum(ds, 0))) / np.maximum(aa, 1e-30)
        valid = (aa > 1e-20) & (ds >= 0) & (root > 0) & (root < end) & (p[:, 2] + root * d[:, 2] < 0)
        roots.append(np.where(valid, root, end))
    return np.sort(np.stack([np.zeros_like(end), tangent, *roots, end], -1), axis=-1)


def profiles(aerosol, h, *, domain=None):
    from sosfuse2.atmosphere.density import no2_profile

    domain = domain or Domain()
    values = np.stack(
        [
            np.exp(-np.maximum(h, 0) / 8),
            aerosol(h),
            np.maximum(0, 1 - np.abs(h - 25) / 15),
            no2_profile(np.minimum(h, domain.lower_end)),
            np.exp(-np.maximum(h, 0) / 4),
        ],
        -1,
    )
    if domain.top_km > TOP_KM:
        values = np.where(np.asarray(h)[..., None] > np.array([domain.air_end, *([domain.lower_end] * 4)]), 0, values)
    return values


def grids(scale):
    h = np.array([0, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2, 3, 4.5, 6.5, 9, 13, 18, 26, 40, 60, Z])
    chi = np.unique(
        np.r_[
            np.arange(0, 81, 10),
            np.arange(80, 85, 1),
            np.arange(85, 110, 0.5),
            np.arange(110, 120, 2),
            np.arange(120, 181, 10),
        ]
    )
    if scale == 2:
        h = np.sort(np.r_[h, (h[:-1] + h[1:]) / 2])
        chi = np.sort(np.r_[chi, (chi[:-1] + chi[1:]) / 2])
    return h, np.deg2rad(chi)


def small_heights():
    return np.array([0.0, 0.1, 8.0, Z])


def sun_grid(scale, top=None):
    """Solar-table coordinates; height sample count is independent of the boundary."""
    top = Z if top is None else top
    return np.linspace(0, np.sqrt(top), SUN_HEIGHT_INTERVALS * scale + 1) ** 2, np.linspace(0, 1, 400 * scale + 1)


def geometry_segments(aerosol, p, d, n, *, domain=None):
    domain = domain or Domain()
    edges = boundaries(p, d, domain=domain)
    end = edges[:, -1]
    b = np.sum(p * d, -1)
    pp = np.sum(p * p, -1)
    more = []
    for height in domain.extra_shells:
        disc = b * b - pp + (R + height) ** 2
        for sign in (-1, 1):
            root = -b + sign * np.sqrt(np.maximum(disc, 0))
            more.append(np.where((disc > 0) & (root > 0) & (root < end), root, end))
    edges = np.sort(np.column_stack([edges, *more]), axis=-1)
    u, wu = gl(n)
    tau = np.zeros((len(d), 5))
    for k in range(edges.shape[1] - 1):
        active = np.flatnonzero(edges[:, k + 1] - edges[:, k] > 1e-8)
        if not len(active):
            continue
        a, b = edges[active, k, None], edges[active, k + 1, None]
        pa, da = p[active], d[active]
        ra = np.sum((pa + a * da) ** 2, -1)
        rb = np.sum((pa + b * da) ** 2, -1)
        lower = (ra <= rb)[:, None]
        t = np.where(lower, a + (b - a) * u * u, b - (b - a) * (1 - u) ** 2)
        ds = np.where(lower, 2 * (b - a) * u * wu, 2 * (b - a) * (1 - u) * wu)
        q = pa[:, None, :] + t[..., None] * da[:, None, :]
        r = np.sqrt(np.sum(q * q, -1))
        h = r - R
        dt = profiles(aerosol, h, domain=domain) * ds[..., None]
        prefix = tau[active, None, :] + np.cumsum(dt, axis=1) - dt / 2
        tau[active] += dt.sum(axis=1)
        yield active, q, r, ds, prefix


# Native loops replace these numerical operations only for the ordinary NumPy backend.
_NUMPY = np


def _numpy_export(name):
    """An override already installed during import is not a numerical primitive."""
    fn = getattr(np, name)
    namespace = getattr(fn, '__globals__', {})
    if not isinstance(namespace, dict):
        return None
    owner = namespace.get('__name__', getattr(fn, '__module__', type(fn).__module__))
    if (isinstance(owner, str) and owner.partition('.')[0] == 'numpy'
            and getattr(fn, '__name__', None) == ('absolute' if name == 'abs' else name)):
        return fn
    return None


_NUMPY_BINDINGS = tuple((name, _numpy_export(name)) for name in (
    'abs', 'arange', 'arcsinh', 'array', 'asarray', 'broadcast_to', 'clip', 'column_stack', 'cumsum',
    'all', 'ascontiguousarray', 'empty', 'require', 'float64', 'diff', 'exp', 'finfo', 'flatnonzero', 'full', 'hypot', 'interp', 'isfinite', 'linspace',
    'max', 'maximum', 'minimum',
    'any', 'ndim', 'shape', 'sort', 'sqrt', 'square', 'stack', 'sum', 'where', 'zeros', 'zeros_like',
))
_BOUNDARIES, _QUADRATURE, _PROFILES, _GEOMETRY = boundaries, gl, profiles, geometry_segments
_DOMAIN = Domain
_DOMAIN_PROPERTIES = (Domain.extra_shells, Domain.air_end, Domain.lower_end, Domain.identity)


def _standard_numpy(module):
    return (module is _NUMPY and np.inf == float('inf')
            and all(fn is not None and getattr(module, name) is fn for name, fn in _NUMPY_BINDINGS))


def _standard_domain(domain):
    return (type(domain) is _DOMAIN and Domain is _DOMAIN
            and all(a is b for a, b in zip((Domain.extra_shells, Domain.air_end, Domain.lower_end, Domain.identity),
                                        _DOMAIN_PROPERTIES)))


def _standard_probe(old):
    if (Domain is not _DOMAIN or type(old) is not SimpleNamespace or type(getattr(old, 'probe', None)) is not SimpleNamespace
            or getattr(old.probe, 'gl', None) is not _QUADRATURE or gl is not _QUADRATURE
            or np.polynomial.legendre.leggauss is not _LEGGAUSS or boundaries is not _BOUNDARIES):
        return False
    bound = getattr(old.probe, 'boundaries', None)
    return (bound is _BOUNDARIES or
            (type(bound) is partial and bound.func is _BOUNDARIES and not bound.args
             and bound.keywords.keys() <= {'domain'} and _standard_domain(bound.keywords.get('domain') or Domain())))


def _standard_binding(bound, function):
    from ..atmosphere import density, tropo_profile

    return (Domain is _DOMAIN and type(bound) is partial and bound.func is function and len(bound.args) == 1
            and bound.keywords.keys() <= {'domain'} and _standard_domain(bound.keywords.get('domain') or Domain())
            and profiles is _PROFILES and density._standard_no2() and tropo_profile._standard_aerosol(bound.args[0])
            and np is density.np is tropo_profile.np and _standard_numpy(np))


def _plain_geometry(bound):
    from ..atmosphere import density

    if (geometry_segments is not _GEOMETRY or boundaries is not _BOUNDARIES or gl is not _QUADRATURE
            or np.polynomial.legendre.leggauss is not _LEGGAUSS or not _standard_binding(bound, _GEOMETRY)):
        return False
    # The full geometry validates NO2 when it computes its first optical prefix.
    # A cold asset, or support outside its validated grid, must still run that code.
    if not density.no2_asset.cache_info().currsize:
        return False
    domain = bound.keywords.get('domain') or Domain()
    return domain.lower_end <= density.no2_asset()[0][-1]


def total_columns(aerosol, p, d, n, *, domain=None):
    out = np.zeros((len(d), 5))
    for active, q, r, ds, prefix in geometry_segments(aerosol, p, d, n, domain=domain):
        out[active] += np.sum(profiles(aerosol, r - R, domain=domain) * ds[..., None], axis=1)
    return out


class SunGeometry:
    @staticmethod
    def identity(scale, aerosol_height, descriptor, domain=None):
        """Everything the table depends on. A file without this identity, or with another one, is rebuilt: an unkeyed
        3-column table is incompatible with the 5-column solver. descriptor is the
        aerosol profile record (kind and parameters), so two profiles with one scale height never share a table even in
        one folder. sun_parallel.prebuild writes and checks the same string."""
        from sosfuse2.atmosphere import components, density

        files = (paths.PKG / name for name in dependencies.files('sun_table'))
        domain = domain or Domain()
        height = aerosol_height
        return json.dumps(
            dict(
                table='sun_columns',
                version=3,
                components=list(components.COMPONENTS),
                R=float(R),
                Z=float(domain.top_km),
                domain=domain.identity(),
                extra_shells_km=[float(h) for h in domain.extra_shells],
                scale=int(scale),
                aerosol_height=None if height is None else float(height),
                aerosol=descriptor,
                no2_asset_sha256=density.no2_asset()[2],
                native=density.host_native_source_hash(),
                sources={f.name: source_identity.file_sha(f) for f in files},
            ),
            sort_keys=True,
        )

    @staticmethod
    def stored_identity(path):
        """The identity string a saved table carries; None for a table written before identities (or without one)."""
        with np.load(path) as z:
            return str(z['identity']) if 'identity' in z.files else None

    def __init__(self, scale, folder, *, aerosol, aerosol_height, descriptor, domain=None):
        self.domain = domain or Domain()
        path = folder / 'sun_columns.npz'
        t = time.perf_counter()
        ident = self.identity(scale, aerosol_height, descriptor, self.domain)
        if path.exists():
            with np.load(path) as z:  # identity and arrays from one open: a concurrent replace cannot pair them wrongly
                stored = str(z['identity']) if 'identity' in z.files else None
                if stored == ident:
                    self.h, self.x, self.col = z['h'], z['x'], z['col']
                    self.build_seconds = 0.0
                    return
            import warnings

            warnings.warn(
                f"SUN_TABLE_STALE: {path}: {'no identity' if stored is None else 'identity differs'}; rebuilding",
                RuntimeWarning,
            )
        self.h, self.x = sun_grid(scale, self.domain.top_km)
        col = np.empty((len(self.h), len(self.x), 5))
        for i, h in enumerate(self.h):
            r = R + h
            mg = -np.sqrt(max(0, 1 - (R / r) ** 2))
            mu = mg + (1 - mg) * self.x**2
            mu[0] = mg + 1e-9
            p = np.column_stack([np.zeros(len(mu)), np.zeros(len(mu)), np.full(len(mu), r)])
            d = np.column_stack([np.sqrt(np.maximum(1 - mu * mu, 0)), np.zeros(len(mu)), mu])
            col[i] = total_columns(aerosol, p, d, 32 * scale, domain=self.domain)
        self.col = col
        import os  # tmp + replace: a reader never sees a half-written table (also when a stale one is replaced)
        import tempfile

        fd, name = tempfile.mkstemp(prefix='sun_columns.', suffix='.tmp', dir=folder)
        try:
            with os.fdopen(fd, 'wb') as f:
                np.savez(f, h=self.h, x=self.x, col=col, identity=np.array(ident))
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self.build_seconds = time.perf_counter() - t

    def lookup(self, q, r):
        h = np.clip(r - R, 0, self.domain.top_km)
        mu = q[..., 2] / r
        mg = -np.sqrt(np.maximum(0, 1 - (R / np.maximum(r, R)) ** 2))
        x = np.sqrt(np.clip((mu - mg) / (1 - mg), 0, 1))
        i = np.clip(np.searchsorted(self.h, h, side='right') - 1, 0, len(self.h) - 2)
        a = (h - self.h[i]) / (self.h[i + 1] - self.h[i])
        f = x * (len(self.x) - 1)
        j = np.clip(f.astype(int), 0, len(self.x) - 2)
        b = f - j
        col = (
            (1 - a)[..., None] * (1 - b)[..., None] * self.col[i, j]
            + a[..., None] * (1 - b)[..., None] * self.col[i + 1, j]
            + (1 - a)[..., None] * b[..., None] * self.col[i, j + 1]
            + a[..., None] * b[..., None] * self.col[i + 1, j + 1]
        )
        return col, mu >= mg
