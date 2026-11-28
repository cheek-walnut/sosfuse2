"""One quadrature walk for independent columns, retaining each reduction shape.

No optical prefixes are needed for endpoint columns. Segment order, GL nodes,
XYZ summation order and per-component profiles follow geometry_segments.
"""

import numpy as np
from ..atmosphere.components import NCOMP, MOLECULAR_COLUMNS
from ..foundation.top import Domain


def _host_loops():
    from ..atmosphere.density import _host_native

    return _host_native()


def _profile_columns(p, d, edges, old, n, optics):
    """Evaluate a standalone legacy or molecular profile without materializing the ray-by-node arrays."""
    from . import shell_segments
    from ..atmosphere import density, tropo_profile

    if (Domain is not shell_segments._DOMAIN or not shell_segments._standard_numpy(np)
            or not shell_segments._standard_probe(old)
            or not shell_segments._standard_domain(getattr(old, 'domain', Domain()))
            or type(d) is not np.ndarray or type(p) is not np.ndarray or d.ndim != 2 or d.shape[1] != 3
            or NCOMP != 5 or not len(d) or p.dtype != np.float64 or d.dtype != np.float64):
        return None
    if optics is None:
        binding = getattr(old, 'profiles', None)
        if not shell_segments._standard_binding(binding, shell_segments._PROFILES):
            return None
    elif (type(optics) is not dict or not density._standard_molecular() or not shell_segments._standard_numpy(density.np)
          or not shell_segments._standard_domain(density.domain_of(optics))):
        return None
    loops = _host_loops()
    if loops is None:
        return None
    edges = _edges(p, d, old, edges)
    if (type(edges) is not np.ndarray or edges.ndim != 2 or len(edges) != len(d) or edges.dtype != np.float64
            or not np.any(np.diff(edges, axis=1) > 1e-8)):
        return None
    parameter = np.empty(0)
    if optics is None:
        profile = binding.args[0]
        if type(profile) is tropo_profile.ExponentialProfile:
            parameter = np.array([profile.height], 'f8')
            kind = 0
        else:
            parameter = np.require(profile.vector, dtype='f8', requirements=['C', 'A'])
            kind = 1
        domain = binding.keywords.get('domain') or Domain()
        z, no2, _ = density.no2_asset()
        if (len(z) < 2 or z.shape != no2.shape or z[0] != 0 or z[1] <= 0
                or not np.all(np.diff(z) == z[1]) or not 0 <= domain.lower_end <= z[-1]):
            return None
        step = z[1] - z[0]
        values = no2[:, None]
        width, mask = NCOMP, 0
        extended = domain.top_km > shell_segments.TOP_KM
    else:
        rows = density.check_consumed(optics, where='profiles')
        kind = 2
        domain = density.domain_of(optics)
        step = domain.step_km
        density.grid(rows, step=step, top_km=domain.top_km)
        if type(optics['table']) is not np.ndarray or optics['table'].dtype != np.float64:
            return None
        values = optics['table'][:, :2]
        width, mask = 4, optics['mask']
        extended = domain.top_km > density.top.NORMALIZATION_TOP_KM
    nodes, weights = old.probe.gl(n)
    if (type(nodes) is not np.ndarray or type(weights) is not np.ndarray or nodes.ndim != 1
            or nodes.shape != weights.shape or nodes.dtype != np.float64 or weights.dtype != np.float64):
        return None
    table = np.ascontiguousarray(np.column_stack((values, np.diff(values, axis=0, append=values[-1:]))), 'f8')
    arrays = [np.require(v, dtype='f8', requirements=['C', 'A']) for v in (p, d, edges, nodes, weights)]
    out = np.zeros((len(d), width))
    for array, shape in zip(arrays, (d.shape, d.shape, edges.shape, nodes.shape, nodes.shape)):
        density._host_buffer(array, shape)
    density._host_buffer(parameter, ((1, 7, 0)[kind],))
    density._host_buffer(table, (len(values), 4 if kind == 2 else 2))
    density._host_buffer(out, (len(d), width), writable=True)
    rc = loops.profile_columns(len(d), edges.shape[1], len(nodes), *arrays, 6371.0, kind, mask, parameter,
                               len(table), step, table, domain.air_end if extended else np.inf,
                               domain.lower_end if extended else np.inf, domain.lower_end if optics is None else np.inf, out)
    if rc:
        return None
    return out


def _edges(p, d, old, edges=None):
    if edges is not None:
        return edges
    edges = old.probe.boundaries(p, d)
    end = edges[:, -1]
    dot = np.sum(p * d, -1)
    pp = np.sum(p * p, -1)
    return _add_shells(edges, end, dot, pp, getattr(old, 'domain', Domain()), 6371.0)


def _add_shells(edges, end, dot, pp, domain, radius):
    more = []
    for height in domain.extra_shells:
        disc = dot * dot - pp + (radius + height) ** 2
        for sign in (-1, 1):
            root = -dot + sign * np.sqrt(np.maximum(disc, 0))
            more.append(np.where((disc > 0) & (root > 0) & (root < end), root, end))
    return np.sort(np.column_stack([edges, *more]), axis=-1)


def _plain_layer(p, d, profile):
    return (type(d) is np.ndarray and type(p) is np.ndarray and d.ndim == 2 and d.shape[1] == 3
            and d.dtype == np.float64 and p.dtype == np.float64 and type(profile) is dict
            and all(type(profile.get(k)) in (float, int, np.float64) for k in ('center', 'sigma', 'norm'))
            and ('end_km' not in profile or type(profile['end_km']) in (float, int, np.float64))
            and profile['sigma'] > 0 and profile['norm'] > 0)


def _strato_columns(p, d, binding, profile, n, out, backend):
    from . import shell_segments as shell

    if (backend is not np or not _plain_layer(p, d, profile)
            or not shell._standard_numpy(np) or not shell._plain_geometry(binding)):
        return False
    loops = _host_loops()
    if loops is None:
        return False
    domain = binding.keywords.get('domain') or shell.Domain()
    edges = shell.boundaries(p, d, domain=domain)
    edges = _add_shells(edges, edges[:, -1], np.sum(p * d, -1), np.sum(p * p, -1), domain, shell.R)
    return _integrate_layer(loops, p, d, edges, *shell.gl(n), profile, out)


def _integrate_layer(loops, p, d, edges, u, w, profile, out):
    from ..atmosphere.density import _host_buffer

    if type(u) is not np.ndarray or type(w) is not np.ndarray or u.ndim != 1 or u.shape != w.shape:
        return False
    # The optical prefix uses shell.R. The consuming stratosphere profile measures h from 6371 km.
    params = np.array([profile['center'], profile['sigma'], profile['norm']], 'f8')
    arrays = [np.require(v, dtype='f8', requirements=['C', 'A']) for v in (p, d, edges, u, w)]
    for array, shape in zip(arrays, (d.shape, d.shape, (len(d), edges.shape[1]), u.shape, u.shape)):
        _host_buffer(array, shape)
    _host_buffer(params, (3,))
    _host_buffer(out, (len(d),), writable=True)
    rc = loops.profile_columns(len(d), edges.shape[1], len(u), *arrays, 6371.0, 3, 0, params, 0, 1.0,
                               np.empty(0), np.inf, profile.get('end_km', np.inf), np.inf, out)
    if rc:
        out.fill(0)
    return rc == 0


def segments(p, d, old, n, edges=None):
    edges = _edges(p, d, old, edges)
    u, wu = old.probe.gl(n)
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
        # Equivalent to np.sum(q*q, -1) for a trailing axis of length three.
        radius2 = np.square(pa[:, None, 0] + t * da[:, None, 0])
        for axis in (1, 2):
            radius2 += np.square(pa[:, None, axis] + t * da[:, None, axis])
        yield active, np.sqrt(radius2), ds


_DEFAULT_SEGMENTS = segments


def integrate(p, d, old, n=24, *, legacy=False, optics=None, profile=None, edges=None, main=False):
    if main:
        from . import column_table

        if column_table.method() != 'gl24':
            return column_table.integrate(p, d, old, legacy=legacy, optics=optics, profile=profile, edges=edges)
    from ..atmosphere.density import molecular_profiles
    from ..atmosphere import stratosphere
    from ..atmosphere.stratosphere import density

    if segments is _DEFAULT_SEGMENTS and type(d) is np.ndarray and (d.ndim != 2 or d.shape[1] != 3):
        raise ValueError('E_COLUMN_RAYS: expected Cartesian direction vectors')
    p = np.broadcast_to(p, np.shape(d))
    base = np.zeros((len(d), NCOMP)) if legacy else None
    corrected = np.zeros((len(d), 4)) if optics is not None else None
    strato = np.zeros(len(d)) if profile is not None else None
    walk = segments
    if walk is _DEFAULT_SEGMENTS and type(edges) is np.ndarray and (
            edges.ndim != 2 or edges.shape[0] != len(d) or not edges.shape[1]):
        raise ValueError('E_COLUMN_RAYS: expected one nonempty edge vector per ray')
    nodes = walk(p, d, old, n, edges)
    if walk is _DEFAULT_SEGMENTS and profile is None and type(legacy) is bool and legacy != (optics is not None):
        values = _profile_columns(p, d, edges, old, n, optics)
        if values is not None:
            if legacy:
                return values, None, None
            corrected = values
            nodes = ()
    if walk is _DEFAULT_SEGMENTS and legacy is False and optics is None and strato is not None:
        from . import shell_segments as shell

        if (density is stratosphere._COLUMN_DENSITY and stratosphere.np is np and Domain is shell._DOMAIN
                and _plain_layer(p, d, profile) and shell._standard_numpy(np) and shell._standard_probe(old)
                and shell._standard_domain(getattr(old, 'domain', Domain()))
                and (edges is None or type(edges) is np.ndarray and edges.dtype == np.float64)):
            loops = _host_loops()
            if loops is not None:
                cut = _edges(p, d, old, edges)
                if _integrate_layer(loops, p, d, cut, *old.probe.gl(n), profile, strato):
                    return None, None, strato
    for active, r, ds in nodes:
        h = r - 6371.0
        if base is not None:
            base[active] += np.sum(old.profiles(h) * ds[..., None], axis=1)
        if corrected is not None:
            corrected[active] += np.sum(molecular_profiles(h, optics) * ds[..., None], axis=1)
        if strato is not None:
            strato[active] += np.sum(density(h, profile) * ds, axis=1)
    if corrected is not None:
        from ..atmosphere.density import exact_profile_columns

        end = old.probe.boundaries(p, d)[:, -1] if edges is None else edges[:, -1]
        corrected[:, 2:] = exact_profile_columns(p, d, end, optics, selected=(2, 3))[:, 2:]
    return base, corrected, strato


def replace(base, corrected, optics):
    out = np.array(base, copy=True)
    if corrected is not None:
        for k, col in enumerate(MOLECULAR_COLUMNS):
            if k >= 2 or optics['mask'] & (1 << k):
                out[:, col] = corrected[:, k]
    return out


_DEFAULT_INTEGRATE, _DEFAULT_REPLACE = integrate, replace
