"""MLX transport for lower-atmosphere and extended-top HG sources with optional stratosphere and arbitrary Lambertian surface albedo."""

import numpy as np


def _domain_key(domain):
    return (domain.top_km, domain.step_km, domain.lower_end, domain.air_end)


def _check_extended_profile(c):
    from ..atmosphere.density import check_consumed

    optics = c.get('density_optics')
    if optics is None or 'table' not in optics:
        raise RuntimeError('E_PROFILE_N_MISMATCH: an extended domain requires density_optics')
    domain = c['domain']
    check_consumed(optics, compiled_n=domain.n, where='mlx', domain=domain)


def _supported(c, first=False):
    if 'tropo_lut' in c:
        return False
    domain = c.get('domain')
    if domain is not None and domain.top_km > domain.lower_end:
        _check_extended_profile(c)
    if 'strato_ext' in c:
        if 'strato_profile' not in c or 'strato_density' not in c:
            return False
        if first and not all(k in c for k in ('strato_sca', 'strato_chi', 'strato_sun_col')):
            return False
        return True
    return not c.get('sh_combine', False)


def evaluate(geometry, s, field, first, phase, ground_only, boundary):
    count = len(s.geo.chi) if ground_only else s.geo.ns
    cells = [geometry.cell(i) for i in range(count)]
    by_order = {n: s.kernel.setup(s, n) for n in {g['nq'] for g in cells}}
    if not all(_supported(c, first=first) for c in by_order.values()):
        return None
    constants = [by_order[g['nq']] for g in cells]
    if not count:
        return np.zeros((0, 4 + s.kernel.K, s.W), 'f8'), np.zeros((0, s.W), 'f8')
    if phase == 1:
        if all(c['albedo'] == 0 for c in by_order.values()):
            # With a black surface the Lambert flux is identically zero.
            return np.zeros((count, 4 + s.kernel.K, s.W), 'f8'), np.zeros((count, s.W), 'f8')
        from .mlx_warm import CompactRays, evaluate as evaluate_tiles

        cache = getattr(geometry, '_mlx_rays_cache', None)
        if cache is None:
            cache = geometry._mlx_rays_cache = {}
        include_empty = boundary is not None and bool(np.any(boundary))
        key = (bool(ground_only), include_empty, True)
        packed = cache.get(key)
        if packed is None or not packed.matches(cells):
            packed = cache[key] = CompactRays(cells, include_empty=include_empty, upward_only=True)
        return evaluate_tiles(cells, constants, field, boundary, packed, first=first, ground_only=True)
    if phase == 0:
        from .mlx_warm import CompactRays, evaluate as evaluate_tiles

        cache = getattr(geometry, '_mlx_rays_cache', None)
        if cache is None:
            cache = geometry._mlx_rays_cache = {}
        include_empty = boundary is not None and bool(np.any(boundary))
        key = (bool(ground_only), include_empty, False)
        packed = cache.get(key)
        if packed is None or not packed.matches(cells):
            packed = cache[key] = CompactRays(cells, include_empty=include_empty)
        return evaluate_tiles(cells, constants, field, boundary, packed, first=first, ground_only=False)
    return None


def query(packs, constants, field, first):
    """Integrate the four observer tensor components for each input ray."""
    if not all(_supported(c, first=first) for c in constants):
        return None
    from .mlx_query import query as query_rays

    return query_rays(packs, constants, field, first)
