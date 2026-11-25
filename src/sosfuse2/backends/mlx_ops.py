"""MLX transport for a black surface and a lower-atmosphere HG source."""

import numpy as np

from . import numpy_reference


def _supported(c):
    return (
        'strato_ext' not in c
        and not numpy_reference.support_extended(c['domain'])
        and c['albedo'] == 0
        and not np.any(c['gm'])
        and not np.any(c['gv'])
        and 'tropo_lut' not in c
    )


def evaluate(geometry, s, field, first, phase, ground_only, boundary):
    count = len(s.geo.chi) if ground_only else s.geo.ns
    cells = [geometry.cell(i) for i in range(count)]
    by_order = {n: s.kernel.setup(s, n) for n in {g['nq'] for g in cells}}
    if not all(_supported(c) for c in by_order.values()):
        return None
    constants = [by_order[g['nq']] for g in cells]
    if phase == 1 or not count:
        # With a black surface the Lambert flux is identically zero.
        return np.zeros((count, 4 + s.kernel.K, s.W), 'f8'), np.zeros((count, s.W), 'f8')
    if phase == 0:
        from .mlx_warm import CompactRays, evaluate as evaluate_tiles

        packed = getattr(geometry, '_mlx_rays', None)
        include_empty = boundary is not None and bool(np.any(boundary))
        if packed is None or packed.include_empty != include_empty or not packed.matches(cells):
            packed = geometry._mlx_rays = CompactRays(cells, include_empty)
        return evaluate_tiles(cells, constants, field, boundary, packed, first=first)
    return None
