"""Optional MLX (Apple Metal) execution of the transport operator.

SOSFUSE2_MLX=1 routes TransientGeometry.evaluate (the host-side NumPy path of the solver) here. A configuration the MLX
code does not cover yet returns None and the NumPy reference runs instead, so every call path stays complete while
modules move over one at a time. `coverage` counts which path served each call; the bench harness reports it.

The work behind `evaluate` lives in `mlx_ops`. Its contract: same inputs as TransientGeometry.evaluate, same outputs
as the NumPy branch there - moments (cells, 4 + K, W) float64 and flux (cells, W) float64 - within the numerical rule
of the gates (rtol 1e-3 against the NumPy reference).

Memory: device work is split into tiles of at most `tile_samples()` path samples (nonzero intervals times path order,
the `geometry.stats['samples']` definition). What a tile allocates is released before the next tile, so the resident
set of one call grows with the tile bound, not with the number of samples in the call. A single cell with more samples
than the bound is one tile by itself.
"""

import os

ENV = 'SOSFUSE2_MLX'
TILE_ENV = 'SOSFUSE2_MLX_TILE_SAMPLES'
DEFAULT_TILE_SAMPLES = 1 << 20
coverage = {}


def enabled():
    return os.environ.get(ENV, '').strip() == '1'


def tile_samples():
    """Upper bound on the path samples one device tile may hold (SOSFUSE2_MLX_TILE_SAMPLES, default 2**20)."""
    value = os.environ.get(TILE_ENV, '').strip()
    if not value:
        return DEFAULT_TILE_SAMPLES
    n = int(value)
    if n < 1:
        raise ValueError(f'{TILE_ENV} must be a positive integer, got {value!r}')
    return n


def mode_name(first, phase, ground_only):
    return 'first' if first else 'ground' if (phase == 1 or ground_only) else 'warm'


def require_mlx():
    """SOSFUSE2_MLX=1 asks for the MLX transport explicitly, so a missing package is an error, not a silent fallback."""
    try:
        import mlx.core  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f'E_MLX_UNAVAILABLE: {ENV}=1 needs the mlx package (Apple silicon); install it or unset {ENV}'
        ) from exc


def evaluate(geometry, s, field, first=False, phase=0, ground_only=False, boundary=None):
    """(moments, flux) from the MLX implementation, or None to fall back to the NumPy reference."""
    require_mlx()
    from . import mlx_ops

    result = mlx_ops.evaluate(geometry, s, field, first, phase, ground_only, boundary)
    key = mode_name(first, phase, ground_only) + (':mlx' if result is not None else ':numpy')
    coverage[key] = coverage.get(key, 0) + 1
    return result
