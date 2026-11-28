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
    try:
        n = int(value)
    except ValueError:
        pass
    else:
        if n > 0:
            return n
    raise ValueError(f'E_MLX_TILE_SAMPLES: {TILE_ENV} must be a positive integer, got {value!r}')


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
    if result is not None and os.environ.get(EVAL_SHADOW_ENV, '').strip() == '1':
        _eval_shadow(geometry, s, field, first, phase, ground_only, boundary, result)
    return result


EVAL_SHADOW_ENV = 'SOSFUSE2_MLX_EVAL_SHADOW'
eval_shadow = []


def _eval_shadow(geometry, s, field, first, phase, ground_only, boundary, result):
    """Check one MLX evaluate call against the NumPy branch of TransientGeometry.evaluate (bench harness only).

    The record keeps the mode, the shapes and the largest |MLX - NumPy| over cells and components divided by that
    wavelength's largest |NumPy| value, for the moments and for the flux separately (a zero NumPy flux must stay zero).
    """
    import numpy as np

    from . import numpy_reference

    count = len(s.geo.chi) if ground_only else s.geo.ns
    rec = dict(mode=mode_name(first, phase, ground_only), phase=int(phase), cells=count, max_rel_moments=0.0,
               max_rel_flux=0.0, flux_nonzero=False, problems=[])
    moments = np.zeros((count, 4 + s.kernel.K, s.W), 'f8')
    flux = np.empty((count, s.W), 'f8')
    for i in range(count):
        g = geometry.cell(i)
        c = s.kernel.setup(s, g['nq'])
        value, Y = numpy_reference.transport(g, c, field, first, boundary, None, return_basis=True)
        moments[i], flux[i] = numpy_reference.project(g, value, c, Y, ground_only=phase == 1)
    got_m, got_f = (np.asarray(x, 'f8') for x in result)
    if got_m.shape != moments.shape or got_f.shape != flux.shape:
        rec['problems'].append(f'shapes {got_m.shape} {got_f.shape} != {moments.shape} {flux.shape}')
    elif not (np.all(np.isfinite(got_m)) and np.all(np.isfinite(got_f))):
        rec['problems'].append('non-finite values')
    else:
        def rel(got, ref, axes):
            scale = np.abs(ref).max(axis=axes)
            err = np.abs(got - ref).max(axis=axes)
            return float(np.where(scale > 0, err / np.where(scale > 0, scale, 1.0), np.where(err > 0, np.inf, 0.0)).max())

        rec['max_rel_moments'] = rel(got_m, moments, (0, 1))
        rec['max_rel_flux'] = rel(got_f, flux, (0,))
        rec['flux_nonzero'] = bool(np.any(flux))
        rec['got_flux_nonzero'] = bool(np.any(got_f))
    eval_shadow.append(rec)


def query(packs, constants, field, first=False):
    """Phase-2 values (observer monitors and table rays) from the MLX implementation, or None for the NumPy reference.

    Same contract as one `numpy_reference.transport(pack, constant, field, first)` call per pack: `packs` are the
    packed ray chunks, `constants[i]` is `kernel.setup(s, packs[i]['nq'])`, `field` is the float32 source field
    (cells, source size, W) - the table path passes the converged field to first-order calls too, so the work depends
    on `first`, not on the field values - and the result is a list with one (rays_i, W, 4) array per pack, in pack
    order, within rtol 1e-3 of the NumPy reference. `coverage` counts the calls as query:mlx or query:numpy.
    """
    require_mlx()
    from . import mlx_ops

    result = mlx_ops.query(packs, constants, field, first)
    key = 'query:mlx' if result is not None else 'query:numpy'
    coverage[key] = coverage.get(key, 0) + 1
    if result is not None and os.environ.get(SHADOW_ENV, '').strip() == '1':
        _shadow(packs, constants, field, first, result)
    return result


SHADOW_ENV = 'SOSFUSE2_MLX_QUERY_SHADOW'
shadow = []


def _shadow(packs, constants, field, first, result):
    """Check one MLX query call against the NumPy reference (bench harness only, SOSFUSE2_MLX_QUERY_SHADOW=1).

    Every pack is recomputed with numpy_reference.transport; the record keeps the pack count, the shapes and, per pack, the
    largest |MLX - NumPy| over rays and the four components divided by that wavelength's largest |NumPy| value.
    """
    import numpy as np

    from . import numpy_reference

    rec = dict(first=bool(first), packs=len(packs), results=len(result), rays=0, max_rel=0.0, problems=[])
    if len(result) != len(packs):
        rec['problems'].append(f'{len(result)} results for {len(packs)} packs')
    for i, (pack, constant, value) in enumerate(zip(packs, constants, result)):
        ref = np.asarray(numpy_reference.transport(pack, constant, field, first), 'f8')
        got = np.asarray(value, 'f8')
        rec['rays'] += len(ref)
        if got.shape != ref.shape:
            rec['problems'].append(f'pack {i}: shape {got.shape} != {ref.shape}')
            continue
        if not np.all(np.isfinite(got)):
            rec['problems'].append(f'pack {i}: non-finite values')
            continue
        if not ref.size:
            continue
        scale = np.abs(ref).max(axis=(0, 2))
        err = np.abs(got - ref).max(axis=(0, 2))
        rel = np.where(scale > 0, err / np.where(scale > 0, scale, 1.0), np.where(err > 0, np.inf, 0.0))
        rec['max_rel'] = max(rec['max_rel'], float(rel.max()))
    shadow.append(rec)
