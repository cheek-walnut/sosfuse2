"""Atmospheric calculation boundary and independent sampling policies, in km."""

import math

TOP_KM = 80.0
NORMALIZATION_TOP_KM = 80.0
SUN_HEIGHT_INTERVALS = 80
COLUMN_TOP_MARGIN_KM = 1.0


def profile_n(step, top=None):
    """Number of nodes on a uniform profile grid, including both endpoints."""
    top = TOP_KM if top is None else top
    try:
        if isinstance(step, bool) or isinstance(top, bool):
            raise ValueError
        step, top = float(step), float(top)
        if not math.isfinite(step) or not math.isfinite(top) or step <= 0 or top <= 0:
            raise ValueError
        intervals = top / step
        nearest = round(intervals)
        if nearest < 1 or not math.isclose(intervals, nearest, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError('E_PROFILE_GRID: top must be a positive integral multiple of step') from exc
    return nearest + 1


def profile_kernel_constants(step, top=None):
    """Return the inverse spacing and last interval for the same grid."""
    n = profile_n(step, top)
    return 1.0 / float(step), n - 2


def _float_literal(value):
    text = format(float(value), '.9g')
    if '.' not in text and 'e' not in text.lower():
        text += '.0'
    return text + 'f'


def profile_kernel_defines(step, top=None):
    """NVRTC definitions for the boundary and its uniform profile grid."""
    top = TOP_KM if top is None else top
    n = profile_n(step, top)
    inverse, last = profile_kernel_constants(step, top)
    return (
        f'-DPROFILE_N={n}',
        f'-DPROFILE_INV={_float_literal(inverse)}',
        f'-DPROFILE_LAST={last}',
        f'-DTOP_KM={_float_literal(top)}',
    )
