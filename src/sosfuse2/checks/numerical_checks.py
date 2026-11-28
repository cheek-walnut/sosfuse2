"""Numerical equivalence for research harnesses (not file-integrity checks)."""

import numpy as np


def assert_equivalent(actual, expected, rtol=1e-6, floor=1e-12):
    """Equal within rtol, with an absolute floor of `floor` times the largest |expected|.

    The floor keeps rounding noise in entries that cancel to nearly zero (for example 1e-20 next to 2e-4, which the
    compiled and NumPy host paths round differently) from failing a relative comparison; any difference above
    floor x scale still fails.
    """
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    if a.dtype.kind in 'fc':
        finite = np.abs(b[np.isfinite(b)]) if b.size else b
        atol = float(floor * finite.max()) if finite.size else 0.0
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    else:
        np.testing.assert_array_equal(a, b)


def within_tol(actual, expected, rtol=1e-6):
    try:
        assert_equivalent(actual, expected, rtol)
    except AssertionError:
        return False
    return True
