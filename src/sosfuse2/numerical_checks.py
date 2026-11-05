"""Numerical equivalence for research harnesses (not file-integrity checks)."""
import numpy as np


def assert_equivalent(actual, expected, rtol=1e-6):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    if a.dtype.kind in 'fc':
        np.testing.assert_allclose(a, b, rtol=rtol, atol=0, equal_nan=False)
    else:
        np.testing.assert_array_equal(a, b)


def within_tol(actual, expected, rtol=1e-6):
    try:
        assert_equivalent(actual, expected, rtol)
    except AssertionError:
        return False
    return True
