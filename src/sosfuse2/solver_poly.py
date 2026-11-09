"""Four-point Lagrange interpolation polynomials over the zenith-angle grid (CPU reference engine)."""

import numpy as np


def chi_poly(ch):
    poly = np.empty((len(ch) - 1, 4, 4))
    left = np.empty(len(ch) - 1, 'i4')
    for j in range(len(ch) - 1):
        lo = min(max(j - 1, 0), len(ch) - 4)
        left[j] = lo
        x = (ch[lo : lo + 4] - ch[j]) / (ch[j + 1] - ch[j])
        for c in range(4):
            p = np.polynomial.Polynomial([1.0])
            for k in range(4):
                if k != c:
                    p *= np.polynomial.Polynomial([-x[k], 1]) / (x[c] - x[k])
            poly[j, c] = p.coef
    return poly, left
