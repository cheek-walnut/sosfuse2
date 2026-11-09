"""Solver definitions loaded into per-call scopes by accepted.engine.definitions.

The loader executes only the top-level def/class nodes, in a fresh scope per state (filled by accepted.engine.create). The imports and
placeholders below never run there; they name the injected globals for static analysis (ruff, pyright)."""

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
