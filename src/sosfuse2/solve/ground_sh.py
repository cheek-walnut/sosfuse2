"""Opt-in first-ground convolution in the mirror-even real SH basis.

The stored ground values already contain quadrature weights. Their mirrored
directions each carry half that value, so sine modes vanish and cosine modes
need no extra factor of two. Orthonormal SH convolution eigenvalues are g**l
for HG and 2*pi*integral(phase(u)*P_l(u),du) for a per-steradian LUT.
"""

import os
import numpy as np
from ..backends.arithmetic import basis

GRID_INTERVALS = 32


def configured_order():
    text = os.environ.get('SOSFUSE2_GROUND_SH_L', '0').strip()
    if text not in ('0', '16', '24', '32'):
        raise ValueError('E_GROUND_SH_L: expected 0,16,24,32')
    return int(text)


def real_basis(directions, order):
    d = np.asarray(directions, 'f8')
    y = basis(d[:, 2], order).astype('f8')
    # atan2 keeps polar directions finite and avoids an inconsistent sqrt(1-z*z)
    # when input unit vectors were rounded to FP32 independently.
    phi = np.arctan2(d[:, 1], d[:, 0])
    m = np.array([m for l in range(order + 1) for m in range(l + 1)])
    return y * np.cos(phi[:, None] * m)


def lut_eigenvalues(lut, order):
    """Integrate the actual piecewise-linear device LUT, without renormalizing.

    Gauss order ceil((L+2)/2) integrates each linear segment times P_L exactly
    in real arithmetic. This preserves the builder's FP32 node values.
    """
    lut = np.asarray(lut, 'f8')
    if lut.ndim != 2 or len(lut) < 2 or not np.isfinite(lut).all():
        raise ValueError('E_GROUND_SH_LUT: finite [node,wavelength] required')
    bins = len(lut) - 1
    nodes, weights = np.polynomial.legendre.leggauss((order + 3) // 2)
    out = np.zeros((order + 1, lut.shape[1]), 'f8')
    left = -1 + 2 * np.arange(bins) / bins
    for x, weight in zip(nodes, weights):
        fraction = (x + 1) / 2
        u = left + 2 * fraction / bins
        phase = lut[:-1] * (1 - fraction) + lut[1:] * fraction
        p = np.polynomial.legendre.legvander(u, order)
        out += (2 * np.pi * weight / bins) * (p.T @ phase)
    return out


def project(goff, gdir, gval, g, strato_eigen, order, rows_per_block=1 << 18):
    """[cell,wavelength,mode,phase] FP32 coefficients; FP64 projection sums.

    The ground quadrature repeats a fixed angular rule, so the directions of a
    state are a few thousand distinct rows reused across cells. The basis is
    evaluated once per distinct row and gathered per cell: the same per-element
    arithmetic and the same per-cell matmul as evaluating it cell by cell.
    Rows are visited in blocks of whole cells so that the basis of a state with
    few repeated directions stays below about rows_per_block*modes*8 bytes.
    """
    levels = np.array([l for l in range(order + 1) for m in range(l + 1)])
    k = len(levels)
    cells, waves = len(goff) - 1, gval.shape[1]
    out = np.zeros((cells, waves, k, 2), 'f4')
    if not gval.size:
        return out
    d = np.ascontiguousarray(gdir)
    rows = d.view(f'V{3 * d.dtype.itemsize}').ravel()
    values = np.asarray(gval, 'f8')
    scale = float(g) ** levels[:, None]
    eigen = strato_eigen[levels]
    first = 0
    while first < cells:
        last = first + 1
        while last < cells and goff[last + 1] - goff[first] <= rows_per_block:
            last += 1
        start, stop = goff[first], goff[last]
        unique, row = np.unique(rows[start:stop], return_inverse=True)
        y = real_basis(unique.view(d.dtype).reshape(-1, 3), order)
        for cell in range(first, last):
            a, b = goff[cell] - start, goff[cell + 1] - start
            if a == b:
                continue
            moments = y[row[a:b]].T @ values[start + a : start + b]
            out[cell, :, :, 0] = (moments * scale).T
            out[cell, :, :, 1] = (moments * eigen).T
        first = last
    return out


def angular_basis(order, intervals=GRID_INTERVALS):
    """Include one low and one high ghost node for cubic interpolation."""
    angle = (np.arange(intervals + 3) - 1) * (np.pi / intervals)
    theta, phi = np.meshgrid(angle, angle, indexing='ij')
    dirs = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1).reshape(-1, 3)
    return np.ascontiguousarray(real_basis(dirs, order), 'f4')


def synthesize(coeff, order, xp, intervals=GRID_INTERVALS):
    """Batched matrix multiply, bounded to 64 cells per GPU launch.

    The table is a cache of the SH convolution, not a new directional ground
    quadrature. Coefficients and angular interpolation remain state-private.
    """
    cells, waves, modes, _ = coeff.shape
    y = xp.asarray(angular_basis(order, intervals))
    result = xp.empty((cells, (intervals + 3) ** 2, waves * 2), 'f4')
    for start in range(0, cells, 64):
        end = min(cells, start + 64)
        c = xp.asarray(
            np.ascontiguousarray(coeff[start:end].transpose(0, 2, 1, 3).reshape(end - start, modes, waves * 2))
        )
        result[start:end] = y @ c
    return result
