"""One quadrature walk for independent columns, retaining each reduction shape.

No optical prefixes are needed for endpoint columns. Segment order, GL nodes,
XYZ summation order and the legacy 3 / density 2 / strato scalar reductions
match the engine geometry_segments and its consumers exactly.
"""

import numpy as np
from .components import NCOMP, MOLECULAR_COLUMNS


def segments(p, d, old, n, edges=None):
    if edges is None:
        edges = old.probe.boundaries(p, d)
        end = edges[:, -1]
        dot = np.sum(p * d, -1)
        pp = np.sum(p * p, -1)
        more = []
        for height in (10.0, 25.0, 40.0):
            disc = dot * dot - pp + (6371.0 + height) ** 2
            for sign in (-1, 1):
                root = -dot + sign * np.sqrt(np.maximum(disc, 0))
                more.append(np.where((disc > 0) & (root > 0) & (root < end), root, end))
        edges = np.sort(np.column_stack([edges, *more]), axis=-1)
    u, wu = old.probe.gl(n)
    for k in range(edges.shape[1] - 1):
        active = np.flatnonzero(edges[:, k + 1] - edges[:, k] > 1e-8)
        if not len(active):
            continue
        a, b = edges[active, k, None], edges[active, k + 1, None]
        pa, da = p[active], d[active]
        ra = np.sum((pa + a * da) ** 2, -1)
        rb = np.sum((pa + b * da) ** 2, -1)
        lower = (ra <= rb)[:, None]
        t = np.where(lower, a + (b - a) * u * u, b - (b - a) * (1 - u) ** 2)
        ds = np.where(lower, 2 * (b - a) * u * wu, 2 * (b - a) * (1 - u) * wu)
        # Equivalent to np.sum(q*q, -1) for a trailing axis of length three.
        radius2 = np.square(pa[:, None, 0] + t * da[:, None, 0])
        for axis in (1, 2):
            radius2 += np.square(pa[:, None, axis] + t * da[:, None, axis])
        yield active, np.sqrt(radius2), ds


def integrate(p, d, old, n=24, *, legacy=False, optics=None, profile=None, edges=None, main=False):
    if main:
        from . import column_table

        if column_table.method() != 'gl24':
            return column_table.integrate(p, d, old, legacy=legacy, optics=optics, profile=profile, edges=edges)
    from .density import molecular_profiles
    from .stratosphere import density

    p = np.broadcast_to(p, np.shape(d))
    base = np.zeros((len(d), NCOMP)) if legacy else None
    corrected = np.zeros((len(d), 4)) if optics is not None else None
    strato = np.zeros(len(d)) if profile is not None else None
    for active, r, ds in segments(p, d, old, n, edges):
        h = r - 6371.0
        if base is not None:
            base[active] += np.sum(old.profiles(h) * ds[..., None], axis=1)
        if corrected is not None:
            corrected[active] += np.sum(molecular_profiles(h, optics) * ds[..., None], axis=1)
        if strato is not None:
            strato[active] += np.sum(density(h, profile) * ds, axis=1)
    if corrected is not None:
        from .density import exact_profile_columns

        end = old.probe.boundaries(p, d)[:, -1] if edges is None else edges[:, -1]
        corrected[:, 2:] = exact_profile_columns(p, d, end, optics, selected=(2, 3))[:, 2:]
    return base, corrected, strato


def replace(base, corrected, optics):
    out = np.array(base, copy=True)
    if corrected is not None:
        for k, col in enumerate(MOLECULAR_COLUMNS):
            if k >= 2 or optics['mask'] & (1 << k):
                out[:, col] = corrected[:, k]
    return out
