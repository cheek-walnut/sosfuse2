"""Experimental tropospheric transport on the normal-height domain.

Host preparation retains the reference double-single geometry and Simpson optical
depth. MLX contracts the source, rotates and interpolates the four corner tensors,
and integrates the attenuated field and Lambert boundary along each ray.
"""

import mlx.core as mx
import numpy as np

from . import numpy_reference as ref
from .arithmetic import basis
from .sh_layout import count


def _ftz(x):
    """Match the reference's FP32 flush at underflow-sensitive products and sums."""
    return mx.where(mx.abs(x) < float(ref.TRACE_MIN), 0.0, x)


class WarmTransport:
    """Reusable quadrature for one cell's rays; arrays follow the caller's MLX stream."""

    def __init__(self, geometry, constants, *, first_light=False):
        g, c = geometry, constants
        if 'strato_ext' in c or ref.support_extended(c['domain']):
            raise ValueError("WarmTransport supports only the lower-atmosphere source")
        self.order = c['L']
        self.field_shape = (len(c['h']) * len(c['chi']), 10 + count(self.order), c['W'])
        self.mix = float(c['mix'])
        self.nrays = len(g['d'])
        with ref.flush_to_zero():
            self.basis = basis(g['d'][:, 2], self.order)
            _, _, pu, direct = ref.frame_terms(g)
            self.pu = mx.array(pu)
            if first_light:
                if c["albedo"] != 0 or np.any(c["gm"]) or np.any(c["gv"]):
                    raise ValueError("First-light transport requires a black ground boundary")
                if "tropo_lut" in c:
                    raise ValueError("First-light transport requires the Henyey-Greenstein phase function")
                self.direct = mx.array(direct)
                self.phase = mx.array(ref.hg(g["d"][:, 2].astype("f4"), c["g"]))
                self.ext = mx.array(c["ext"])
                self.asca = mx.array(c["asca"])
            self._nodes(g, c, first_light)
            end_col = ref.boundary_end_columns(g, c, False)[0]
            depth = np.zeros((self.nrays, c['W']), 'f4')
            for col, ext in zip(end_col.T, c['ext'], strict=True):
                depth += col.astype('f4')[:, None] * ext
            self.end_depth = mx.array(depth)
            self.end_index = mx.array(g['end_index'])
            self.end_fraction = mx.array(g['end_fraction'].astype('f4'))[:, None]
            self.hits = mx.array(g['hits_ground'])[:, None]

    def _nodes(self, g, c, first_light):
        # Compact valid nodes. The dense index restores the reference path order
        # for the compensated sum without constructing a source tensor per ray step.
        depths, lengths = [], []
        indices, cells = [], []
        weights, rotations = [], []
        cosines, heights = [], []
        columns, visibility, profiles = [], [], []
        prefix = ref.SimpsonPrefix(g, c)
        steps = 0
        for seg in range(g['edges'].shape[1] - 1):
            prefix.begin(seg)
            rays = np.flatnonzero(prefix.ok)
            if not rays.size:
                continue
            for step in range(len(c['u'])):
                q = ref.sample(g, seg, step, c)
                if first_light:
                    col, lit, _ = ref.sun_columns(q, c)
                    columns.append(col[rays])
                    visibility.append(lit[rays])
                    profiles.append(q["prof"][rays, :2])
                depths.append(prefix.node(q)[rays])
                lengths.append(q['ds'][rays])
                indices.append(steps * self.nrays + rays)
                cells.append(q['cells'][rays])
                weights.append(q['sw'][rays])
                rotations.append(q['T'][rays])
                cosines.append(q['cm'][rays])
                heights.append(q['ir'][rays, None] + np.array([0, 0, 1, 1]))
                steps += 1
            prefix.finish()
        self.steps = steps
        if not steps:
            return
        index = np.concatenate(indices)
        ray = index % self.nrays
        cell = np.concatenate(cells)
        ncell = self.field_shape[0]
        pairs, inverse = np.unique(ray[:, None] * ncell + cell, return_inverse=True)
        self.pair_cell = mx.array((pairs % ncell).astype('i4'))
        self.pair_basis = mx.array(self.basis[pairs // ncell])
        self.lookup = mx.array(inverse.reshape(cell.shape).astype('i4'))
        self.index = mx.array(index.astype('i4'))
        self.rays = mx.array(ray.astype('i4'))
        self.cell = mx.array(cell.astype('i4'))
        self.weights = mx.array(np.concatenate(weights))
        self.rotation = mx.array(np.concatenate(rotations))
        self.cosines = mx.array(np.concatenate(cosines))
        height = np.concatenate(heights)
        self.hr = mx.array(c['hr'][height])
        self.ha = mx.array(c['ha'][height])
        self.ds = mx.array(np.concatenate(lengths))
        self.depth = mx.array(np.concatenate(depths))
        if first_light:
            self.sun_columns = mx.array(np.concatenate(columns))
            self.lit = mx.array(np.concatenate(visibility))[:, None]
            self.profile = mx.array(np.concatenate(profiles))

    def _first_scattered(self):
        optical = mx.zeros_like(self.depth)
        for k in range(self.ext.shape[0]):
            optical = optical + self.sun_columns[:, k, None] * self.ext[k]
        sun = mx.exp(-optical) * self.lit
        beam = (self.profile[:, 1, None] * self.asca) * (sun * self.phase[self.rays, None])
        ray = (sun * self.profile[:, 0, None]) * self.ext[0]
        pu = self.pu[self.rays, None, :]
        directional = self.mix * float(np.float32(3 / (8 * np.pi))) * self.direct[self.rays, None, :]
        isotropic = (1 - self.mix) * float(np.float32(1 / (8 * np.pi))) * pu
        return ray[..., None] * (directional + isotropic) + beam[..., None] * pu / 2

    def _scattered(self, field):
        # Accumulate degrees in increasing order within each azimuthal order,
        # once per distinct (ray, source cell) pair.
        source = field[self.pair_cell, 10:, :]
        angular = mx.zeros((source.shape[0], 33, source.shape[-1]))
        for degree in range(self.order + 1):
            start = degree * (degree + 1) // 2
            band = slice(start, start + degree + 1)
            term = self.pair_basis[:, band, None] * source[:, band, :]
            angular = mx.concatenate((angular[:, :degree + 1] + term, angular[:, degree + 1:]), axis=1)
        modes = self.cosines[:, None, :, None] * angular[self.lookup]
        modes = mx.concatenate((modes[:, :, :1] + modes[:, :, 32:], modes[:, :, 1:32]), axis=2)
        for stride in (16, 8, 4, 2, 1):
            modes = modes[:, :, :stride] + modes[:, :, stride:]
        aerosol = modes[:, :, 0] * self.ha[:, :, None]

        src = field[self.cell, :4, :].transpose(0, 1, 3, 2)
        rotated = _ftz(self.rotation[:, None, None, :, 0] * src[:, :, :, None, 0])
        for k in range(1, 4):
            term = _ftz(self.rotation[:, None, None, :, k] * src[:, :, :, None, k])
            rotated = _ftz(rotated + term)
        trace = (src[..., 0] + src[..., 1]) + src[..., 2]
        pu = self.pu[self.rays, None, None, :]
        isotropic = _ftz((1 - self.mix) / 3 * trace)
        polarized = _ftz(_ftz(self.mix * rotated) + _ftz(isotropic[..., None] * pu))
        tensor = _ftz(self.hr[:, :, None, None] * polarized)
        tensor = _ftz(tensor + _ftz(_ftz(aerosol[..., None] * pu) / 2))

        trace = (tensor[..., 0] + tensor[..., 1]) + tensor[..., 2]
        valid = trace >= float(ref.TRACE_MIN)
        safe = mx.where(valid, trace, 1.0)
        w = mx.where(valid, self.weights[:, :, None], 0.0)

        def corners(x):
            return ((x[:, 0] + x[:, 1]) + x[:, 2]) + x[:, 3]

        wg = corners(w)
        am = corners(_ftz(w * safe))
        good = (wg > 0) & (am >= float(ref.TRACE_MIN))
        mean = mx.where(good, am, 1.0) / mx.where(good, wg, 1.0)
        scale = mx.exp(mx.where(good, corners(w * mx.log(safe)) - wg * mx.log(mean), 0.0))
        linear = corners(_ftz(self.weights[:, :, None, None] * tensor))
        return _ftz(linear * scale[..., None])

    def __call__(self, field, boundary=None, *, first=False):
        if field.shape != self.field_shape:
            raise ValueError(f"expected source shape {self.field_shape}, received {field.shape}")
        waves = field.shape[-1]
        total = mx.zeros((self.nrays, waves, 4))
        if self.steps:
            attenuation = _ftz(self.ds[:, None] * _ftz(mx.exp(-self.depth)))
            scattered = self._first_scattered() if first else self._scattered(field)
            weighted = _ftz(scattered * attenuation[..., None])
            dense = mx.zeros((self.steps * self.nrays, waves, 4))
            dense[self.index] = weighted
            correction = mx.zeros_like(total)
            for panel in dense.reshape(self.steps, self.nrays, waves, 4):
                y = _ftz(panel - correction)
                updated = _ftz(total + y)
                correction = _ftz(_ftz(updated - total) - y)
                total = updated
        if boundary is not None:
            frac = self.end_fraction
            b = boundary.astype(mx.float32)
            end = (1 - frac) * b[self.end_index] + frac * b[self.end_index + 1]
            end = end * (mx.exp(-self.end_depth) * self.hits)
            total = total + _ftz(_ftz(end[..., None] * self.pu[:, None, :]) / 2)
        return _ftz(total)


class FixtureTransport:
    """Metal transport adapter for the fixed lower-atmosphere, black-ground fixture."""

    def __init__(self):
        self.groups = {}
        self.extinction = None
        self.preparations = 0
        self.calls = {"first": 0, "multiple": 0}

    def __call__(self, g, c, field, first=False, boundary=None, statistics=None, return_basis=False):
        # A new spectral chunk has different optical columns and coefficients.
        if self.extinction is None or not np.array_equal(self.extinction, c["ext"]):
            self.groups.clear()
            self.extinction = c["ext"]
        # Cell and observer quadratures may use different constants in one chunk.
        # Keep the constants alive for as long as their identity keys are cached.
        if id(c) not in self.groups:
            self.groups[id(c)] = (c, {})
        _, cache = self.groups[id(c)]
        # Queries rebuild the same small ray groups on successive applies.
        # Include every geometry input, also the observer frame and quadrature.
        key = tuple((name, value.shape, value.dtype.str, value.tobytes())
                    for name, value in sorted(g.items()) if isinstance(value, np.ndarray))
        if key not in cache:
            cache[key] = WarmTransport(g, c, first_light=True)
            self.preparations += 1
        op = cache[key]
        self.calls["first" if first else "multiple"] += 1
        b = None if boundary is None else mx.array(boundary.astype("f4"))
        result = op(mx.array(field), b, first=first)
        mx.eval(result)
        values = np.asarray(result)
        return (values, op.basis) if return_basis else values
