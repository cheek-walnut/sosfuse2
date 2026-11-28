"""Tiled warm transport over compact ray intervals on Metal."""

from collections import OrderedDict
from functools import lru_cache
import hashlib
import math

import mlx.core as mx
import numpy as np

from ..foundation.top import Domain
from . import mlx_backend, mlx_warm_kernels as msl, numpy_reference as ref
from .arithmetic import cosine, norm_constants, split
from .mlx_ops import _check_extended_profile, _domain_key
from .sh_layout import count


def _device(x, dtype='f4'):
    return mx.array(np.ascontiguousarray(x, dtype))


@lru_cache(maxsize=None)
def _basis_table(L):
    """Cubic Taylor coefficients in polar angle for the associated Legendre functions."""
    intervals = 128 * L
    dt = np.pi / intervals
    theta = np.arange(intervals + 1) * dt

    def samples(x):
        mu, st = np.cos(x), np.sin(x)
        a,b,d=norm_constants(L)
        vals=np.empty((len(x),(L+1)*(L+2)//2))
        diag=np.full(len(x),1 / np.sqrt(4*np.pi))
        for m in range(L+1):
            if m:
                diag=(diag*d[m])*st
            z0=diag
            factor=np.sqrt(2.) if m else 1.
            for l in range(m,L+1):
                k=l*(l+1)//2+m
                if l==m: z=z0
                elif l==m+1: z=z1=np.sqrt(2*m+3)*mu*z0
                else:
                    z=(a[k]*mu)*z1-b[k]*z0
                    z0,z1=z1,z
                vals[:,k]=factor*z
        return vals
    center=samples(theta)
    left=samples(theta-dt); right=samples(theta+dt)
    left2=samples(theta-2*dt); right2=samples(theta+2*dt)
    odd=right-left; odd2=right2-left2
    return _device(np.stack((center, (8*odd-odd2)/12,
                     (16*(right+left-2*center) - (right2+left2-2*center))/24,
                     (odd2-2*odd)/12),axis=-1))


@lru_cache(maxsize=1)
def _flushes_subnormals():
    probe = mx.fast.metal_kernel(
        name='sos_warm_ftz', input_names=['x'], output_names=['out'],
        source='out[0]=x[0]*0.5f; out[1]=x[1]*4.f;')
    values = np.asarray([0x00800000, 0x00400000], 'u4').view('f4')
    output, = probe(inputs=[_device(values)], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                    output_shapes=[(2,)], output_dtypes=[mx.float32])
    return not np.any(np.asarray(output).view('u4'))


def _norms_ml_header(L):
    aa, bb, dd = norm_constants(L)
    entries = []
    for m in range(L + 1):
        for l in range(m, L + 1):
            k = l * (l + 1) // 2 + m
            if l == m:
                x, y = np.float32(dd[m]), np.float32(0.0)
            elif l == m + 1:
                x, y = np.float32(np.sqrt(np.float32(2 * m + 3))), np.float32(0.0)
            else:
                x, y = np.float32(aa[k]), np.float32(bb[k])
            entries.append(f'float2({float(x):.9e}f,{float(y):.9e}f)')
    return f'constant float2 NORMS_ML[{len(entries)}] = {{\n' + ',\n'.join(entries) + '\n};\n'


@lru_cache(maxsize=None)
def _kernel(name, const_L=None):
    inputs, outputs, body = msl.KERNELS[name]
    header = msl.HEADER.replace("NATIVE_FTZ", str(_flushes_subnormals()).lower())
    kname = 'sos_warm_' + name
    if const_L is not None:
        header = header + _norms_ml_header(const_L)
        kname = f'{kname}_L{const_L}'
    return mx.fast.metal_kernel(name=kname, input_names=inputs.split(),
                                output_names=outputs.split(), source=body, header=header)


def _launch(name, inputs, templates, grid, group, outputs):
    const_L = templates['L'] if name == 'vp_vprod_f4_w64' else None
    return _kernel(name, const_L)(inputs=inputs, template=list(templates.items()), grid=grid,
                                  threadgroup=group, output_shapes=[shape for shape, _ in outputs],
                                  output_dtypes=[dtype for _, dtype in outputs])


def _shadow_interval(p, d):
    """Open interval behind the Earth inside the solar cylinder along each ray (km)."""
    r_earth = 6371.0
    rho = np.hypot(p[0], p[1])
    c = (rho - r_earth) * (rho + r_earth)
    a = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]
    b = p[0] * d[:, 0] + p[1] * d[:, 1]
    disc = b * b - a * c
    q = -b - np.copysign(np.sqrt(np.maximum(disc, 0.0)), b)
    first = np.divide(q, a, out=np.zeros_like(a), where=a != 0.0)
    second = np.divide(c, q, out=np.zeros_like(q), where=q != 0.0)
    lo, hi = np.minimum(first, second), np.maximum(first, second)
    axial = a == 0.0
    lo[axial], hi[axial] = -np.inf, np.inf
    plane = np.divide(-p[2], d[:, 2], out=np.zeros_like(a), where=d[:, 2] != 0.0)
    lo = np.where(d[:, 2] < 0.0, np.maximum(lo, plane), lo)
    hi = np.where(d[:, 2] > 0.0, np.minimum(hi, plane), hi)
    active = np.where(axial, c < 0.0, disc > 0.0) & ((d[:, 2] != 0.0) | (p[2] < 0.0)) & (lo < hi)
    bound = float(np.finfo(np.float32).max)
    return (
        np.where(active, np.clip(lo, -bound, bound), 0.0),
        np.where(active, np.clip(hi, -bound, bound), 0.0),
    )


def _shadow_nodes(p, d, edges, lower, quad, counts):
    """Encode the open cylinder interval as two ranks in each compact ray."""
    lo, hi = _shadow_interval(p, d)
    nq = len(quad)
    qu = np.asarray(quad, dtype=np.float64)
    up, down = qu * qu, -((1.0 - qu) ** 2)

    def rank(t, side):
        result = np.zeros(len(t), np.int32)
        after = (t >= edges[:, -1]) & (hi > lo)
        result[after] = counts[after] * nq
        mid = np.flatnonzero((t > edges[:, 0]) & (t < edges[:, -1]) & (hi > lo))
        if len(mid):
            edge = edges[mid]
            j = (edge[:, 1:] <= t[mid, None]).sum(axis=1)
            a, b = edge[np.arange(len(j)), j], edge[np.arange(len(j)), j + 1]
            asc = lower[mid, j]
            x = (t[mid] - np.where(asc, a, b)) / (b - a)
            k = np.where(asc, np.searchsorted(up, x, side), np.searchsorted(down, x, side))
            before = (np.diff(edge, axis=1) > 1e-8) & (np.arange(edge.shape[1] - 1) < j[:, None])
            result[mid] = before.sum(axis=1) * nq + np.where(b - a > 1e-8, k, 0)
        return result

    return np.column_stack((rank(lo, 'right'), rank(hi, 'left')))


def _segment_heights(p, d, e0, e1, anchor_h, domain_key):
    """Local radial origin for interior samples, and FP32 Simpson boundary geometry."""
    top, _, lower_end, air_end = domain_key
    # Measure radial displacement from a nearby shell before casting the anchor
    # to FP32. Adding a tiny step to a pre-rounded absolute height can otherwise
    # put an interior node on the wrong side of a half-ULP rounding threshold.
    shell = np.where(anchor_h < lower_end * 0.5, 0.0, lower_end)
    shell = np.where(abs(anchor_h - air_end) < abs(anchor_h - shell), air_end, shell)
    # SimpsonPrefix splits the original path coordinates, including the FP64
    # segment midpoint. Evaluate its three double-single radii in one batch.
    with ref.flush_to_zero():
        t = np.column_stack((e0, (e0 + e1) / 2, e1))
        radius = ref.path_point(split(p[..., None]), split(d[..., None]), split(t))[2]
        start, inside, end = ref.path_height(radius, Domain(top_km=top)).T
    return anchor_h - shell, shell, start, inside, end


class CompactRays:
    """Host-side ray inputs. Storage depends only on rays and retained intervals."""

    def __init__(self, cells, include_empty=False, upward_only=False):
        # Angular weights can be accumulated in any ray order with the
        # compensated projection. Group similar walk lengths within each cell.
        self.cells = tuple(cells)
        self.include_empty = include_empty
        self.upward_only = upward_only
        self._boundary_cache = {}
        self._shadow_cache = {}
        self._dev_tile_cache = {}
        self._ext_seg_cache = {}
        self._panels_cache = {}
        self._vp_cache = {}
        counts = [np.count_nonzero((g['edges'][:, 1:] - g['edges'][:, :-1]) > 1e-8, axis=1) for g in cells]
        self.order = []
        for g, n in zip(cells, counts):
            mask = np.ones(len(n), bool) if include_empty else (n > 0)
            if upward_only and len(n):
                mask &= (np.asarray(g['dloc'][:, 2], 'f4') > 0) & (np.asarray(g['weight'], 'f4') != 0)
            rays = np.flatnonzero(mask)
            self.order.append(rays[np.argsort(n[rays], kind='stable')].astype('i4'))
        self.cell_off = np.r_[0, np.cumsum([len(q) for q in self.order])].astype('i4')
        ordered_counts = np.concatenate([n[q] for n, q in zip(counts, self.order)]) if cells else np.zeros(0, 'i4')
        self.seg_off = np.r_[0, ordered_counts.cumsum()].astype('i4')
        self.frames = np.empty((len(cells), 14), 'f4')
        self.rays = np.empty((self.cell_off[-1], 9), 'f4')
        self.mu = np.empty(self.cell_off[-1], 'f8')
        self.segments = np.empty((self.seg_off[-1], 2), 'f4')
        self.samples = []
        for i, g in enumerate(cells):
            lo, hi = self.cell_off[i:i+2]
            a, b = self.seg_off[[lo, hi]]
            self.samples.append(int(b-a) * g['nq'])
            if not len(g['d']):
                self.frames[i] = 0.
                continue
            p = np.asarray(g['p'][0], 'f8')
            self.frames[i, :3] = p
            self.frames[i, 3:12] = g['target'][0].T.ravel()
            self.frames[i, 12] = np.linalg.norm(p) - 6371.0
            rho_p = np.hypot(p[0], p[1])
            self.frames[i, 13] = (rho_p - 6371.0) * (rho_p + 6371.0)
            index = self.order[i]
            d = np.asarray(g['d'][index], 'f8')
            ray = self.rays[lo:hi]
            ray[:, :3], ray[:, 3:6] = d, g['dloc'][index]
            ray[:, 6] = np.hypot(d[:, 0], d[:, 1])
            ray[:, 7] = d @ p
            ray[:, 8] = g['weight'][index]
            self.mu[lo:hi] = d[:, 2]
            edges = g['edges'][index]
            e0, e1 = edges[:, :-1], edges[:, 1:]
            span = e1 - e0
            row, col = np.nonzero(span > 1e-8)
            s = span[row, col]
            lower = g['lower'][index][row, col]
            self.segments[a:b, 0] = np.where(lower, e0[row, col], e1[row, col])
            self.segments[a:b, 1] = np.where(lower, s, -s)


    def matches(self, cells):
        return len(cells) == len(self.cells) and all(old is new for old, new in zip(self.cells, cells))

    def tiles(self, limit):
        begin = used = 0
        for i, samples in enumerate(self.samples):
            if i > begin and (used + samples > limit or self.cells[i]['nq'] != self.cells[begin]['nq']):
                yield begin, i
                begin, used = i, 0
            used += samples
        if begin < len(self.cells):
            yield begin, len(self.cells)

    def tile(self, lo, hi):
        co = self.cell_off[lo:hi + 1]
        begin, end = co[[0, -1]]
        ro = self.seg_off[begin:end + 1]
        return co - begin, ro - ro[0], self.rays[begin:end], self.segments[ro[0]:ro[-1]], self.frames[lo:hi]

    def device_tile(self, lo, hi, L, bd, *, need_polar=True):
        key = (lo, hi, L if need_polar else -1)
        cached = self._dev_tile_cache.get(key)
        if cached is not None:
            return cached
        co, ro, host, intervals, frames = self.tile(lo, hi)
        nr, nseg = len(host), len(intervals)
        if not nr:
            return 0, 0, None, None, None, None, None, None
        d_co, d_ro = _device(co, 'i4'), _device(ro, 'i4')
        d_host, d_frames = _device(host), _device(frames)
        d_intervals = _device(intervals) if nseg else bd
        if need_polar:
            polar = (128 * L / np.pi) * np.arccos(np.clip(self.mu[self.cell_off[lo]:self.cell_off[hi]], -1., 1.))
            knot = np.rint(polar).astype('i4')
            d_polar = _device(np.column_stack((knot, polar - knot)))
        else:
            d_polar = None
        result = (nr, nseg, d_co, d_ro, d_host, d_frames, d_intervals, d_polar)
        nbytes = co.nbytes + ro.nbytes + host.nbytes + intervals.nbytes + frames.nbytes + (nr * 8 if need_polar else 0)
        if nbytes <= (8 << 20) and len(self._dev_tile_cache) < 4:
            self._dev_tile_cache[key] = result
        return result

    def shadow_tile(self, lo, hi, u):
        u_arr = np.asarray(u, 'f8')
        key = (lo, hi, u_arr.tobytes())
        cached = self._shadow_cache.get(key)
        if cached is not None:
            return cached
        co = self.cell_off[lo:hi + 1]
        begin, end = int(co[0]), int(co[-1])
        ro = self.seg_off[begin:end + 1] - self.seg_off[begin]
        shadow = np.empty((end - begin, 2), 'i4')
        for i in range(lo, hi):
            g = self.cells[i]
            index = self.order[i]
            if len(index):
                a, b = self.cell_off[i:i + 2] - begin
                shadow[a:b] = _shadow_nodes(g['p'][0], g['d'][index], g['edges'][index],
                                             g['lower'][index], u_arr, np.diff(ro[a:b + 1]))
        d_shadow = _device(shadow, 'i4')
        if shadow.nbytes <= (4 << 20) and len(self._shadow_cache) < 4:
            self._shadow_cache[key] = d_shadow
        return d_shadow

    def ext_seg_tile(self, lo, hi, domain):
        """Local anchors and Simpson boundary geometry for extended domains."""
        if not ref.support_extended(domain):
            return None
        domain_key = _domain_key(domain)
        key = (lo, hi, domain_key)
        cached = self._ext_seg_cache.get(key)
        if cached is not None:
            return cached
        co = self.cell_off[lo:hi + 1]
        begin, end = int(co[0]), int(co[-1])
        ro = self.seg_off[begin:end + 1]
        nseg = int(ro[-1] - ro[0])
        if not nseg:
            return None
        ext = np.empty((nseg, 11), 'f4')
        R = 6371.0
        pos_off = 0
        for i in range(lo, hi):
            g = self.cells[i]
            index = self.order[i]
            if not len(index):
                continue
            d = np.asarray(g['d'][index], 'f8')
            edges = np.asarray(g['edges'][index], 'f8')
            e0, e1 = edges[:, :-1], edges[:, 1:]
            span = e1 - e0
            row, col = np.nonzero(span > 1e-8)
            m = len(row)
            if not m:
                continue
            lo_e = e0[row, col]
            hi_e = e1[row, col]
            asc = g['lower'][index][row, col]
            base = np.where(asc, lo_e, hi_e)
            dr = d[row]
            p = np.asarray(g['p'][index[row]], 'f8')
            pos = p + base[:, None] * dr
            r = np.linalg.norm(pos, axis=1)
            rho = np.hypot(pos[:, 0], pos[:, 1])
            sl = ext[pos_off:pos_off + m]
            sl[:, :3] = pos
            sl[:, 4] = np.sum(pos * dr, axis=1)
            sl[:, 5] = (rho - R) * (rho + R)
            sl[:, 6] = np.sum(pos[:, :2] * dr[:, :2], axis=1)
            sl[:, 3], sl[:, 7], sl[:, 8], sl[:, 9], sl[:, 10] = _segment_heights(
                p, dr, lo_e, hi_e, r - R, domain_key)
            pos_off += m
        d_ext = _device(ext)
        if ext.nbytes <= (8 << 20) and len(self._ext_seg_cache) < 4:
            self._ext_seg_cache[key] = d_ext
        return d_ext

    def boundary_tile(self, lo, hi, constants):
        key = _boundary_key(constants)
        cached = self._boundary_cache.get(key)
        if cached is None:
            if len(self._boundary_cache) >= 2:
                self._boundary_cache.clear()
            all_data, all_indices = _boundary_all(self.cells, self.order, constants)
            dev = (_device(all_data), _device(all_indices, 'i4')) if (all_data.nbytes + all_indices.nbytes) <= (8 << 20) else None
            cached = self._boundary_cache[key] = (all_data, all_indices, dev)
        begin, end = int(self.cell_off[lo]), int(self.cell_off[hi])
        if cached[2] is not None and begin == 0 and end == int(self.cell_off[-1]):
            return cached[2]
        return _device(cached[0][begin:end]), _device(cached[1][begin:end], 'i4')


def _boundary_all(cells, order, constants):
    """Compute (end_data, end_indices) for all cells, fast-pathing monotone hitting rays."""
    from ..rays import column_table

    opt = constants.get('density_optics')
    dom = constants.get('domain')
    strat = 'strato_ext' in constants
    extended = bool(dom is not None and ref.support_extended(dom))
    aligned = True
    if dom is not None and extended:
        step = float(dom.step_km)
        aligned = (
            abs(dom.lower_end / step - round(dom.lower_end / step)) < 1e-9
            and abs(dom.air_end / step - round(dom.air_end / step)) < 1e-9
        )
    fast_ok = (opt is not None or strat) and column_table.method() != 'gl24' and dom is not None and aligned
    if fast_ok:
        R = 6371.0
        step = float(dom.step_km)
        top = float(dom.top_km)
        lower_end = float(dom.lower_end)
        air_end = float(dom.air_end)
        max_end = max(air_end, lower_end) if extended else top
        if opt is not None:
            tab = np.asarray(opt['table'], 'f8')
            mask = int(opt['mask'])
            air0 = float(opt.get('air_surface', tab[0, 0] if mask & 1 else 1.0))
        if strat:
            sp = constants['strato_profile']
            st_center = float(sp['center'])
            st_inv_sigma = 1.0 / float(sp['sigma'])
            st_inv_norm = 1.0 / float(sp['norm'])
        gl0 = 0.5 - 0.5 / math.sqrt(3.0)
        gl1 = 0.5 + 0.5 / math.sqrt(3.0)
    data, indices = [], []
    for g, rays in zip(cells, order):
        if not len(rays):
            data.append(np.empty((0, 7), 'f4'))
            indices.append(np.empty(0, 'i4'))
            continue
        hits = np.asarray(g['hits_ground'][rays], bool)
        idx = np.where(hits, g['end_index'][rays], -1).astype('i4', copy=False)
        frac = np.asarray(g['end_fraction'][rays], 'f4')
        stcol = np.zeros(len(rays), 'f4')
        if opt is None and not strat:
            cols = np.asarray(g['end_columns'][rays], 'f4')
        elif fast_ok:
            cols = np.array(g['end_columns'][rays], 'f4', copy=True)
            end_all = np.asarray(g['edges'][rays, -1], 'f8')
            active = hits & (end_all > 1e-8)
            if np.any(active):
                p0 = np.asarray(g['p'][0], 'f8')
                d_hit = np.asarray(g['d'][rays[active]], 'f8')
                end_hit = end_all[active]
                r0 = float(np.linalg.norm(p0))
                b = d_hit @ p0
                q_end = p0[None, :] + end_hit[:, None] * d_hit
                r_end = np.linalg.norm(q_end, axis=1)
                if np.all(b < 0.0) and (R - 1e-5 <= r0 <= R + top + 1e-5) and np.all(r_end <= R + 1e-2):
                    h0 = max(0.0, r0 - R)
                    k_lim = (len(tab) - 1) if opt is not None else max(1, int(math.ceil(top / step)))
                    k_max = min(k_lim, max(1, int(math.ceil(min(h0, max_end) / step))))
                    hk = np.arange(k_max + 1, dtype='f8') * step
                    Rk = R + hk
                    delta = (r0 - Rk) * (r0 + Rk)
                    out4 = np.zeros((len(d_hit), 4), 'f8') if opt is not None else None
                    out_st = np.zeros(len(d_hit), 'f8') if strat else None
                    if opt is not None:
                        t_lo_a = tab[:k_max]
                        t_d = tab[1 : k_max + 1] - t_lo_a
                    for start in range(0, len(d_hit), 4096):
                        sl = slice(start, start + 4096)
                        bs = b[sl]
                        es = end_hit[sl]
                        rad = np.sqrt(np.maximum(0.0, bs[:, None] * bs[:, None] - delta[None, :]))
                        tk = np.where(delta[None, :] > 0.0, np.clip(delta[None, :] / (-bs[:, None] + rad), 0.0, es[:, None]), 0.0)
                        tk[:, 0] = es
                        dt = np.maximum(0.0, tk[:, :-1] - tk[:, 1:])
                        t_lo = tk[:, 1:]
                        w_dt = 0.5 * dt
                        for gq in (gl0, gl1):
                            tq = t_lo + gq * dt
                            dq = tq * (2.0 * bs[:, None] + tq)
                            rq = np.sqrt(np.maximum(0.0, r0 * r0 + dq))
                            hq = np.clip(h0 + dq / (rq + r0), 0.0, top)
                            if extended:
                                in_air = hq <= air_end
                                in_low = hq <= lower_end
                            if opt is not None:
                                fk = np.clip((hq - hk[:-1][None, :]) / step, 0.0, 1.0)
                                air = (t_lo_a[:, 0] + fk * t_d[:, 0]) if (mask & 1) else np.exp(-hq / 8.0)
                                o3 = (t_lo_a[:, 1] + fk * t_d[:, 1]) if (mask & 2) else np.maximum(0.0, 1.0 - np.abs(hq - 25.0) / 15.0)
                                no2 = t_lo_a[:, 2] + fk * t_d[:, 2]
                                if extended:
                                    air = np.where(in_air, air, 0.0)
                                    o3 = np.where(in_low, o3, 0.0)
                                    no2 = np.where(in_low, no2, 0.0)
                                    o4 = np.where(in_low, (air / air0) ** 2, 0.0)
                                else:
                                    o4 = (air / air0) ** 2
                                if mask & 1:
                                    out4[sl, 0] += np.sum(w_dt * air, axis=1)
                                if mask & 2:
                                    out4[sl, 1] += np.sum(w_dt * o3, axis=1)
                                out4[sl, 2] += np.sum(w_dt * no2, axis=1)
                                out4[sl, 3] += np.sum(w_dt * o4, axis=1)
                            if strat:
                                tt = (hq - st_center) * st_inv_sigma
                                st_val = np.exp(-0.5 * (tt * tt)) * st_inv_norm
                                if extended:
                                    st_val = np.where(in_low, st_val, 0.0)
                                out_st[sl] += np.sum(w_dt * st_val, axis=1)
                    if opt is not None:
                        if mask & 1:
                            cols[active, 0] = out4[:, 0]
                        if mask & 2:
                            cols[active, 2] = out4[:, 1]
                        cols[active, 3] = out4[:, 2]
                        cols[active, 4] = out4[:, 3]
                    if strat:
                        stcol[active] = out_st.astype('f4')
                else:
                    full_cols, full_st = ref.boundary_end_columns(g, constants, strat)
                    cols = np.asarray(full_cols[rays], 'f4')
                    if strat and full_st is not None:
                        stcol = np.asarray(full_st[rays], 'f4')
        else:
            full_cols, full_st = ref.boundary_end_columns(g, constants, strat)
            cols = np.asarray(full_cols[rays], 'f4')
            if strat and full_st is not None:
                stcol = np.asarray(full_st[rays], 'f4')
        row_data = np.empty((len(rays), 7), 'f4')
        row_data[:, :5] = cols
        row_data[:, 5] = frac
        row_data[:, 6] = stcol
        data.append(row_data)
        indices.append(idx)
    all_data = np.concatenate(data) if data else np.empty((0, 7), 'f4')
    all_indices = np.concatenate(indices) if indices else np.empty(0, 'i4')
    return all_data, all_indices


def _boundary_key(constants):
    from ..rays import column_table

    opt = constants.get('density_optics')
    strat = 'strato_ext' in constants
    st_key = None
    if strat:
        sp = constants['strato_profile']
        st_key = (float(sp['center']), float(sp['sigma']), float(sp['norm']), sp.get('end_km'))
    dom = getattr(opt.get('old'), 'domain', constants.get('domain')) if opt is not None else constants.get('domain')
    dom_key = None if dom is None else (dom.top_km, dom.step_km, dom.lower_end, dom.air_end)
    if opt is None:
        return (column_table.method(), None, dom_key, st_key)
    return (
        column_table.method(),
        int(opt['mask']),
        opt.get('air_surface'),
        dom_key,
        opt['table'].shape,
        hashlib.blake2b(np.ascontiguousarray(opt['table']).tobytes(), digest_size=16).digest(),
        st_key,
    )


_HG_TABLE_CACHE = OrderedDict()
_ST_TABLE_CACHE = OrderedDict()
_HG_TABLE_CACHE_MAX = 4
_HG_TABLE_CACHE_BYTES = 32 << 20
_HG_TABLE_LIMIT_BYTES = 256 << 20
_STRATO_LUT_CACHE = OrderedDict()
_STRATO_LUT_CACHE_MAX = 8


def _strato_lut(chi, W):
    """Build and cache the 1D Mie phase LUT [16385, W] on device from strato_chi [33, W]."""
    arr = np.ascontiguousarray(chi, 'f4')
    key = (int(W), arr.shape, hashlib.blake2b(arr.tobytes(), digest_size=16).digest())
    cached = _STRATO_LUT_CACHE.get(key)
    if cached is not None:
        _STRATO_LUT_CACHE.move_to_end(key)
        return cached
    total = 16385 * int(W)
    grid_x = ((total + 255) // 256) * 256
    lut, = _launch(
        'strato_lut',
        [_device(arr)],
        dict(W=int(W)),
        (grid_x, 1, 1),
        (256, 1, 1),
        [((16385, int(W)), mx.float32)],
    )
    _STRATO_LUT_CACHE[key] = lut
    while len(_STRATO_LUT_CACHE) > _STRATO_LUT_CACHE_MAX:
        _STRATO_LUT_CACHE.popitem(last=False)
    return lut


def _ground_csr(c):
    """Extract (gm, goff, gdir, gval, has_ground) from constants in CSR form."""
    nh, nc, W = len(c['h']), len(c['chi']), int(c['W'])
    ns = nh * nc
    gm = np.ascontiguousarray(c['gm'], 'f4') if 'gm' in c else np.zeros((ns, 4, W), 'f4')
    if 'goff' in c and 'gdir' in c and 'gval' in c:
        goff = np.ascontiguousarray(c['goff'], 'i4')
        gdir = np.ascontiguousarray(c['gdir'], 'f4')
        gval = np.ascontiguousarray(c['gval'], 'f4')
    elif 'gd' in c and 'gv' in c:
        gd = np.asarray(c['gd'], 'f4')
        gv = np.asarray(c['gv'], 'f4')
        if gv.size == 0 or not np.any(gv):
            goff = np.zeros(ns + 1, 'i4')
            gdir = np.zeros((0, 3), 'f4')
            gval = np.zeros((0, W), 'f4')
        else:
            keep = np.any(gv != 0, axis=2)
            counts = keep.sum(axis=1, dtype='i4')
            goff = np.r_[0, np.cumsum(counts, dtype='i4')].astype('i4')
            gdir = np.ascontiguousarray(gd[keep], 'f4')
            gval = np.ascontiguousarray(gv[keep], 'f4')
    else:
        goff = np.zeros(ns + 1, 'i4')
        gdir = np.zeros((0, 3), 'f4')
        gval = np.zeros((0, W), 'f4')
    has_gm = bool(np.any(gm))
    has_gv = bool(int(goff[-1]) > 0 and np.any(gval))
    return gm, goff, gdir, gval, has_gm or has_gv


def _ground_device(c, phys_dev):
    """Prepare device ground buffers, using angular tables when cost-effective."""
    nh, nc, W = len(c['h']), len(c['chi']), int(c['W'])
    ns = nh * nc
    strat = 'strato_ext' in c
    gm, goff, gdir, gval, has_ground = _ground_csr(c)
    zero_f = _device([0.0])
    zero_i = _device([0], 'i4')
    if not has_ground:
        return dict(
            has_ground=False, hg_table=False, gm=zero_f, cell_gflag=zero_i,
            g_idx=zero_i, g_f0=zero_f, g_f1=zero_f, st_lut=zero_f, st_table=zero_f,
        )
    cell_nz = (np.any(gm != 0, axis=(1, 2)) | (np.diff(goff) > 0)).reshape(nh, nc)
    flag_grid = np.zeros((nh, nc), 'i4')
    if nh > 1 and nc > 1:
        flag_grid[:-1, :-1] = (
            cell_nz[:-1, :-1] | cell_nz[:-1, 1:] | cell_nz[1:, :-1] | cell_nz[1:, 1:]
        ).astype('i4')
    counts = np.diff(goff)
    owners = np.flatnonzero(counts).astype('i4')
    n_active = len(owners)
    max_ng = int(counts.max()) if n_active else 0
    hg_nodes = 37
    table_bytes = n_active * hg_nodes * hg_nodes * W * 4
    st_lut = _strato_lut(c['strato_chi'], W) if (strat and n_active) else zero_f
    st_smooth_ok = True
    if strat:
        chi_arr = np.asarray(c['strato_chi'], 'f4')
        st_smooth_ok = bool(
            float(np.max(np.abs(chi_arr[1]) / 3.0)) <= 0.78
            and float(np.max(np.abs(chi_arr[16:]))) <= 0.05
        )
    need_bytes = (2 if strat else 1) * table_bytes
    use_table = bool(
        n_active and max_ng >= 16 and abs(float(c['g'])) <= 0.78 and st_smooth_ok and need_bytes <= _HG_TABLE_LIMIT_BYTES
    )
    if use_table:
        goff_h = hashlib.blake2b(goff.tobytes(), digest_size=16).digest()
        gdir_h = hashlib.blake2b(gdir.tobytes(), digest_size=16).digest()
        gval_h = hashlib.blake2b(gval.tobytes(), digest_size=16).digest()
        key = (float(c['albedo']), float(c['g']), W, ns, goff_h, gdir_h, gval_h)
        cached = _HG_TABLE_CACHE.get(key)
        d_owners = d_goff = d_gdir = d_gval = None
        if cached is not None:
            _HG_TABLE_CACHE.move_to_end(key)
            d_table, d_hg_map = cached
        else:
            hg_map = np.full(ns, -1, 'i4')
            hg_map[owners] = np.arange(n_active, dtype='i4')
            d_hg_map = _device(hg_map, 'i4')
            d_owners, d_goff, d_gdir, d_gval = _device(owners, 'i4'), _device(goff, 'i4'), _device(gdir), _device(gval)
            wg = (W + 15) // 16
            grid_x = ((hg_nodes + 7) // 8) * 128
            d_table, = _launch(
                'hg_table',
                [d_owners, d_goff, d_gdir, d_gval, phys_dev],
                dict(W=W, WG=wg),
                (grid_x, hg_nodes, n_active),
                (128, 1, 1),
                [((n_active, hg_nodes, hg_nodes, W), mx.float32)],
            )
            if table_bytes <= _HG_TABLE_CACHE_BYTES:
                _HG_TABLE_CACHE[key] = (d_table, d_hg_map)
                while len(_HG_TABLE_CACHE) > _HG_TABLE_CACHE_MAX:
                    _HG_TABLE_CACHE.popitem(last=False)
        d_st_table = zero_f
        if strat:
            chi_h = hashlib.blake2b(np.ascontiguousarray(c['strato_chi'], 'f4').tobytes(), digest_size=16).digest()
            skey = (float(c['albedo']), W, ns, chi_h, goff_h, gdir_h, gval_h)
            scached = _ST_TABLE_CACHE.get(skey)
            if scached is not None:
                _ST_TABLE_CACHE.move_to_end(skey)
                d_st_table = scached
            else:
                if d_owners is None:
                    d_owners, d_goff, d_gdir, d_gval = _device(owners, 'i4'), _device(goff, 'i4'), _device(gdir), _device(gval)
                wg = (W + 15) // 16
                grid_x = ((hg_nodes + 7) // 8) * 128
                d_st_table, = _launch(
                    'mie_table',
                    [d_owners, d_goff, d_gdir, d_gval, st_lut],
                    dict(W=W, WG=wg),
                    (grid_x, hg_nodes, n_active),
                    (128, 1, 1),
                    [((n_active, hg_nodes, hg_nodes, W), mx.float32)],
                )
                if table_bytes <= _HG_TABLE_CACHE_BYTES:
                    _ST_TABLE_CACHE[skey] = d_st_table
                    while len(_ST_TABLE_CACHE) > _HG_TABLE_CACHE_MAX:
                        _ST_TABLE_CACHE.popitem(last=False)
        return dict(
            has_ground=True, hg_table=True, gm=_device(gm), cell_gflag=_device(flag_grid.ravel(), 'i4'),
            g_idx=d_hg_map, g_f0=d_table, g_f1=d_st_table, st_lut=st_lut,
        )
    return dict(
        has_ground=True, hg_table=False, gm=_device(gm), cell_gflag=_device(flag_grid.ravel(), 'i4'),
        g_idx=_device(goff, 'i4'),
        g_f0=_device(gdir if len(gdir) else np.zeros((1, 3), 'f4')),
        g_f1=_device(gval if len(gval) else np.zeros((1, W), 'f4')),
        st_lut=st_lut,
    )


_CONST_CACHE = OrderedDict()
_CONST_CACHE_MAX = 8


def _arr_key(x):
    arr = x if (isinstance(x, np.ndarray) and x.dtype == np.float32 and x.flags.c_contiguous) else np.ascontiguousarray(x, 'f4')
    raw = arr.tobytes()
    return (arr.shape, raw if len(raw) <= 1024 else hashlib.blake2b(raw, digest_size=16).digest())


def _const_base_key(c, mask, air0, optics_table):
    dom = c['domain']
    strat = 'strato_ext' in c
    combine = bool(strat and c.get('sh_combine', False))
    skip_thin = bool(combine and c.get('sh_combine_skip', False))
    parts = [
        dom.top_km, dom.step_km, dom.lower_end, dom.air_end,
        int(c['L']), int(c['W']), int(mask), float(air0),
        float(c['height']), float(c['mix']), float(c['g']),
        optics_table.shape,
        hashlib.blake2b(np.ascontiguousarray(optics_table).tobytes(), digest_size=16).digest(),
        c['aer2'].tobytes() if 'aer2' in c else None,
        strat, combine, skip_thin,
    ]
    for k in ('h', 'chi', 'u', 'wu', 'hr', 'ha', 'ext'):
        parts.append(_arr_key(c[k]))
    if strat:
        sp = c['strato_profile']
        parts.append((float(sp['center']), float(sp['sigma']), float(sp['norm']), sp.get('end_km')))
        for k in ('strato_ext', 'strato_density'):
            parts.append(_arr_key(c[k]))
    return tuple(parts)


def _constants(c, first=False, scheduling=None):
    from ..atmosphere.density import LEGACY, build

    domain = c['domain']
    if ref.support_extended(domain):
        _check_extended_profile(c)
    optics = c.get('density_optics')
    if optics is None:
        optics = build(dict(density=dict(LEGACY)))
    mask = optics['mask'] if 'density_optics' in c else 0
    air0 = optics.get('air_surface', optics['table'][0, 0] if optics['mask'] & 1 else 1.)
    strat = 'strato_ext' in c
    combine = bool(strat and c.get('sh_combine', False))
    skip_thin = bool(combine and c.get('sh_combine_skip', False))
    bkey = _const_base_key(c, mask, air0, optics['table'])
    base = _CONST_CACHE.get(bkey)
    if base is not None:
        _CONST_CACHE.move_to_end(bkey)
    else:
        aa, bb, dd = norm_constants(c['L'])
        ext_arr = np.asarray(c['ext'], 'f4')
        if strat:
            ext_arr = np.vstack([ext_arr, np.asarray(c['strato_ext'], 'f4')[None, :]])
        nonneg_ext = bool(np.all(ext_arr >= 0))
        ext_min = np.min(ext_arr, axis=1) if nonneg_ext else np.zeros(len(ext_arr), 'f4')
        if strat:
            sp = c['strato_profile']
            st_p = [float(sp['center']), 1.0 / float(sp['sigma']), 1.0 / float(sp['norm'])]
        else:
            st_p = [0.0, 1.0, 0.0]
        lo_f32 = np.float32(domain.lower_end)
        air_f32 = np.float32(domain.air_end)
        phys_dev = _device([
            domain.top_km, 1 / domain.step_km, 1 / c['height'], 1 / air0, c['mix'], c['g'], *st_p,
            float(lo_f32), float(air_f32),
            float(np.nextafter(lo_f32, np.float32(-np.inf))),
            float(np.nextafter(lo_f32, np.float32(np.inf))),
            float(np.nextafter(air_f32, np.float32(-np.inf))),
            float(np.nextafter(air_f32, np.float32(np.inf))),
        ])
        hr_arr = np.asarray(c['hr'], 'f4')
        ha_arr = np.asarray(c['ha'], 'f4')
        ha_eff = np.ones_like(ha_arr) if combine else ha_arr
        if strat and not combine:
            hs_arr = np.asarray(c['strato_density'], 'f4')
            dens_arr = np.vstack([np.stack([hr_arr, ha_eff], axis=1), np.stack([hr_arr, hs_arr], axis=1)])
        else:
            dens_arr = np.stack([hr_arr, ha_eff], axis=1)
        base = dict(
            profile=_device(optics['table']), heights=_device(c['h']), angles=_device(c['chi']),
            quad=_device(np.stack([c['u'], c['wu']], axis=1)),
            physics=phys_dev,
            layer=_device(c.get('aer2', np.zeros(7))), shnorm=_device(np.r_[aa, bb, dd]),
            densities=_device(dens_arr),
            ext=_device(ext_arr),
            ext_min=_device(ext_min),
            nonneg_ext=nonneg_ext,
            strato=strat,
            sh_combine=combine,
            bkey=bkey,
            templates=dict(NH=len(c['h']), NC=len(c['chi']), NQ=len(c['u']), L=c['L'], W=c['W'],
                           MASK=int(mask), LAYER='aer2' in c, NT=len(optics['table']), STRATO=strat,
                           EXTENDED=bool(ref.support_extended(domain))))
        if strat:
            hs_arr = np.asarray(c['strato_density'], 'f4')
            base['densities2'] = _device(np.stack([hr_arr, hs_arr], axis=1))
            base['st_dens'] = _device(hs_arr)
            base['st_comb'] = _device(np.stack([hr_arr, ha_arr, hs_arr], axis=1))
            base['hr_dev'] = _device(hr_arr)
            base['ha_dev'] = _device(ha_arr)
            base['hs_dev'] = base['st_dens']
        _CONST_CACHE[bkey] = base
        while len(_CONST_CACHE) > _CONST_CACHE_MAX:
            _CONST_CACHE.popitem(last=False)
    if scheduling is None:
        ha_live = np.asarray(c['ha'], 'f4')
        if strat:
            hs_live = np.asarray(c['strato_density'], 'f4')
            if skip_thin and len(ha_live) <= 128:
                cutoff = np.float32(1e-7) * np.asarray(c['hr'], 'f4')
                ha_live = np.where(ha_live > cutoff, ha_live, np.float32(0.0))
                hs_live = np.where(hs_live > cutoff, hs_live, np.float32(0.0))
            live_mask = (ha_live != 0) | (hs_live != 0)
        else:
            live_mask = ha_live != 0
    else:
        live_mask = scheduling
    live_ir = np.flatnonzero(live_mask).astype('i4')
    live_map = np.full(len(c['h']), -1, 'i4')
    live_map[live_ir] = np.arange(len(live_ir), dtype='i4')
    out = dict(
        base,
        templates=dict(base['templates']),
        live_map=_device(live_map, 'i4'),
        live_ir=_device(live_ir if len(live_ir) else np.zeros(1, 'i4'), 'i4'),
        live_key=live_ir.tobytes(),
        nlive=int(len(live_ir)),
    )
    if first:
        out['sun_h'] = _device(c['sun_h'])
        if strat:
            sc_comb = np.concatenate(
                [np.asarray(c['sun_col'], 'f4'), np.asarray(c['strato_sun_col'], 'f4')[..., None]], axis=-1
            )
            out['sun_col'] = _device(sc_comb)
            out['asca'] = _device(np.vstack([np.asarray(c['asca'], 'f4'), np.asarray(c['strato_sca'], 'f4')]))
        else:
            out['sun_col'] = _device(c['sun_col'])
            out['asca'] = _device(c['asca'])
        out['templates']['NSH'] = len(c['sun_h'])
        out['templates']['NSX'] = int(c['sun_col'].shape[1])
        out.update(_ground_device(c, base['physics']))
        if strat:
            st_chi = _device(c['strato_chi'])
            if out['has_ground'] and not out['hg_table'] and out['st_lut'].ndim == 2:
                out['st_phase'] = mx.concatenate([st_chi, out['st_lut']], axis=0)
            else:
                out['st_phase'] = st_chi
        else:
            out['st_phase'] = _device([0.0])
    return out


def _project_dev(offsets, compact, basis, azimuth, weighted, nc, L, W, mode=0, premul=False):
    K = count(L)
    nwq = (W + 3) // 4
    tg = min(256, ((4 + K + 31) // 32) * 32)
    rows = 1 if mode == 2 else (5 + K if mode == 1 else 4 + K)
    proj_mode = _device([mode], 'i4')
    moment, = _launch('project', [offsets, basis, azimuth, weighted, compact, proj_mode],
                      dict(W=W, K=K, L=L, PREMUL=premul, TG=tg),
                      (nwq * tg, nc, 1), (tg, 1, 1),
                      [((nc, rows, W), mx.float32)])
    return moment


def _reduce_tile(offsets, compact, basis, azimuth, weighted, nc, L, W, albedo, premul=False, ground_only=False):
    if ground_only:
        if albedo == 0:
            return None, np.zeros((nc, W), 'f8')
        f = _project_dev(offsets, compact, basis, azimuth, weighted, nc, L, W, mode=2, premul=premul)
        return None, np.asarray(f, 'f8')[:, 0, :] * (float(albedo) / math.pi)
    if albedo == 0:
        m = _project_dev(offsets, compact, basis, azimuth, weighted, nc, L, W, mode=0, premul=premul)
        return np.asarray(m, 'f8'), np.zeros((nc, W), 'f8')
    out = _project_dev(offsets, compact, basis, azimuth, weighted, nc, L, W, mode=1, premul=premul)
    host = np.asarray(out, 'f8')
    return host[:, :-1, :], host[:, -1, :] * (float(albedo) / math.pi)


def _project(offsets, basis, azimuth, weighted, nc, L, W, premul=False):
    dummy = _device([0.0])
    return np.asarray(_project_dev(offsets, dummy, basis, azimuth, weighted, nc, L, W, mode=0, premul=premul)).astype('f8')


def project_cell(g, c, radiance, basis):
    """Project first-light ray radiances using the same angular reduction."""
    L, W = c['L'], c['W']
    if not len(g['d']):
        return np.zeros((4 + count(L), W), 'f8')
    d = g['d']
    transverse = np.hypot(d[:, 0], d[:, 1])
    cp = np.divide(d[:, 0], transverse, out=np.ones_like(transverse), where=transverse > 1e-12)
    cm = _device(cosine(cp)[:, :L+1])
    stokes = radiance.transpose(2, 0, 1)
    trace = (stokes[0] + stokes[1]) + stokes[2]
    weighted = mx.concatenate([stokes, trace[None, :, :]]) * _device(g['weight'])[None, :, None]
    return _project(_device([0, len(d)], 'i4'), _device(basis), cm, weighted, 1, L, W)[0]


@lru_cache(maxsize=4)
def _pack_sh_index(L):
    ml_idx = mx.array([l * (l + 1) // 2 + m for m in range(L + 1) for l in range(m, L + 1)], dtype=mx.int32)
    m_scale = mx.array([1.0 if m == 0 else np.sqrt(2.0) for m in range(L + 1) for _ in range(m, L + 1)],
                       dtype=mx.float32)
    return ml_idx, m_scale


def _combine_sh_device(raw_source, c, dc=None):
    """Combine tropospheric and stratospheric SH planes on device in FP32."""
    K = count(c['L'])
    nh, nc = len(c['h']), len(c['chi'])
    ha = dc['ha_dev'] if (dc is not None and 'ha_dev' in dc) else _device(c['ha'])
    hs = dc['hs_dev'] if (dc is not None and 'hs_dev' in dc) else _device(c['strato_density'])
    if c.get('sh_combine_skip', False) and nh <= 128:
        hr = dc['hr_dev'] if (dc is not None and 'hr_dev' in dc) else _device(c['hr'])
        cutoff = mx.array(np.float32(1e-7), dtype=mx.float32) * hr
        zero = mx.array(0.0, dtype=mx.float32)
        ha = mx.where(ha > cutoff, ha, zero)
        hs = mx.where(hs > cutoff, hs, zero)
    ha_cell = mx.broadcast_to(ha[:, None], (nh, nc)).reshape(nh * nc, 1, 1)
    hs_cell = mx.broadcast_to(hs[:, None], (nh, nc)).reshape(nh * nc, 1, 1)
    a = raw_source[:, 10 : 10 + K, :] * ha_cell
    b = raw_source[:, 10 + K : 10 + 2 * K, :] * hs_cell
    return a + b


def _pack_sh_plane(sh_plane, L, W):
    K = count(L)
    ns = sh_plane.shape[0]
    if L <= 16 and W <= 16:
        ml_idx, m_scale = _pack_sh_index(L)
        sh = mx.take(sh_plane, ml_idx, axis=1) * m_scale[None, :, None]
        if W < 16:
            sh = mx.pad(sh, [(0, 0), (0, 0), (0, 16 - W)])
        return mx.flatten(sh).reshape(ns, K, 16)
    return mx.flatten(sh_plane).reshape(ns, K, W)


def _pack_source(raw_source, L, W, c=None, dc=None):
    K = count(L)
    ns = raw_source.shape[0]
    src_rayleigh = mx.flatten(raw_source[:, :4, :]).reshape(ns, 4, W)
    strat = c is not None and 'strato_ext' in c
    combine = bool(strat and c.get('sh_combine', False))
    if combine:
        sh_plane = mx.flatten(_combine_sh_device(raw_source, c, dc)).reshape(ns, K, W)
        src_sh = _pack_sh_plane(sh_plane, L, W)
        return src_rayleigh, src_sh, sh_plane
    sh_plane = mx.flatten(raw_source[:, 10 : 10 + K, :]).reshape(ns, K, W)
    src_sh = _pack_sh_plane(sh_plane, L, W)
    if strat:
        sh_plane2 = mx.flatten(raw_source[:, 10 + K : 10 + 2 * K, :]).reshape(ns, K, W)
        src_sh2 = _pack_sh_plane(sh_plane2, L, W)
        return src_rayleigh, src_sh, sh_plane, src_sh2, sh_plane2
    return src_rayleigh, src_sh, sh_plane


@lru_cache(maxsize=2)
def _angular_grid(L, intervals):
    from .arithmetic import basis
    theta = np.arange(intervals + 1) * (np.pi / intervals)
    y = basis(np.cos(theta), L)
    m = np.asarray([m for l in range(L + 1) for m in range(l + 1)])
    az = np.cos(theta[:, None] * m).astype('f4')
    return _device((y[:, None, :] * az[None, :, :]).reshape(-1, len(m)))


def _angular_field(sh_plane, c, intervals):
    n, _, w = sh_plane.shape
    sh = sh_plane.transpose(1, 0, 2).reshape(-1, n * w)
    return mx.matmul(_angular_grid(c['L'], intervals), sh)


def _source_bounds_live(field, source, c):
    # By the addition theorem, ||Y||_2 <= (L+1)/sqrt(4*pi). A max-norm
    # bound avoids squaring subnormal or very large SH coefficients.
    K, nc = count(c['L']), len(c['chi'])
    nh = len(c['h'])
    factor = np.float32(1.001 * (c['L'] + 1) * np.sqrt(K / (4 * np.pi)))
    shape = (nh, nc, c['W'])
    strat = 'strato_ext' in c
    combine = bool(strat and c.get('sh_combine', False))
    skip_thin = bool(combine and c.get('sh_combine_skip', False))
    if field.size <= 2_000_000:
        f = np.asarray(field, 'f4')
        hr = np.asarray(c['hr'], 'f4')
        ha = np.asarray(c['ha'], 'f4')
        ray_scale = np.max(np.abs(f[:, :4, :]), axis=1).reshape(shape) * np.abs(hr).reshape(-1, 1, 1)
        if strat:
            hs = np.asarray(c['strato_density'], 'f4')
            if skip_thin and nh <= 128:
                cutoff = np.float32(1e-7) * hr
                ha = np.where(ha > cutoff, ha, np.float32(0.0))
                hs = np.where(hs > cutoff, hs, np.float32(0.0))
            c1 = np.max(np.abs(f[:, 10 : 10 + K, :]), axis=1)
            c2 = np.max(np.abs(f[:, 10 + K : 10 + 2 * K, :]), axis=1)
            bound = (
                (c1 * factor).reshape(shape) * np.abs(ha).reshape(-1, 1, 1)
                + (c2 * factor).reshape(shape) * np.abs(hs).reshape(-1, 1, 1)
            )
        else:
            coeff = np.max(np.abs(f[:, 10 : 10 + K, :]), axis=1)
            bound = (coeff * factor).reshape(shape) * np.abs(ha).reshape(-1, 1, 1)
        return np.any(bound > np.float32(2.0**-23) * ray_scale, axis=(1, 2))
    src_rayleigh, _, sh_plane = source[:3]
    ray_scale = mx.max(mx.abs(src_rayleigh), axis=1).reshape(shape) * mx.abs(_device(c['hr'])).reshape(-1, 1, 1)
    coeff = mx.max(mx.abs(sh_plane), axis=1)
    if combine:
        bound = (coeff * factor).reshape(shape)
    elif strat:
        sh_plane2 = source[4]
        coeff2 = mx.max(mx.abs(sh_plane2), axis=1)
        bound = (
            (coeff * factor).reshape(shape) * mx.abs(_device(c['ha'])).reshape(-1, 1, 1)
            + (coeff2 * factor).reshape(shape) * mx.abs(_device(c['strato_density'])).reshape(-1, 1, 1)
        )
    else:
        bound = (coeff * factor).reshape(shape) * mx.abs(_device(c['ha'])).reshape(-1, 1, 1)
    return np.asarray(mx.any(bound > (2.0**-23) * ray_scale, axis=(1, 2)))


def _tile(packed, lo, hi, c, source, boundary, dc, bd, capacity=None, *, ground_only=False, return_flux=False):
    nc = hi - lo
    W, L = c['W'], c['L']
    K = count(L)
    strat = bool(dc.get('strato', False))
    strat_sep = bool(strat and not dc.get('sh_combine', False))
    nr, nseg, d_co, d_ro, d_host, d_frames, d_intervals, d_polar = packed.device_tile(lo, hi, L, bd, need_polar=True)
    ns = nseg * len(c['u'])
    if not nr:
        m = None if ground_only else np.zeros((nc, 4 + K, W), 'f8')
        f = np.zeros((nc, W), 'f8')
        return (m, f) if return_flux else m
    if boundary is not None:
        end_data, end_indices = packed.boundary_tile(lo, hi, c)
    else:
        end_data, end_indices = bd, _device([0], 'i4')
    bd_flag = _device([1 if boundary is not None else 0], 'i4')
    max_rays, max_nodes, max_segments = capacity or (nr, ns, nseg)
    if isinstance(source, tuple):
        src_rayleigh, src_sh, sh_plane = source[:3]
        src_sh2, sh_plane2 = (source[3], source[4]) if len(source) == 5 else (None, None)
    else:
        packed_src = _pack_source(source, L, W, c, dc)
        src_rayleigh, src_sh, sh_plane = packed_src[:3]
        src_sh2, sh_plane2 = (packed_src[3], packed_src[4]) if len(packed_src) == 5 else (None, None)
    template = dc['templates']
    panel_lanes = min(32, 1 << template['NQ'].bit_length())
    use_f4 = L <= 16 and W <= 16
    wp = 16 if use_f4 else ((W + 3) // 4) * 4
    cos_shape = (1,) if use_f4 else (max_rays, L + 1)
    nseg_cols = 6 if strat else 5
    # Reuse source-independent basis/panels outputs for compact reduced grids only (<= 8 MiB).
    panel_bytes = (
        max_rays * (K + 29 + (0 if use_f4 else L + 1)) * 4
        + max_nodes * (12 + 2 + (1 if strat else 0)) * 4
        + max_segments * (nseg_cols + 2) * 4
    )
    can_cache_panels = ns > 0 and panel_bytes <= (8 << 20) and capacity == (nr, ns, nseg)
    pkey = (
        (lo, hi, L, use_f4, boundary is not None, _boundary_key(c) if boundary is not None else None, dc['bkey'])
        if can_cache_panels
        else None
    )
    d_seg_ext = packed.ext_seg_tile(lo, hi, c['domain']) if ns else None
    seg_ext_in = d_seg_ext if d_seg_ext is not None else bd
    pcached = packed._panels_cache.get(pkey) if pkey is not None else None
    if pcached is not None:
        y, cosine, d_ray, point, c0cp, segment_sum, owners, st_acc, seg_depth = pcached
    else:
        y, cosine, d_ray = _launch('basis', [d_host, dc['shnorm'], d_co, d_frames, end_data, bd_flag],
                                 dict(L=L, PREMUL=use_f4, NEED_EXP=True),
                                 (L + 1, nr, 1), (L + 1, 4, 1),
                                 [((max_rays, K), mx.float32), (cos_shape, mx.float32), ((max_rays, 29), mx.float32)])
        if ns:
            point, c0cp, segment_sum, owners, st_acc = _launch(
                'panels', [d_ray, d_ro, d_intervals, dc['quad'], dc['heights'], dc['angles'],
                           dc['profile'], dc['physics'], dc['layer'], seg_ext_in], dict(template, P=panel_lanes),
                (nseg * panel_lanes, 1, 1), (128, 1, 1),
                [((max_nodes, 12), mx.float32), ((max_nodes, 2), mx.int32),
                 ((max_segments, nseg_cols), mx.float32), ((max_segments,), mx.int32),
                 ((max_nodes,) if strat else (1,), mx.float32)])
            seg_depth, = _launch(
                'seg_prefix', [d_ro, segment_sum, dc['ext_min']], dict(STRATO=strat),
                (nr, 1, 1), (min(128, nr), 1, 1), [((max_segments,), mx.float32)])
        else:
            point, c0cp, segment_sum, owners, st_acc, seg_depth = bd, end_indices, bd, end_indices, bd, bd
        if pkey is not None:
            if len(packed._panels_cache) >= 2:
                packed._panels_cache.clear()
            packed._panels_cache[pkey] = (y, cosine, d_ray, point, c0cp, segment_sum, owners, st_acc, seg_depth)
    if ns:
        nlcell = dc['nlive'] * template['NC']
        if dc.get('angular') is not None:
            aero, inv_slot = dc['angular'], end_indices
        elif nlcell:
            vpkey = (pkey, dc['live_key']) if pkey is not None else None
            vpcached = packed._vp_cache.get(vpkey) if vpkey is not None else None
            if vpcached is not None:
                sorted_rcp, inv_slot, bstart, blk, blk_cap = vpcached
            else:
                nb = 256
                seg_n = (ns + nb - 1) // nb
                meta = _device([ns, seg_n], 'i4')
                counts, = _launch(
                    'vp_count', [c0cp, dc['live_map'], seg_depth, meta],
                    dict(NLCELL=nlcell, NC=template['NC'], NB=nb, NQ=template['NQ'], EXTENDED=template['EXTENDED']),
                    (nb * 256, 1, 1), (256, 1, 1), [((nlcell, nb), mx.int32)])
                flat = counts.reshape(-1)
                inc = mx.cumsum(flat)
                offs = inc - flat
                sorted_rcp, inv_slot = _launch(
                    'vp_scatter', [c0cp, owners, dc['live_map'], seg_depth, offs, meta],
                    dict(NLCELL=nlcell, NC=template['NC'], NB=nb, NQ=template['NQ'], EXTENDED=template['EXTENDED']),
                    (nb * 32, 1, 1), (32, 1, 1), [((4 * max_nodes, 2), mx.uint32), ((4 * max_nodes,), mx.int32)])
                blk_recs = 1024
                blk_cap = nlcell + (4 * max_nodes + blk_recs - 1) // blk_recs
                bstart, blk = _launch(
                    'vp_blocks', [offs, inc, dc['live_ir']],
                    dict(NLCELL=nlcell, NC=template['NC'], NB=nb, BLK=blk_recs),
                    (256, 1, 1), (256, 1, 1), [((nlcell + 1,), mx.int32), ((blk_cap, 3), mx.int32)])
                if vpkey is not None:
                    if len(packed._vp_cache) >= 2:
                        packed._vp_cache.clear()
                    packed._vp_cache[vpkey] = (sorted_rcp, inv_slot, bstart, blk, blk_cap)
            if use_f4:
                ntg = 256
                aero, = _launch(
                    'vp_vprod_f4_w64', [bstart, blk, sorted_rcp, d_host, src_sh, dc['densities']],
                    dict(L=L, W=W, NC=template['NC'], NLCELL=nlcell, NTG=ntg),
                    (blk_cap * ntg, 1, 1), (ntg, 1, 1), [((4 * max_nodes, wp), mx.float32)])
                if strat_sep:
                    saero, = _launch(
                        'vp_vprod_f4_w64', [bstart, blk, sorted_rcp, d_host, src_sh2, dc['densities2']],
                        dict(L=L, W=W, NC=template['NC'], NLCELL=nlcell, NTG=ntg),
                        (blk_cap * ntg, 1, 1), (ntg, 1, 1), [((4 * max_nodes, wp), mx.float32)])
                    aero = mx.concatenate([aero, saero], axis=1)
            else:
                wchunk = 16 if K * 16 * 4 <= 28672 else 8
                aero, = _launch(
                    'vp_vprod', [bstart, blk, sorted_rcp, y, src_sh, dc['densities']],
                    dict(L=L, W=W, WP=wp, NC=template['NC'], NLCELL=nlcell, WCHUNK=wchunk),
                    (blk_cap * 128, 1, 1), (128, 1, 1), [((4 * max_nodes, wp), mx.float32)])
                if strat_sep:
                    saero, = _launch(
                        'vp_vprod', [bstart, blk, sorted_rcp, y, src_sh2, dc['densities2']],
                        dict(L=L, W=W, WP=wp, NC=template['NC'], NLCELL=nlcell, WCHUNK=wchunk),
                        (blk_cap * 128, 1, 1), (128, 1, 1), [((4 * max_nodes, wp), mx.float32)])
                    aero = mx.concatenate([aero, saero], axis=1)
        else:
            aero, inv_slot = bd, end_indices
    else:
        # The transport loop is empty, but a ray may still see its boundary.
        point, c0cp, segment_sum, st_acc, aero, inv_slot = bd, end_indices, bd, bd, bd, end_indices
    wave_lanes = 1 << (W - 1).bit_length()
    raw_sh_in = mx.concatenate([sh_plane, sh_plane2], axis=0) if strat_sep else sh_plane
    inputs = [d_ray, d_ro, point, c0cp, segment_sum, src_rayleigh, aero, inv_slot,
              dc['densities'], dc['live_map'], dc['ext'], dc['physics'], bd, end_indices,
              d_frames, d_co, dc['angles'], raw_sh_in, d_intervals, dc['quad'], d_polar, _basis_table(L),
              st_acc, seg_ext_in]
    params = dict(W=W, WP=wp, NC=template['NC'], NQ=template['NQ'], L=L,
                  NONNEG_EXT=dc['nonneg_ext'], BOUNDS='bounds' in dc, ANG=dc.get('ang_bins', 0),
                  NS=len(c['h']) * len(c['chi']), BD=boundary is not None,
                  STRATO=strat, STRATO_SEP=strat_sep, EXTENDED=template['EXTENDED'])
    grid = (wave_lanes, nr, 1)
    group = (min(128, wave_lanes), 8, 1)
    shape = (5, max_rays, W)
    weighted, flags = _launch('integrate', inputs + [bd, end_indices], dict(params, REFINE=False),
                              grid, group, [(shape, mx.float32), ((nr, W), mx.int32)])
    weighted, _ = _launch('integrate', inputs + [weighted, flags], dict(params, REFINE=True),
                           grid, group, [(shape, mx.float32), ((1,), mx.int32)])
    m, f = _reduce_tile(d_co, d_host, y, cosine, weighted, nc, L, W, c['albedo'],
                        premul=use_f4, ground_only=ground_only)
    return (m, f) if return_flux else m


def _tile_first(packed, lo, hi, c, boundary, dc, bd, capacity=None, *, ground_only=False):
    nc = hi - lo
    W, L = c['W'], c['L']
    K = count(L)
    strat = bool(dc.get('strato', False))
    nr, nseg, d_co, d_ro, d_host, d_frames, d_intervals, _ = packed.device_tile(lo, hi, L, bd, need_polar=False)
    if not nr:
        return (None if ground_only else np.zeros((nc, 4 + K, W), 'f8')), np.zeros((nc, W), 'f8')
    d_shadow = packed.shadow_tile(lo, hi, c['u'])
    if boundary is not None:
        end_data, end_indices = packed.boundary_tile(lo, hi, c)
    else:
        end_data, end_indices = bd, _device([0], 'i4')
    bd_flag = _device([1 if boundary is not None else 0], 'i4')
    max_rays = capacity[0] if capacity else nr
    if ground_only:
        y, cosine = bd, bd
    else:
        y, cosine, _ = _launch('basis', [d_host, dc['shnorm'], d_co, d_frames, end_data, bd_flag],
                               dict(L=L, PREMUL=True, NEED_EXP=False),
                               (L + 1, nr, 1), (L + 1, 4, 1),
                               [((max_rays, K), mx.float32), ((1,), mx.float32), ((1,), mx.float32)])
    template = dc['templates']
    wg = (W + 15) // 16
    grid_x = ((nr + 1) // 2) * 32
    d_seg_ext = packed.ext_seg_tile(lo, hi, c['domain'])
    seg_ext_in = d_seg_ext if d_seg_ext is not None else bd
    weighted, = _launch(
        'first',
        [d_host, d_frames, d_co, d_ro, d_intervals, d_shadow, dc['quad'], dc['sun_h'], dc['sun_col'],
         dc['profile'], dc['physics'], dc['layer'], dc['ext'], dc['asca'], bd, end_data, end_indices,
         dc['heights'], dc['angles'], dc['gm'], dc['cell_gflag'], dc['g_idx'], dc['g_f0'], dc['g_f1'],
         dc['st_phase'], seg_ext_in],
        dict(W=W, WG=wg, NQ=template['NQ'], NSH=template['NSH'], NSX=template['NSX'],
             NH=template['NH'], NC=template['NC'], MASK=template['MASK'], LAYER=template['LAYER'],
             NT=template['NT'], BD=boundary is not None, GROUND=dc['has_ground'], HG_TABLE=dc['hg_table'],
             STRATO=strat, EXTENDED=template['EXTENDED']),
        (grid_x, 1, 1), (min(128, grid_x), 1, 1), [((5, max_rays, W), mx.float32)])
    return _reduce_tile(d_co, d_host, y, cosine, weighted, nc, L, W, c['albedo'],
                        premul=True, ground_only=ground_only)


def evaluate(cells, constants, field, boundary, packed=None, *, first=False, ground_only=False):
    """Each completed tile returns its reduced FP32 moments and Lambert flux in float64."""
    c = constants[0]
    strat = 'strato_ext' in c
    combine = bool(strat and c.get('sh_combine', False))
    expected = (len(c['h']) * len(c['chi']), 10 + count(c['L']) * (2 if strat else 1), c['W'])
    if not first and field.shape != expected:
        raise ValueError(f"expected source shape {expected}, received {field.shape}")
    if boundary is not None and np.shape(boundary) != (len(c['chi']), c['W']):
        raise ValueError("the boundary must have one spectrum per solar angle")
    boundary = boundary if boundary is not None and np.any(boundary) else None
    bd = _device([0.] if boundary is None else boundary)
    W, K = c['W'], count(c['L'])
    moments = np.zeros((len(cells), 4 + K, W), 'f8')
    flux = np.zeros((len(cells), W), 'f8')
    if packed is None:
        packed = CompactRays(cells, include_empty=boundary is not None, upward_only=ground_only)
    # Fixed physical capacities let the allocator reuse the preceding tile's
    # buffers. Every kernel dispatch and reduction still uses the actual counts.
    tiles = list(packed.tiles(mlx_backend.tile_samples()))
    devices = {}
    if first:
        source = None
    else:
        c0 = constants[0]
        live0 = _source_bounds_live(field, None, c0) if field.size <= 2_000_000 else None
        dc0 = _constants(c0, scheduling=live0) if live0 is not None else None
        raw_source = _device(field)
        source = _pack_source(raw_source, c0['L'], c0['W'], c0, dc0)
        if len(tiles) > 1:
            mx.eval(*source)
        del raw_source
        if dc0 is not None:
            devices[id(c0)] = dict(dc0, bounds=True)
    max_rays = max((int(packed.cell_off[b] - packed.cell_off[a]) for a, b in tiles), default=0)
    max_nodes = max((sum(packed.samples[a:b]) for a, b in tiles), default=0)
    max_segments = max((int(packed.seg_off[packed.cell_off[b]] - packed.seg_off[packed.cell_off[a]]) for a, b in tiles), default=0)
    capacity = (max_rays, max_nodes, max_segments)
    wp = 16 if (c['L'] <= 16 and W <= 16) else ((W + 3) // 4) * 4
    ray_bytes = 4 * (9 + 29 + K + c['L'] + 1 + 5 * W + 1)
    extended = bool(ref.support_extended(c['domain']))
    tile_node_bytes = 0 if first else (108 + 16 * wp * (2 if (strat and not combine) else 1)) * max_nodes + (80 if extended else 36) * max_segments
    pool_limit = int(1.5 * (tile_node_bytes + ray_bytes * max_rays)) + (moments[0].nbytes if len(cells) else 0) * max((b - a for a, b in tiles), default=1)
    ang_bins = 6 * c['L']
    angular_bytes = (ang_bins + 1)**2 * field.shape[0] * W * 4 if not first else 0
    # Build the angle table only when the ray workload can amortize its construction.
    use_angular = (not first and not (strat and not combine) and c['L'] <= 16 and W <= 16 and angular_bytes < 3 * 2**30
                   and sum(packed.samples) >= field.shape[0] * (ang_bins + 1)**2)
    for lo, hi in tiles:
        c = constants[lo]
        if id(c) not in devices:
            if first:
                devices[id(c)] = _constants(c, first=True)
            else:
                live = _source_bounds_live(field, source, c)
                devices[id(c)] = dict(_constants(c, scheduling=live), bounds=True)
        if not first and use_angular and 'angular' not in devices[id(c)]:
            table = _angular_field(source[2], c, ang_bins)
            mx.eval(table)
            devices[id(c)].update(angular=table, ang_bins=ang_bins)
        if first:
            m_tile, f_tile = _tile_first(packed, lo, hi, c, boundary, devices[id(c)], bd, capacity, ground_only=ground_only)
        else:
            m_tile, f_tile = _tile(packed, lo, hi, c, source, boundary, devices[id(c)], bd, capacity,
                                   ground_only=ground_only, return_flux=True)
        if not ground_only:
            moments[lo:hi] = m_tile
        flux[lo:hi] = f_tile
        # Finish deferred Metal buffer releases and bound cached free blocks.
        if len(tiles) > 1 or mx.get_cache_memory() > max(pool_limit, 32 << 20):
            mx.synchronize()
            if hi == len(cells) or mx.get_cache_memory() > pool_limit:
                mx.clear_cache()
    return moments, flux
