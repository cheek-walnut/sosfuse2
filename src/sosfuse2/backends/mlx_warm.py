"""Tiled warm transport over compact ray intervals on Metal."""

from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import mlx_backend, mlx_warm_kernels as msl, numpy_reference as ref
from .arithmetic import cosine, norm_constants
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


class CompactRays:
    """Host-side ray inputs. Storage depends only on rays and retained intervals."""

    def __init__(self, cells, include_empty=False):
        # Angular weights can be accumulated in any ray order with the
        # compensated projection. Group similar walk lengths within each cell.
        self.cells = tuple(cells)
        self.include_empty = include_empty
        counts = [np.count_nonzero((g['edges'][:, 1:] - g['edges'][:, :-1]) > 1e-8, axis=1) for g in cells]
        self.order = []
        for n in counts:
            rays = np.arange(len(n), dtype='i4') if include_empty else np.flatnonzero(n)
            self.order.append(rays[np.argsort(n[rays], kind='stable')].astype('i4'))
        self.cell_off = np.r_[0, np.cumsum([len(q) for q in self.order])].astype('i4')
        ordered_counts = np.concatenate([n[q] for n, q in zip(counts, self.order)]) if cells else np.zeros(0, 'i4')
        self.seg_off = np.r_[0, ordered_counts.cumsum()].astype('i4')
        self.frames = np.empty((len(cells), 13), 'f4')
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


def _boundary(cells, order, constants):
    data, indices = [], []
    for g, rays in zip(cells, order):
        columns, _ = ref.boundary_end_columns(g, constants, False)
        data.append(np.column_stack([columns, g['end_fraction']])[rays])
        indices.append(np.where(g['hits_ground'], g['end_index'], -1)[rays])
    return _device(np.concatenate(data)), _device(np.concatenate(indices), 'i4')


def _constants(c, first=False, scheduling=None):
    from ..atmosphere.density import LEGACY, build

    optics = c.get('density_optics')
    if optics is None:
        optics = build(dict(density=dict(LEGACY)))
    mask = optics['mask'] if 'density_optics' in c else 0
    air0 = optics.get('air_surface', optics['table'][0, 0] if optics['mask'] & 1 else 1.)
    domain = c['domain']
    aa, bb, dd = norm_constants(c['L'])
    live_mask = (np.asarray(c['ha']) != 0) if scheduling is None else scheduling
    live_ir = np.flatnonzero(live_mask).astype('i4')
    live_map = np.full(len(c['h']), -1, 'i4')
    live_map[live_ir] = np.arange(len(live_ir), dtype='i4')
    ext_arr = np.asarray(c['ext'], 'f4')
    nonneg_ext = bool(np.all(ext_arr >= 0))
    ext_min = np.min(ext_arr, axis=1) if nonneg_ext else np.zeros(5, 'f4')
    out = dict(
        profile=_device(optics['table']), heights=_device(c['h']), angles=_device(c['chi']),
        quad=_device(np.stack([c['u'], c['wu']], axis=1)),
        physics=_device([domain.top_km, 1 / domain.step_km, 1 / c['height'], 1 / air0, c['mix'], c['g']]),
        layer=_device(c.get('aer2', np.zeros(7))), shnorm=_device(np.r_[aa, bb, dd]),
        densities=_device(np.stack([c['hr'], c['ha']], axis=1)),
        live_map=_device(live_map, 'i4'),
        live_ir=_device(live_ir if len(live_ir) else np.zeros(1, 'i4'), 'i4'),
        nlive=int(len(live_ir)),
        ext=_device(ext_arr),
        ext_min=_device(ext_min),
        nonneg_ext=nonneg_ext,
        templates=dict(NH=len(c['h']), NC=len(c['chi']), NQ=len(c['u']), L=c['L'], W=c['W'],
                       MASK=int(mask), LAYER='aer2' in c, NT=len(optics['table'])))
    if first:
        out['sun_h'] = _device(c['sun_h'])
        out['sun_col'] = _device(c['sun_col'])
        out['asca'] = _device(c['asca'])
        out['templates']['NSH'] = len(c['sun_h'])
        out['templates']['NSX'] = int(c['sun_col'].shape[1])
    return out


def _project(offsets, basis, azimuth, weighted, nc, L, W, premul=False):
    K = count(L)
    nwq = (W + 3) // 4
    tg = min(256, ((4 + K + 31) // 32) * 32)
    moment, = _launch('project', [offsets, basis, azimuth, weighted], dict(W=W, K=K, L=L, PREMUL=premul, TG=tg),
                      (nwq * tg, nc, 1), (tg, 1, 1),
                      [((nc, 4 + K, W), mx.float32)])
    return np.asarray(moment).astype('f8')


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


def _pack_source(raw_source, L, W):
    K = count(L)
    src_rayleigh = mx.flatten(raw_source[:, :4, :]).reshape(raw_source.shape[0], 4, W)
    if L <= 16 and W <= 16:
        ml_idx = mx.array([l * (l + 1) // 2 + m for m in range(L + 1) for l in range(m, L + 1)], dtype=mx.int32)
        m_scale = mx.array([1.0 if m == 0 else np.sqrt(2.0) for m in range(L + 1) for _ in range(m, L + 1)],
                           dtype=mx.float32)
        sh = mx.take(raw_source[:, 10:, :], ml_idx, axis=1) * m_scale[None, :, None]
        if W < 16:
            sh = mx.pad(sh, [(0, 0), (0, 0), (0, 16 - W)])
        src_sh = mx.flatten(sh).reshape(raw_source.shape[0], K, 16)
    else:
        src_sh = mx.flatten(raw_source[:, 10:, :]).reshape(raw_source.shape[0], K, W)
    return src_rayleigh, src_sh, raw_source


@lru_cache(maxsize=2)
def _angular_grid(L, intervals):
    from .arithmetic import basis
    theta = np.arange(intervals + 1) * (np.pi / intervals)
    y = basis(np.cos(theta), L)
    m = np.asarray([m for l in range(L + 1) for m in range(l + 1)])
    az = np.cos(theta[:, None] * m).astype('f4')
    return _device((y[:, None, :] * az[None, :, :]).reshape(-1, len(m)))


def _angular_field(raw, c, intervals):
    n, _, w = raw.shape
    sh = raw[:, 10:, :].transpose(1, 0, 2).reshape(-1, n*w)
    return mx.matmul(_angular_grid(c['L'], intervals), sh)


def _source_bounds(raw, c):
    # By the addition theorem, ||Y||_2 <= (L+1)/sqrt(4*pi). A max-norm
    # bound avoids squaring subnormal or very large SH coefficients.
    K, nc = count(c['L']), len(c['chi'])
    coeff = mx.max(mx.abs(raw[:, 10:, :]), axis=1)
    factor = np.float32(1.001 * (c['L'] + 1) * np.sqrt(K / (4 * np.pi)))
    bound = coeff * factor
    hr = _device(c['hr']).reshape(-1, 1, 1)
    ha = mx.abs(_device(c['ha'])).reshape(-1, 1, 1)
    shape = (len(c['h']), nc, c['W'])
    bound = bound.reshape(shape) * ha
    ray_scale = mx.max(mx.abs(raw[:, :4, :]), axis=1).reshape(shape) * mx.abs(hr)
    live = np.asarray(mx.any(bound > (2.0**-23) * ray_scale, axis=(1, 2)))
    return mx.flatten(bound), live


def _tile(packed, lo, hi, c, source, boundary, dc, bd, capacity=None):
    co, ro, host, intervals, frames = packed.tile(lo, hi)
    if boundary is not None:
        end_data, end_indices = _boundary(packed.cells[lo:hi], packed.order[lo:hi], c)
    else:
        end_data, end_indices = bd, _device([0], 'i4')
    nr, ns, nc = len(host), len(intervals) * len(c['u']), hi - lo
    max_rays, max_nodes, max_segments = capacity or (nr, ns, len(intervals))
    W, L = c['W'], c['L']
    K = count(L)
    if not nr:
        return np.zeros((nc, 4 + K, W), 'f8')
    if isinstance(source, tuple):
        src_rayleigh, src_sh, raw = source
    else:
        src_rayleigh, src_sh, raw = _pack_source(source, L, W)
    d_co, d_ro = _device(co, 'i4'), _device(ro, 'i4')
    d_host = _device(host)
    polar = (128 * L / np.pi) * np.arccos(np.clip(packed.mu[packed.cell_off[lo]:packed.cell_off[hi]], -1., 1.))
    knot = np.rint(polar).astype('i4')
    # The interpolation plan contains a table index and a local coordinate.
    d_polar = _device(np.column_stack((knot, polar - knot)))
    template = dc['templates']
    panel_lanes = min(32, 1 << template['NQ'].bit_length())
    use_f4 = L <= 16 and W <= 16
    wp = 16 if use_f4 else ((W + 3) // 4) * 4
    cos_shape = (1,) if use_f4 else (max_rays, L + 1)
    d_frames = _device(frames)
    y, cosine, d_ray = _launch('basis', [d_host, dc['shnorm'], d_co, d_frames, end_data],
                             dict(L=L, BD=boundary is not None, PREMUL=use_f4, NEED_EXP=True),
                             (L + 1, nr, 1), (L + 1, 4, 1),
                             [((max_rays, K), mx.float32), (cos_shape, mx.float32), ((max_rays, 28), mx.float32)])
    d_intervals = _device(intervals) if ns else bd
    if ns:
        point, c0cp, segment_sum, owners = _launch(
            'panels', [d_ray, d_ro, d_intervals, dc['quad'], dc['heights'], dc['angles'],
                       dc['profile'], dc['physics'], dc['layer']], dict(template, P=panel_lanes),
            (len(intervals) * panel_lanes, 1, 1), (128, 1, 1),
            [((max_nodes, 12), mx.float32), ((max_nodes, 2), mx.int32),
             ((max_segments, 5), mx.float32), ((max_segments,), mx.int32)])
        nlcell = dc['nlive'] * template['NC']
        if dc.get('angular') is not None:
            aero, inv_slot = dc['angular'], end_indices
        elif nlcell:
            seg_depth, = _launch(
                'seg_prefix', [d_ro, segment_sum, dc['ext_min']], {},
                (nr, 1, 1), (min(128, nr), 1, 1), [((max_segments,), mx.float32)])
            nb = 256
            seg_n = (ns + nb - 1) // nb
            meta = _device([ns, seg_n], 'i4')
            counts, = _launch(
                'vp_count', [c0cp, dc['live_map'], seg_depth, meta],
                dict(NLCELL=nlcell, NC=template['NC'], NB=nb, NQ=template['NQ']),
                (nb * 256, 1, 1), (256, 1, 1), [((nlcell, nb), mx.int32)])
            flat = counts.reshape(-1)
            inc = mx.cumsum(flat)
            offs = inc - flat
            sorted_rcp, inv_slot = _launch(
                'vp_scatter', [c0cp, owners, dc['live_map'], seg_depth, offs, meta],
                dict(NLCELL=nlcell, NC=template['NC'], NB=nb, NQ=template['NQ']),
                (nb * 32, 1, 1), (32, 1, 1), [((4 * max_nodes, 2), mx.uint32), ((4 * max_nodes,), mx.int32)])
            blk_recs = 1024
            blk_cap = nlcell + (4 * max_nodes + blk_recs - 1) // blk_recs
            bstart, blk = _launch(
                'vp_blocks', [offs, inc, dc['live_ir']],
                dict(NLCELL=nlcell, NC=template['NC'], NB=nb, BLK=blk_recs),
                (256, 1, 1), (256, 1, 1), [((nlcell + 1,), mx.int32), ((blk_cap, 3), mx.int32)])
            if use_f4:
                ntg = 256
                aero, = _launch(
                    'vp_vprod_f4_w64', [bstart, blk, sorted_rcp, d_host, src_sh, dc['densities']],
                    dict(L=L, W=W, NC=template['NC'], NLCELL=nlcell, NTG=ntg),
                    (blk_cap * ntg, 1, 1), (ntg, 1, 1), [((4 * max_nodes, wp), mx.float32)])
            else:
                wchunk = 16 if K * 16 * 4 <= 28672 else 8
                aero, = _launch(
                    'vp_vprod', [bstart, blk, sorted_rcp, y, src_sh, dc['densities']],
                    dict(L=L, W=W, WP=wp, NC=template['NC'], NLCELL=nlcell, WCHUNK=wchunk),
                    (blk_cap * 128, 1, 1), (128, 1, 1), [((4 * max_nodes, wp), mx.float32)])
        else:
            aero, inv_slot = bd, end_indices
    else:
        # The transport loop is empty, but a ray may still see its boundary.
        point, c0cp, segment_sum, aero, inv_slot = bd, end_indices, bd, bd, end_indices
    wave_lanes = 1 << (W - 1).bit_length()
    inputs = [d_ray, d_ro, point, c0cp, segment_sum, src_rayleigh, aero, inv_slot,
              dc['densities'], dc['live_map'], dc['ext'], dc['physics'], bd, end_indices,
              d_frames, d_co, dc['angles'], raw, d_intervals, dc['quad'], d_polar, _basis_table(L)]
    params = dict(W=W, WP=wp, NC=template['NC'], NQ=template['NQ'], L=L,
                  BD=boundary is not None, NONNEG_EXT=dc['nonneg_ext'], BOUNDS='bounds' in dc, ANG=dc.get('ang_bins', 0), NS=len(c['h'])*len(c['chi']))
    grid = (wave_lanes, nr, 1)
    group = (min(128, wave_lanes), 8, 1)
    shape = (5, max_rays, W)
    weighted, flags = _launch('integrate', inputs + [bd, end_indices], dict(params, REFINE=False),
                              grid, group, [(shape, mx.float32), ((nr, W), mx.int32)])
    weighted, _ = _launch('integrate', inputs + [weighted, flags], dict(params, REFINE=True),
                           grid, group, [(shape, mx.float32), ((1,), mx.int32)])
    return _project(d_co, y, cosine, weighted, nc, L, W, premul=use_f4)


def _tile_first(packed, lo, hi, c, boundary, dc, bd, capacity=None):
    co, ro, host, intervals, frames = packed.tile(lo, hi)
    begin, end = packed.cell_off[lo], packed.cell_off[hi]
    shadow = np.empty((end - begin, 2), 'i4')
    for i in range(lo, hi):
        g = packed.cells[i]
        index = packed.order[i]
        if len(index):
            a, b = packed.cell_off[i:i + 2] - begin
            shadow[a:b] = _shadow_nodes(g['p'][0], g['d'][index], g['edges'][index],
                                         g['lower'][index], c['u'], np.diff(ro[a:b + 1]))
    if boundary is not None:
        end_data, end_indices = _boundary(packed.cells[lo:hi], packed.order[lo:hi], c)
    else:
        end_data, end_indices = bd, _device([0], 'i4')
    nr, nc = len(host), hi - lo
    max_rays = capacity[0] if capacity else nr
    W, L = c['W'], c['L']
    K = count(L)
    if not nr:
        return np.zeros((nc, 4 + K, W), 'f8')
    d_co, d_ro = _device(co, 'i4'), _device(ro, 'i4')
    d_host, d_frames = _device(host), _device(frames)
    d_intervals = _device(intervals) if len(intervals) else bd
    y, cosine, _ = _launch('basis', [d_host, dc['shnorm'], d_co, d_frames, end_data],
                           dict(L=L, BD=boundary is not None, PREMUL=True, NEED_EXP=False),
                           (L + 1, nr, 1), (L + 1, 4, 1),
                           [((max_rays, K), mx.float32), ((1,), mx.float32), ((1,), mx.float32)])
    template = dc['templates']
    wg = (W + 15) // 16
    grid_x = ((nr + 1) // 2) * 32
    weighted, = _launch(
        'first',
        [d_host, d_frames, d_co, d_ro, d_intervals, _device(shadow, 'i4'), dc['quad'], dc['sun_h'], dc['sun_col'],
         dc['profile'], dc['physics'], dc['layer'], dc['ext'], dc['asca'], bd, end_data, end_indices],
        dict(W=W, WG=wg, NQ=template['NQ'], NSH=template['NSH'], NSX=template['NSX'],
             MASK=template['MASK'], LAYER=template['LAYER'], NT=template['NT'], BD=boundary is not None),
        (grid_x, 1, 1), (min(128, grid_x), 1, 1), [((5, max_rays, W), mx.float32)])
    return _project(d_co, y, cosine, weighted, nc, L, W, premul=True)


def evaluate(cells, constants, field, boundary, packed=None, *, first=False):
    """Each completed tile returns its reduced FP32 moments in the public dtype."""
    c = constants[0]
    expected = (len(c['h'])*len(c['chi']), 10 + count(c['L']), c['W'])
    if not first and field.shape != expected:
        raise ValueError(f"expected source shape {expected}, received {field.shape}")
    if boundary is not None and np.shape(boundary) != (len(c['chi']), c['W']):
        raise ValueError("the boundary must have one spectrum per solar angle")
    if first:
        source = None
    else:
        raw_source = _device(field)
        source = _pack_source(raw_source, c['L'], c['W'])
        mx.eval(*source)
        del raw_source
    boundary = boundary if boundary is not None and np.any(boundary) else None
    bd = _device([0.] if boundary is None else boundary)
    W, K = c['W'], count(c['L'])
    moments = np.empty((len(cells), 4 + K, W), 'f8')
    if packed is None:
        packed = CompactRays(cells, include_empty=boundary is not None)
    # Fixed physical capacities let the allocator reuse the preceding tile's
    # buffers. Every kernel dispatch and reduction still uses the actual counts.
    tiles = list(packed.tiles(mlx_backend.tile_samples()))
    max_rays = max((int(packed.cell_off[b] - packed.cell_off[a]) for a, b in tiles), default=0)
    max_nodes = max((sum(packed.samples[a:b]) for a, b in tiles), default=0)
    max_segments = max((int(packed.seg_off[packed.cell_off[b]] - packed.seg_off[packed.cell_off[a]]) for a, b in tiles), default=0)
    capacity = (max_rays, max_nodes, max_segments)
    wp = 16 if (c['L'] <= 16 and W <= 16) else ((W + 3) // 4) * 4
    ray_bytes = 4 * (9 + 28 + K + c['L'] + 1 + 5*W + 1)
    tile_node_bytes = 0 if first else (104 + 16*wp) * max_nodes + 32 * max_segments
    pool_limit = int(1.5 * (tile_node_bytes + ray_bytes*max_rays)) + moments[0].nbytes*max((b-a for a,b in tiles), default=1)
    devices = {}
    ang_bins = 6 * c['L']
    angular_bytes = (ang_bins + 1)**2 * field.shape[0] * W * 4 if not first else 0
    # Build the angle table only when the ray workload can amortize its construction.
    use_angular = (not first and c['L'] <= 16 and W <= 16 and angular_bytes < 3 * 2**30
                   and sum(packed.samples) >= field.shape[0] * (ang_bins + 1)**2)
    for lo, hi in tiles:
        c = constants[lo]
        if id(c) not in devices:
            if first:
                devices[id(c)] = _constants(c, first=True)
            else:
                bounds, live = _source_bounds(source[2], c)
                devices[id(c)] = dict(_constants(c, scheduling=live), bounds=bounds)
                if use_angular:
                    table = _angular_field(source[2], c, ang_bins)
                    mx.eval(table)
                    devices[id(c)].update(angular=table, ang_bins=ang_bins)
        if first:
            moments[lo:hi] = _tile_first(packed, lo, hi, c, boundary, devices[id(c)], bd, capacity)
        else:
            moments[lo:hi] = _tile(packed, lo, hi, c, source, boundary, devices[id(c)], bd, capacity)
        # Finish deferred Metal buffer releases before starting the next tile.
        mx.synchronize()
        # All tile arrays have been released. Bound cached free blocks to one
        # tile between dispatches and return the pool at the end of the call.
        if hi == len(cells) or mx.get_cache_memory() > pool_limit:
            mx.clear_cache()
    return moments, np.zeros((len(cells), W), 'f8')
