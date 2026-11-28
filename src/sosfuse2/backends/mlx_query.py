"""FP32 transport of unweighted observer rays on Metal."""

from collections import OrderedDict
from functools import lru_cache
import hashlib

import mlx.core as mx
import numpy as np

from . import mlx_backend, numpy_reference as ref
from .arithmetic import basis
from .mlx_ops import _domain_key
from .mlx_warm import _constants, _device, _flushes_subnormals, _segment_heights, _shadow_interval
from .mlx_warm_kernels import HEADER
from .sh_layout import count


_NODES_CACHE = OrderedDict()
_NODES_CACHE_BYTES = 0
_NODES_CACHE_LIMIT = 16 << 20
_STENCIL_CACHE = OrderedDict()
_STENCIL_CACHE_MAX = 64
_QUERY_POOL_LIMIT = 32 << 20
_PLAN_TAG = 0


def _nodes_put(cache_key, value, nbytes):
    global _NODES_CACHE_BYTES
    if nbytes > _NODES_CACHE_LIMIT:
        return
    old = _NODES_CACHE.pop(cache_key, None)
    if old is not None:
        _NODES_CACHE_BYTES -= old[0]
    while _NODES_CACHE and _NODES_CACHE_BYTES + nbytes > _NODES_CACHE_LIMIT:
        _, evicted = _NODES_CACHE.popitem(last=False)
        _NODES_CACHE_BYTES -= evicted[0]
    _NODES_CACHE[cache_key] = (nbytes, value)
    _NODES_CACHE_BYTES += nbytes


def _nodes_get(cache_key):
    entry = _NODES_CACHE.get(cache_key)
    if entry is None:
        return None
    _NODES_CACHE.move_to_end(cache_key)
    return entry[1]


@lru_cache(maxsize=256)
def _plan(order, domain_key, nodes, p_bytes, d_bytes, dl_bytes, frame_bytes, edge_bytes, lower_bytes):
    """Pack geometry by value so regenerated observer rays share a host plan."""

    def doubles(buf, cols):
        return np.frombuffer(buf, np.float64).reshape(-1, cols)

    p, d, dl = (doubles(buf, 3) for buf in (p_bytes, d_bytes, dl_bytes))
    frames = doubles(frame_bytes, 9).reshape(-1, 3, 3).transpose(0, 2, 1)
    edges = np.frombuffer(edge_bytes, np.float64).reshape(len(d), -1)
    lower = np.frombuffer(lower_bytes, bool).reshape(len(d), -1)
    widths = np.diff(edges, axis=1)
    rows, cols = np.nonzero(widths > 1e-8)
    offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(d)))].astype('i4')
    asc = lower[rows, cols]
    e0 = edges[rows, cols]
    e1 = edges[rows, cols + 1]
    base = np.where(asc, e0, e1)
    pos = p[rows] + base[:, None] * d[rows]
    r = np.linalg.norm(pos, axis=1)
    rho = np.hypot(pos[:, 0], pos[:, 1])
    width = widths[rows, cols]
    lo, hi = _shadow_interval(p.T, d)
    u = np.frombuffer(nodes, np.float64)

    # Rank each cylinder crossing relative to the nearer quadrature endpoint.
    def rank(t, side):
        delta = (t[rows] - base) / width
        return np.where(asc, np.searchsorted(u * u, delta, side), np.searchsorted(-((1 - u) ** 2), delta, side))

    h0 = r - 6371.0
    if domain_key[0] > domain_key[2]:
        h0, *extra = _segment_heights(p[rows], d[rows], e0, e1, h0, domain_key)
    else:
        extra = (np.zeros(len(rows)),)

    span = np.where(asc, width, -width)
    segments = np.column_stack(
        (
            pos,
            h0,
            np.sum(pos * d[rows], axis=1),
            span,
            (rho - 6371.0) * (rho + 6371.0),
            np.sum(pos[:, :2] * d[rows, :2], axis=1),
            rank(lo, 'right'),
            rank(hi, 'left'),
            *extra,
        )
    ).astype('f4')
    rays = np.column_stack((d, dl, frames.reshape(-1, 9), np.hypot(d[:, 0], d[:, 1]))).astype('f4')
    global _PLAN_TAG
    _PLAN_TAG += 1
    return rays, basis(d[:, 2], order), offsets, segments, _PLAN_TAG


def _windows(offsets, nq, limit):
    """Partition rays by sample count. Carry one ray across windows if needed."""
    r = 0
    while r + 1 < len(offsets):
        stop = int(np.searchsorted(offsets, int(offsets[r]) + limit // nq, side='right')) - 1
        if stop > r:
            yield r, stop, 0, (int(offsets[stop]) - int(offsets[r])) * nq
            r = stop
        else:
            n = (int(offsets[r + 1]) - int(offsets[r])) * nq
            for k in range(0, n, limit):
                yield r, r + 1, k, min(n, k + limit)
            r += 1


@lru_cache(maxsize=3)
def _kernel(name):
    from .mlx_query_kernel import HELPERS, MODES, MULTIPLE, PREPARE

    if name == 'prepare':
        body = PREPARE
        inputs = 'rays offsets segments quad hg cg tab phys layer dens ext sun_h sun_col asca window prior gm g_idx g_f0 g_f1 st_phase'
        outputs = ['out', 'factors']
    elif name == 'modes':
        body = MODES
        inputs = 'basis field pairs st_comb'
        outputs = ['out']
    else:
        body = MULTIPLE
        inputs = 'rays offsets factors modes slots field window prior smodes st_comb'
        outputs = ['out']
    return mx.fast.metal_kernel(
        name='sos_query_' + name,
        input_names=inputs.split(),
        output_names=outputs,
        header=HEADER.replace('NATIVE_FTZ', str(_flushes_subnormals()).lower()) + HELPERS,
        source=body,
    )


def _f4_bytes(x):
    if isinstance(x, np.ndarray) and x.dtype == np.float32 and x.flags.c_contiguous:
        return x.tobytes()
    return np.ascontiguousarray(x, 'f4').tobytes()


def _f8_bytes(x):
    if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.flags.c_contiguous:
        return x.tobytes()
    return np.ascontiguousarray(x, 'f8').tobytes()


def _optics_key(c):
    """Value key of every optical input used to prepare multiple-scattering nodes."""
    h = hashlib.blake2b(digest_size=16)
    optics = c.get('density_optics')
    strat = 'strato_ext' in c
    combine = bool(strat and c.get('sh_combine', False))
    skip_thin = bool(combine and c.get('sh_combine_skip', False) and len(c['h']) <= 128)
    dom = c['domain']
    scalars = (
        dom.top_km,
        dom.step_km,
        dom.lower_end,
        dom.air_end,
        c['L'],
        float(c['height']),
        float(c['mix']),
        float(c['g']),
        float(c['albedo']),
        c['W'],
        len(c['chi']),
        strat,
        combine,
        skip_thin,
        None if optics is None else (optics['mask'], optics.get('air_surface'), optics['table'].shape),
    )
    h.update(repr(scalars).encode())
    if optics is not None:
        h.update(np.ascontiguousarray(optics['table']).tobytes())
    for k in ('h', 'chi', 'wu', 'hr', 'ha', 'ext'):
        h.update(_f4_bytes(c[k]))
    if 'aer2' in c:
        h.update(b'A2')
        h.update(c['aer2'].tobytes())
    if strat:
        sp = c['strato_profile']
        h.update(repr((float(sp['center']), float(sp['sigma']), float(sp['norm']), sp.get('end_km'))).encode())
        h.update(_f4_bytes(c['strato_ext']))
        h.update(_f4_bytes(c['strato_density']))
    return h.digest()


def _stencils(factors, offsets, nq, nc, nh):
    """Index the distinct (ray, source cell) pairs used by the four node corners."""
    nodes = np.asarray(factors)
    # A partial single-ray tile can begin in the middle of an interval.
    ray = 0 if len(offsets) == 2 else np.repeat(np.arange(len(offsets) - 1), np.diff(offsets) * nq)
    base = nodes[:, 20].view('i4') & np.int32(0x1FFFFFFF)
    corners = np.array([0, 1, nc, nc + 1], 'i4')
    keys = (ray * nc * nh + base)[:, None] + corners
    unique, slots = np.unique(keys, return_inverse=True)
    pairs = np.column_stack(np.divmod(unique, nc * nh)).astype('i4')
    return pairs, slots.reshape(len(base), 4).astype('i4')


WAVE_LANES = 16
NODE_LANES = 8


def _run(plan, c, devices, source, first, limit, key, result, pending):
    rays, y, offsets, segments, plan_id = plan
    W, nq, L = c['W'], len(c['u']), c['L']
    strat = 'strato_ext' in c
    combine = bool(strat and c.get('sh_combine', False))
    skip_thin = bool(combine and c.get('sh_combine_skip', False) and len(c['h']) <= 128)
    strat_sep = bool(strat and not combine)
    extended = bool(ref.support_extended(c['domain']))
    record = 28 + L + W
    # Up to four distinct Fourier coefficient vectors per quadrature point.
    limit = min(limit, max(1, (32 << 20) // (4 * (record + 4 * (L + 1) * W * (2 if strat_sep else 1)))))
    unused = mx.array([0.0])
    unused_i = mx.array([0], dtype=mx.int32)
    carry = prefix = unused
    for a, b, begin, end in _windows(offsets, nq, limit):
        n = b - a
        seg0, seg1 = offsets[[a, b]]
        if seg0 == seg1:
            result[a:b] = 0
            continue
        saved = end < (int(seg1) - int(seg0)) * nq
        common = {'CONT': bool(begin), 'SAVE': saved}
        # Cache compact device inputs only for small complete packs (strictly bounded).
        cacheable = not first and a == begin == 0 and b == len(rays) and not saved and end * record * 4 <= 2**20
        cache_key = (plan_id, key) if cacheable else None
        entry = _nodes_get(cache_key) if cacheable else None
        if entry is not None:
            factors, d_y, ray, ro, window, d_pairs, d_slots, npairs, st_comb = entry
        else:
            ro = _device(offsets[a : b + 1] - seg0, 'i4')
            ray = _device(rays[a:b])
            window = _device([begin, end], 'i4')
            if id(c) not in devices:
                devices[id(c)] = _constants(c, first=first)
            dc = devices[id(c)]
            t = dc['templates']
            zero = dc['physics']
            st_comb = dc.get('st_comb', zero)
            has_ground = bool(first and dc.get('has_ground', False))
            use_hg_table = bool(has_ground and dc.get('hg_table', False))
            inputs = [
                ray,
                ro,
                _device(segments[seg0:seg1]),
                dc['quad'],
                dc['heights'],
                dc['angles'],
                dc['profile'],
                dc['physics'],
                dc['layer'],
                dc['densities'],
                dc['ext'],
                dc.get('sun_h', zero),
                dc.get('sun_col', zero),
                dc.get('asca', zero),
                window,
                prefix,
                dc.get('gm', zero),
                dc.get('g_idx', unused_i),
                dc.get('g_f0', zero),
                dc.get('g_f1', zero),
                dc.get('st_phase', zero),
            ]
            params = dict(
                t,
                NSH=t.get('NSH', 2),
                NSX=t.get('NSX', 2),
                FIRST=bool(first),
                GROUND=has_ground,
                HG_TABLE=use_hg_table,
                **common,
            )
            shape = (n, W, 11 if saved else 4) if first or saved else (1,)
            prefix, factors = _kernel('prepare')(
                inputs=inputs,
                template=list(params.items()),
                grid=(W, n, 1),
                threadgroup=(min(32, W), 4, 1),
                output_shapes=[shape, (1,) if first else (end - begin, record)],
                output_dtypes=[mx.float32] * 2,
            )
            if not first:
                d_y = _device(y[a:b])
                dom = c['domain']
                s_key = (
                    plan_id,
                    dom.top_km,
                    dom.step_km,
                    dom.lower_end,
                    dom.air_end,
                    _f4_bytes(c['h']),
                    _f4_bytes(c['chi']),
                ) if cacheable else None
                st = _STENCIL_CACHE.get(s_key) if s_key is not None else None
                if st is not None:
                    _STENCIL_CACHE.move_to_end(s_key)
                    d_pairs, d_slots, npairs = st
                else:
                    pairs, slots = _stencils(factors, offsets[a : b + 1], nq, len(c['chi']), len(c['h']))
                    d_pairs = _device(pairs, 'i4')
                    d_slots = _device(slots, 'i4')
                    npairs = len(pairs)
                    if s_key is not None:
                        _STENCIL_CACHE[s_key] = (d_pairs, d_slots, npairs)
                        while len(_STENCIL_CACHE) > _STENCIL_CACHE_MAX:
                            _STENCIL_CACHE.popitem(last=False)
                if cacheable:
                    nbytes = (
                        factors.nbytes + d_y.nbytes + ray.nbytes + ro.nbytes + window.nbytes
                        + d_pairs.nbytes + d_slots.nbytes + st_comb.nbytes
                    )
                    _nodes_put(cache_key, (factors, d_y, ray, ro, window, d_pairs, d_slots, npairs, st_comb), nbytes)
            del inputs
        if first:
            value = prefix
            tile_bytes = n * W * 44
        else:
            mode_t = dict(
                L=L, W=W, NC=len(c['chi']),
                STRATO=strat, COMBINE=combine, SKIP_THIN=skip_thin, PLANE=0,
            )
            mt_w = min(32, W)
            mt_m = min(max(1, 128 // mt_w), L + 1)
            (modes,) = _kernel('modes')(
                inputs=[d_y, source, d_pairs, st_comb],
                template=list(mode_t.items()),
                grid=(W, L + 1, npairs),
                threadgroup=(mt_w, mt_m, 1),
                output_shapes=[(npairs, L + 1, W)],
                output_dtypes=[mx.float32],
            )
            if strat_sep:
                (smodes,) = _kernel('modes')(
                    inputs=[d_y, source, d_pairs, st_comb],
                    template=list(dict(mode_t, PLANE=1).items()),
                    grid=(W, L + 1, npairs),
                    threadgroup=(mt_w, mt_m, 1),
                    output_shapes=[(npairs, L + 1, W)],
                    output_dtypes=[mx.float32],
                )
            else:
                smodes = unused
            (value,) = _kernel('multiple')(
                inputs=[ray, ro, factors, modes, d_slots, source, window, carry, smodes, st_comb],
                template=list(
                    dict(
                        L=L, W=W, NC=len(c['chi']), NQ=nq, WN=WAVE_LANES, PN=NODE_LANES,
                        STRATO=strat, STRATO_SEP=strat_sep, EXTENDED=extended, **common,
                    ).items()
                ),
                grid=(WAVE_LANES * ((W + WAVE_LANES - 1) // WAVE_LANES), NODE_LANES, n),
                threadgroup=(WAVE_LANES, NODE_LANES, 1),
                output_shapes=[(n, W, 8 if saved else 4)],
                output_dtypes=[mx.float32],
            )
            tile_bytes = (end - begin) * record * 4 + npairs * (L + 1) * W * 4 * (2 if strat_sep else 1)
        if saved:
            mx.eval(prefix, value)
            carry = value
        else:
            pending.append((result[a:b], value))
            carry = prefix = unused
        del ray, ro, factors, value
        if not first:
            del d_y, modes, smodes, d_pairs, d_slots
        if saved or tile_bytes > (_QUERY_POOL_LIMIT // 2):
            pool_limit = max(_QUERY_POOL_LIMIT, 2 * tile_bytes)
            if mx.get_cache_memory() > pool_limit:
                if pending:
                    mx.eval(*(v for _, v in pending))
                    for dst, v in pending:
                        dst[...] = np.asarray(v)
                    pending.clear()
                mx.synchronize()
                mx.clear_cache()


def query(packs, constants, field, first):
    limit = mlx_backend.tile_samples()
    devices, keys = {}, {}
    if not first:
        for c in constants:
            if id(c) not in keys:
                strat = 'strato_ext' in c
                K = count(c['L'])
                expected = (len(c['h']) * len(c['chi']), 10 + K * (2 if strat else 1), c['W'])
                if np.shape(field) != expected:
                    raise ValueError(f"expected source shape {expected}, received {np.shape(field)}")
                keys[id(c)] = None
        if not np.any(field):
            return [np.zeros((len(g['d']), c['W'], 4), 'f8') for g, c in zip(packs, constants, strict=True)]
        keys.clear()
    source = mx.array([0.0]) if first else _device(field)
    results = []
    pending = []
    for g, c in zip(packs, constants, strict=True):
        if id(c) not in keys:
            keys[id(c)] = None if first else _optics_key(c)
        nr = len(g['d'])
        output = np.empty((nr, c['W'], 4), 'f8')
        u_bytes = _f8_bytes(c['u'])
        domain_key = _domain_key(c['domain'])
        # Bound the size of a cached basis even for callers with large packs.
        for start in range(0, nr, 256):
            if start == 0 and nr <= 256:
                arrays = [_f8_bytes(g[k]) for k in ('p', 'd', 'dloc', 'target', 'edges')]
                lo_bytes = g['lower'].tobytes() if (g['lower'].dtype == np.bool_ and g['lower'].flags.c_contiguous) else np.ascontiguousarray(g['lower'], bool).tobytes()
                dst = output
            else:
                sl = slice(start, start + 256)
                arrays = [_f8_bytes(g[k][sl]) for k in ('p', 'd', 'dloc', 'target', 'edges')]
                lo_bytes = np.ascontiguousarray(g['lower'][sl], bool).tobytes()
                dst = output[sl]
            plan = _plan(c['L'], domain_key, u_bytes, *arrays, lo_bytes)
            _run(plan, c, devices, source, first, limit, keys[id(c)], dst, pending)
        results.append(output)
    if pending:
        mx.eval(*(v for _, v in pending))
        for dst, v in pending:
            dst[...] = np.asarray(v)
        pending.clear()
    del source, devices
    if mx.get_cache_memory() > _QUERY_POOL_LIMIT:
        mx.synchronize()
        mx.clear_cache()
    return results
