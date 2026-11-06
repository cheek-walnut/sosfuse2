"""Operation-matched FP32 sample transport; FP64 only for final ray projection."""

import contextlib
import ctypes
import ctypes.util
import math
import platform
import numpy as np
from .components import NCOMP
from .sh_layout import count, combined_coefficients
from .arithmetic import F, R, split, add, sub, neg, mul, root, divide, quotient, rounded, basis, cosine, warp_sum, kahan

PAIRS = ((0, 0), (1, 1), (2, 2), (0, 2))
# FLT_MIN, same as kernel.cu TRACE_MIN: subnormal (or NaN) traces are excluded from the log-space moments.
TRACE_MIN = np.float32(np.finfo(np.float32).tiny)


# FP32 flush-to-zero of the device operations (2L): every CUDA build
# runs with -ftz=true (CuPy RawKernel adds it, fused.py raw_kernel), so subnormal FP32 inputs and results are 0 on the device. The
# NumPy mirror computes transport() and project() in the same CPU mode (C99 fenv: AArch64 FPCR.FZ, x86-64 MXCSR FTZ+DAZ). Without
# it, a corner trace just above FLT_MIN whose terms are subnormal takes the log path here and the linear fallback on CUDA
# (C3+AEROSOL_2L 375.7 nm: +92 % in deep-shadow cells). Unsupported platforms stop (E_FTZ_UNSUPPORTED), never fall back to IEEE.
class _FenvDarwinArm64(ctypes.Structure):
    _fields_ = [('fpsr', ctypes.c_uint64), ('fpcr', ctypes.c_uint64)]


class _FenvLinuxArm64(ctypes.Structure):
    _fields_ = [('fpcr', ctypes.c_uint32), ('fpsr', ctypes.c_uint32)]


class _FenvLinuxX86_64(ctypes.Structure):
    _fields_ = [('x87', ctypes.c_uint8 * 28), ('mxcsr', ctypes.c_uint32)]


def _ftz_platform():
    system, machine = platform.system(), platform.machine().lower()
    if system == 'Darwin' and machine in ('arm64', 'aarch64'):
        T, bits = _FenvDarwinArm64, ('fpcr', 1 << 24)
    elif system == 'Linux' and machine in ('arm64', 'aarch64'):
        T, bits = _FenvLinuxArm64, ('fpcr', 1 << 24)
    elif system == 'Linux' and machine in ('x86_64', 'amd64'):
        T, bits = _FenvLinuxX86_64, ('mxcsr', 0x8040)
    else:
        raise RuntimeError(f'E_FTZ_UNSUPPORTED: no FP32 flush-to-zero mode for {system} {machine}')
    name = ctypes.util.find_library('m') or ctypes.util.find_library('c')
    lib = ctypes.CDLL(name)
    return lib, T, bits


_FTZ = None


@contextlib.contextmanager
def flush_to_zero():
    """CPU FP32 flush-to-zero (subnormal inputs and results are 0) for the block, as the -ftz=true device code computes it."""
    global _FTZ
    if _FTZ is None:
        _FTZ = _ftz_platform()
    lib, T, (field, bit) = _FTZ
    old, new = T(), T()
    if lib.fegetenv(ctypes.byref(old)) != 0:
        raise RuntimeError('E_FTZ_FENV: fegetenv failed')
    ctypes.memmove(ctypes.byref(new), ctypes.byref(old), ctypes.sizeof(T))
    setattr(new, field, getattr(new, field) | bit)
    if lib.fesetenv(ctypes.byref(new)) != 0:
        raise RuntimeError('E_FTZ_FENV: fesetenv failed')
    try:
        # normal input -> subnormal result (output flush) and subnormal input -> normal result (input flush), scalar and SIMD
        if (
            TRACE_MIN * np.float32(0.5) != 0
            or (np.full(64, TRACE_MIN, 'f4') * np.float32(0.5))[0] != 0
            or (np.full(64, 5.877472e-39, 'f4') * np.float32(4))[0] != 0
        ):
            raise RuntimeError('E_FTZ_INACTIVE: subnormal FP32 values survived the flush-to-zero mode')
        yield
    finally:
        lib.fesetenv(ctypes.byref(old))


def ftz_value(a):
    """FTZ-B (gate v2, kernel.cu ftz_dv): a solar-boundary DS value as an FTZ FP32 build computes it. Components and their
    sum below FLT_MIN count as 0; NaN passes (abs(NaN) < TRACE_MIN is False)."""
    h, l = (np.where(np.abs(x) < TRACE_MIN, F(0), x) for x in a)
    v = h + l
    return np.where(np.abs(v) < TRACE_MIN, F(0), v)


def trace_accumulate(tr, weight, logs, wg, am):
    """Add one corner to the log-space blend moments; traces below TRACE_MIN (or NaN) are selected out before log and product."""
    good = tr >= TRACE_MIN
    t = np.where(good, tr, F(1))
    w = np.where(good, weight, F(0))
    logs += w * np.log(t)
    wg += w
    am += w * t


def trace_scale(val, logs, wg, am):
    ok = (wg > F(0)) & (am >= TRACE_MIN)
    mean = np.where(ok, am, F(1)) / np.where(ok, wg, F(1))
    scale = np.exp(np.where(ok, logs - wg * np.log(mean), F(0)))
    return val * scale[:, :, None]


def reuse_corners(cache, tags, q, nc):
    """Exact adjacent-stencil aliases, identical fixed-index CUDA copies."""
    old = tags[:, 0].copy()
    di = q['cells'][:, 0] // nc - old // nc
    dj = q['cells'][:, 0] % nc - old % nc
    cases = (
        (0, 1, ((0, 1), (2, 3))),
        (0, -1, ((1, 0), (3, 2))),
        (1, 0, ((0, 2), (1, 3))),
        (-1, 0, ((2, 0), (3, 1))),
        (1, 1, ((0, 3),)),
        (1, -1, ((1, 2),)),
        (-1, 1, ((2, 1),)),
        (-1, -1, ((3, 0),)),
    )
    copied = 0
    for ri, cj, copies in cases:
        mask = (di == ri) & (dj == cj) & (old >= 0) & q['valid']
        rr = np.flatnonzero(mask)
        for dst, src in copies:
            cache[rr, dst] = cache[rr, src]
            tags[rr, dst] = q['cells'][rr, dst]
            copied += len(rr)
    return copied


def frame_terms(g):
    B = g['target'].astype('f4')
    d = g['dloc'].astype('f4')
    pu = np.stack(
        [F(1) - d[:, 0] * d[:, 0], F(1) - d[:, 1] * d[:, 1], F(1) - d[:, 2] * d[:, 2], -d[:, 0] * d[:, 2]], -1
    )
    u = B[:, 2]
    dot = (d[:, 0] * u[:, 0] + d[:, 1] * u[:, 1]) + d[:, 2] * u[:, 2]
    e = u - d * dot[:, None]
    direct = np.stack([(pu[:, k] - e[:, i] * e[:, j]) / F(2) for k, (i, j) in enumerate(PAIRS)], -1)
    return B, d, pu, direct


def path_point(pp, dd, t):
    """DS position q = p + t d, q*q and |q| at DS path coordinates t (pp, dd = split(p), split(d)); kernel.cu path_q."""
    q = add(pp, mul((t[0][:, None], t[1][:, None]), dd))
    q2 = mul(q, q)
    rr = root(add(add((q2[0][:, 0], q2[1][:, 0]), (q2[0][:, 1], q2[1][:, 1])), (q2[0][:, 2], q2[1][:, 2])))
    return q, q2, rr


def path_height(rr):
    """FP32 height above the surface from the DS radius, clipped to [0, 80] km (kernel.cu path_h)."""
    return np.clip(rounded(sub(rr, (R, F(0)))), F(0), F(80))


def profile_at(h, c):
    """Five profiles at FP32 heights h: air, tropospheric aerosol, O3, NO2, O4; stratosphere is evaluated separately.
    The one profile evaluation of the transport (kernel.cu profile_at): sample() calls it at the quadrature nodes and the Simpson
    optical-depth prefix at segment ends and panel midpoints. Density table (PROFILE) and two-layer aerosol (AEROSOL_2L) options."""
    h = np.asarray(h, 'f4')
    from .density import build, interpolate32, check_consumed

    optics = c.get('density_optics')
    if optics is None:
        optics = build({})
    check_consumed(optics, where='reference.profile_at')
    tab = interpolate32(h, optics['table'])
    prof = np.stack(
        [
            np.exp(-h / F(8)),
            np.exp(-h / c['height']),
            np.maximum(F(0), F(1) - abs(h - F(25)) / F(15)),
            tab[:, 2],
            np.zeros_like(h),
        ],
        -1,
    )
    if 'density_optics' in c:
        if optics['mask'] & 1:
            prof[:, 0] = tab[:, 0]
        if optics['mask'] & 2:
            prof[:, 2] = tab[:, 1]
    if 'aer2' in c:
        from .tropo_profile import profile32  # kernel.cu -DAEROSOL_2L aerosol_two_layer, FP32

        prof[:, 1] = profile32(h, c['aer2'])
    q = prof[:, 0] / F(optics.get('air_surface', optics['table'][0, 0] if optics['mask'] & 1 else 1.0))
    prof[:, 4] = q * q
    return prof


def extinction(prof, c, st_prof=None):
    """Extinction [n, W] from five components and a separate stratosphere; kernel.cu extinction_at."""
    out = np.zeros((len(prof), c['W']), 'f4')
    for k in range(NCOMP):
        out += prof[:, k, None] * c['ext'][k]
    if 'strato_ext' in c:
        out += st_prof[:, None] * c['strato_ext']
    return out


class SimpsonPrefix:
    """Optical depth from the path start to the transport's quadrature nodes: on every panel [t_{k-1}, t_k] of a segment [a, b] (t_{-1} = a, the GL nodes
    t_0..t_{nq-1}, then b) Simpson (t_k - t_{k-1})/6 (beta_{k-1} + 4 beta(mid) + beta_k), mid = (t_{k-1} + t_k)/2.
    Node betas are the sample() profiles; beta(a), beta(b) and the midpoints are evaluated anew (profile_at) for every segment,
    never carried over from the previous segment (kernel.cu skips segments with b - a <= 1e-8). Path coordinates, panel widths
    and midpoints are DS (sample() arithmetic). The node prefix is (tau - tc) + acc; acc enters tau with Kahan at the segment end.
    Replaces the GL-weight prefix mid = (tau - tc) + dt/2 (wrong at the nodes, +156 % at the first node for a constant beta)."""

    def __init__(self, g, c):
        self.g, self.c = g, c
        self.pp, self.dd = split(g['p']), split(g['d'])
        self.tau = np.zeros((len(g['d']), c['W']), 'f4')
        self.tc = np.zeros_like(self.tau)

    def beta(self, t):
        """Extinction [n, W] at DS path coordinates t."""
        h = path_height(path_point(self.pp, self.dd, t)[2])
        return extinction(profile_at(h, self.c), self.c, strato_profile(h, self.c) if 'strato_ext' in self.c else None)

    def piece(self, t1, b1):
        """Simpson on [self.t, t1]; adds it to acc where the segment is valid and moves the panel start to (t1, b1)."""
        t0, b0 = self.t, self.b
        tm = add(t0, t1)
        tm = (tm[0] * F(0.5), tm[1] * F(0.5))  # exact DS halving
        w = rounded(sub(t1, t0)) / F(6)
        self.acc = self.acc + np.where(self.ok[:, None], w[:, None] * ((b0 + b1) + F(4) * self.beta(tm)), F(0))
        self.t, self.b = t1, b1

    def begin(self, seg):
        e = self.g['edges']
        self.ok = e[:, seg + 1] - e[:, seg] > 1e-8
        self.end = split(e[:, seg + 1])
        self.t = split(e[:, seg])
        self.b = self.beta(self.t)
        self.acc = np.zeros_like(self.tau)

    def node(self, q):
        """Optical depth [n, W] at the node of sample() result q."""
        self.piece(q['t_DS'], extinction(q['prof'], self.c, q['st_prof']))
        return (self.tau - self.tc) + self.acc

    def finish(self):
        """End piece to b, then the segment total into tau (Kahan), valid rays only (kernel.cu: skipped segment)."""
        if not self.ok.any():
            return
        self.piece(self.end, self.beta(self.end))
        tau, tc = kahan(self.tau, self.tc, self.acc)
        self.tau = np.where(self.ok[:, None], tau, self.tau)
        self.tc = np.where(self.ok[:, None], tc, self.tc)


def sample(g, segment, step, constants, st=None):
    """Generate a quadrature point from endpoints with double-single FP32."""
    a = split(g['edges'][:, segment])
    b = split(g['edges'][:, segment + 1])
    length = sub(b, a)
    u = split(constants['u'][step])
    wu = split(constants['wu'][step])
    lower = g['lower'][:, segment]
    v1 = sub((F(1), F(0)), u)
    v = (np.where(lower, u[0], v1[0]), np.where(lower, u[1], v1[1]))
    tl = add(a, mul(length, mul(u, u)))
    tr = sub(b, mul(length, mul(v, v)))
    tt = tuple(np.where(lower, x, y) for x, y in zip(tl, tr))
    ds = rounded(mul(mul(length, v), wu)) * F(2)
    valid = g['edges'][:, segment + 1] - g['edges'][:, segment] > 1e-8
    ds = np.where(valid, ds, F(0))
    pp, dd = split(g['p']), split(g['d'])
    q, q2, rr = path_point(pp, dd, tt)
    h = path_height(rr)
    xy = root(add((q2[0][:, 0], q2[1][:, 0]), (q2[0][:, 1], q2[1][:, 1])))
    rho = rounded(xy)
    cx = np.where(rho > F(1e-12), quotient((q[0][:, 0], q[1][:, 0]), (np.maximum(xy[0], F(1e-30)), xy[1])), F(1))
    sx = np.where(rho > F(1e-12), quotient((q[0][:, 1], q[1][:, 1]), (np.maximum(xy[0], F(1e-30)), xy[1])), F(0))
    cs = np.clip(quotient((q[0][:, 2], q[1][:, 2]), rr), F(-1), F(1))
    ss = np.clip(rho / rounded(rr), F(0), F(1))
    angle = np.arctan2(rho, rounded((q[0][:, 2], q[1][:, 2])))
    ir = np.clip(np.searchsorted(constants['h'], h, side='right') - 1, 0, len(constants['h']) - 2)
    j = np.clip(np.searchsorted(constants['chi'], angle, side='right') - 1, 0, len(constants['chi']) - 2)
    ah = np.clip((h - constants['h'][ir]) / (constants['h'][ir + 1] - constants['h'][ir]), F(0), F(1))
    ac = np.clip((angle - constants['chi'][j]) / (constants['chi'][j + 1] - constants['chi'][j]), F(0), F(1))
    # Rotation uses the accepted reconstructed interpolation angle.
    ang = constants['chi'][j] + ac * (constants['chi'][j + 1] - constants['chi'][j])
    sn, cc = np.sin(ang), np.cos(ang)
    Q = np.stack([-cc * cx, sx, sn * cx, -cc * sx, -cx, sn * sx, sn, np.zeros_like(sn), cc], -1).reshape(-1, 3, 3)
    B = g['target'].astype('f4')
    dloc = g['dloc'].astype('f4')
    d = g['d'].astype('f4')
    A = np.zeros((len(h), 3, 3), 'f4')
    dot = np.zeros((len(h), 3), 'f4')
    for k in range(3):
        A += B[:, k, :, None] * Q[:, k, None, :]
        dot += d[:, k, None] * Q[:, k]
    K = A - dloc[:, :, None] * dot[:, None, :]
    T = np.stack(
        [
            np.stack(
                [
                    K[:, i, 0] * K[:, k, 0],
                    K[:, i, 1] * K[:, k, 1],
                    K[:, i, 2] * K[:, k, 2],
                    K[:, i, 0] * K[:, k, 2] + K[:, i, 2] * K[:, k, 0],
                ],
                -1,
            )
            for i, k in PAIRS
        ],
        1,
    )
    if st is None:
        st = np.sqrt(np.maximum(0.0, 1 - g['d'][:, 2] ** 2)).astype('f4')
    cp = np.where(st > F(1e-12), (d[:, 0] * cx + d[:, 1] * sx) / np.maximum(st, F(1e-30)), F(1))
    cm = cosine(cp)
    prof = profile_at(h, constants)
    st_prof = strato_profile(h, constants) if 'strato_ext' in constants else np.zeros_like(h)
    cells = np.stack(
        [
            ir * len(constants['chi']) + j,
            ir * len(constants['chi']) + j + 1,
            (ir + 1) * len(constants['chi']) + j,
            (ir + 1) * len(constants['chi']) + j + 1,
        ],
        -1,
    )
    sw = np.stack([(F(1) - ah) * (F(1) - ac), (F(1) - ah) * ac, ah * (F(1) - ac), ah * ac], -1)
    return dict(
        h=h,
        ds=ds,
        cells=cells,
        sw=sw,
        ir=ir,
        j=j,
        T=T,
        cm=cm,
        prof=prof,
        st_prof=st_prof,
        cs=cs,
        ss=ss,
        Q=Q,
        d=d,
        valid=valid,
        r_DS=rr,
        qz_DS=(q[0][:, 2], q[1][:, 2]),
        t_DS=tt,
    )


def hg(u, gg):
    den = np.maximum(F(1e-20), (F(1) + gg * gg) - (F(2) * gg) * u)
    return (F(1) - gg * gg) / (F(4 * math.pi) * (den * np.sqrt(den)))


def mie_phase(u, chi):
    """Scalar L32 table, per steradian; FP32 recurrence matches STRATO CUDA."""
    u = np.asarray(u, 'f4')
    p0 = np.ones_like(u)
    p1 = u.copy()
    val = p0[..., None] * chi[0]
    val += p1[..., None] * chi[1]
    for l in range(2, 33):
        pn = ((F(2 * l - 1) * u) * p1 - F(l - 1) * p0) / F(l)
        val += pn[..., None] * chi[l]
        p0, p1 = p1, pn
    return val / F(4 * math.pi)


def tropo_phase(u, c):
    """kernel.cu tropo_phase (-DTROPO_MIE): per-wavelength table per sr, linear in u; FP32 like CUDA.
    u[...] -> [..., W]."""
    bins = int(c['tropo_bins'])
    W = int(c['W'])
    lut = np.asarray(c['tropo_lut'], 'f4').reshape(bins + 1, W)
    x = (np.clip(np.asarray(u, 'f4'), F(-1), F(1)) + F(1)) * F(bins / 2)
    i = np.minimum(x.astype('i4'), bins - 1)
    f = (x - i.astype('f4'))[..., None]
    a = lut[i]
    b = lut[i + 1]
    return a + f * (b - a)


def strato_profile(h, c):
    p = c['strato_profile']
    t = (h - F(p['center'])) / F(p['sigma'])
    return np.exp(F(-0.5) * (t * t)) / F(p['norm'])


def sun_columns(q, c):
    hh = np.clip(q['h'], F(0), F(80))
    height = sub(q['r_DS'], (R, F(0)))
    height = tuple(np.where(ftz_value(height) > F(0), x, F(0)) for x in height)
    rad = add((R, F(0)), height)
    mg = neg(root(divide(mul(height, add((F(2) * R, F(0)), height)), mul(rad, rad))))
    delta = sub(divide(q['qz_DS'], q['r_DS']), mg)
    zero = ftz_value(delta) == F(0)
    delta = tuple(np.where(zero, F(0), x) for x in delta)
    lit = rounded(delta) >= F(0)
    delta = tuple(np.where(lit, x, F(0)) for x in delta)
    x = np.clip(rounded(root(divide(delta, sub((F(1), F(0)), mg)))), F(0), F(1))
    i = np.clip(np.searchsorted(c['sun_h'], hh, side='right') - 1, 0, len(c['sun_h']) - 2)
    a = (hh - c['sun_h'][i]) / (c['sun_h'][i + 1] - c['sun_h'][i])
    f = x * F(c['sun_col'].shape[1] - 1)
    j = np.clip(f.astype('i4'), 0, c['sun_col'].shape[1] - 2)
    b = f - j.astype('f4')
    col = ((F(1) - a) * (F(1) - b))[:, None] * c['sun_col'][i, j]
    col += (a * (F(1) - b))[:, None] * c['sun_col'][i + 1, j]
    col += ((F(1) - a) * b)[:, None] * c['sun_col'][i, j + 1]
    col += (a * b)[:, None] * c['sun_col'][i + 1, j + 1]
    st = np.zeros(len(col), 'f4')
    if 'strato_ext' in c:
        st = ((F(1) - a) * (F(1) - b)) * c['strato_sun_col'][i, j]
        st += (a * (F(1) - b)) * c['strato_sun_col'][i + 1, j]
        st += ((F(1) - a) * b) * c['strato_sun_col'][i, j + 1]
        st += (a * b) * c['strato_sun_col'][i + 1, j + 1]
    return col, lit, st


def transport(g, c, field, first=False, boundary=None, statistics=None, return_basis=False, sh_source=None):
    """Device mirror: _transport in the FP32 flush-to-zero mode of the CUDA build (flush_to_zero)."""
    with flush_to_zero():
        return _transport(g, c, field, first, boundary, statistics, return_basis, sh_source)


def _transport(g, c, field, first=False, boundary=None, statistics=None, return_basis=False, sh_source=None):
    """sh_source (A6 SH_AUX): optional [cell][wave][KPAD] copy of field[:,10:571,:];
    when given the SH contraction reads it, exactly like kernel.cu under -DSH_AUX."""
    n, W = len(g['d']), c['W']
    Y = basis(g['d'][:, 2], c.get('L', 32))
    B, dloc, pu, direct = frame_terms(g)
    st = np.sqrt(np.maximum(0.0, 1 - g['d'][:, 2] ** 2)).astype('f4')
    total = np.zeros((n, W, 4), 'f4')
    correction = np.zeros_like(total)
    prefix = SimpsonPrefix(g, c)
    cache = np.zeros((n, 4, 33, W), 'f4')
    tags = np.full((n, 4), -1, 'i4')
    gg = c['g']
    mix = c['mix']
    rows = np.arange(n)
    mie = 'tropo_lut' in c
    phase = tropo_phase(g['d'][:, 2].astype('f4'), c) if mie else hg(g['d'][:, 2].astype('f4'), gg)
    strat = 'strato_ext' in c
    combine = c.get('sh_combine', False) and not first
    if combine:
        sh_source = combined_coefficients(
            field, c, c.get('L', 32), skip_thin=c.get('sh_combine_skip', False)
        ).transpose(0, 2, 1)
    if strat:
        sphase = mie_phase(g['d'][:, 2], c['strato_chi'])
        scache = np.zeros_like(cache)
        if not first and (field.ndim != 3 or field.shape[1] != 10 + 2 * count(c.get('L', 32))):
            raise ValueError('E_STRATO_SOURCE_SHAPE: expected 1132 coefficients')
    if statistics is None:
        statistics = {}
    statistics.setdefault('samples', 0)
    statistics.setdefault('angular_pairs', 0)
    statistics.setdefault('corner_aliases', 0)
    for seg in range(g['edges'].shape[1] - 1):
        prefix.begin(seg)
        for step in range(len(c['u'])):
            q = sample(g, seg, step, c, st)
            if not q['valid'].any():
                continue
            statistics['samples'] += int(q['valid'].sum())
            if strat:
                qs = q['st_prof']
            mid = prefix.node(q)
            att = q['ds'][:, None] * np.exp(-mid)
            val = np.zeros_like(total)
            logs = np.zeros((n, W), 'f4')
            wg = np.zeros_like(logs)
            am = np.zeros_like(logs)
            if first:
                sc, lit, st_sc = sun_columns(q, c)
                optical = np.zeros((n, W), 'f4')
                for k in range(NCOMP):
                    optical += sc[:, k, None] * c['ext'][k]
                if strat:
                    optical += st_sc[:, None] * c['strato_ext']
                sun = np.exp(-optical) * lit[:, None]
                ground = np.zeros((n, W), 'f4')
                if strat:
                    sground = np.zeros((n, W), 'f4')
                # Ground angular pool: exact paired HG phases; no truncation.
                mu = np.zeros(n, 'f4')
                exd = np.zeros(n, 'f4')
                eyd = np.zeros(n, 'f4')
                for k in range(3):
                    mu += q['d'][:, k] * q['Q'][:, k, 2]
                    exd += q['d'][:, k] * q['Q'][:, k, 0]
                    eyd += q['d'][:, k] * q['Q'][:, k, 1]
                for corner in range(4):
                    cells = q['cells'][:, corner]
                    gd = c['gd'][cells]
                    m0 = mu[:, None] * gd[:, :, 2] + exd[:, None] * gd[:, :, 0]
                    m1 = abs(eyd)[:, None] * gd[:, :, 1]
                    if mie:
                        ph = (
                            tropo_phase(np.clip(m0 + m1, F(-1), F(1)), c)
                            + tropo_phase(np.clip(m0 - m1, F(-1), F(1)), c)
                        ) / F(2)
                        weighted_phase = q['sw'][:, corner, None, None] * ph
                    else:
                        ph = (hg(np.clip(m0 + m1, F(-1), F(1)), gg) + hg(np.clip(m0 - m1, F(-1), F(1)), gg)) / F(2)
                        weighted_phase = q['sw'][:, corner, None] * ph
                    if strat:
                        sph = (
                            mie_phase(np.clip(m0 + m1, F(-1), F(1)), c['strato_chi'])
                            + mie_phase(np.clip(m0 - m1, F(-1), F(1)), c['strato_chi'])
                        ) / F(2)
                        swphase = q['sw'][:, corner, None, None] * sph
                    for gi in range(c['gd'].shape[1]):
                        ground += (weighted_phase[:, gi] if mie else weighted_phase[:, gi, None]) * c['gv'][cells, gi]
                        if strat:
                            sground += swphase[:, gi] * c['gv'][cells, gi]
                beam = (q['prof'][:, 1, None] * c['asca']) * (sun * (phase if mie else phase[:, None]) + ground)
                if strat:
                    sbeam = (qs[:, None] * c['strato_sca']) * (sun * sphase + sground)
                ray = (sun * q['prof'][:, 0, None]) * c['ext'][0]
                for k in range(4):
                    val[:, :, k] = ray * (
                        mix * F(3 / (8 * math.pi)) * direct[:, k, None]
                        + (F(1) - mix) * F(1 / (8 * math.pi)) * pu[:, k, None]
                    ) + beam * pu[:, k, None] / F(2)
                    if strat:
                        val[:, :, k] += (sbeam * pu[:, k, None]) / F(2)
            if not first:
                if strat:
                    reuse_corners(scache, tags.copy(), q, len(c['chi']))
                statistics['corner_aliases'] += reuse_corners(cache, tags, q, len(c['chi']))
            for corner in range(4):
                cell = q['cells'][:, corner]
                ir = q['ir'] + corner // 2
                if first:
                    rm = c['gm'][cell].transpose(0, 2, 1)
                    aero = np.zeros((n, W), 'f4')
                    density = q['prof'][:, 0]
                else:
                    change = (tags[:, corner] != cell) & q['valid']
                    if change.any():
                        rr = rows[change]
                        cc = cell[change]
                        cache[rr, corner] = F(0)
                        if strat:
                            scache[rr, corner] = F(0)
                        for l in range(c.get('L', 32) + 1):
                            for m in range(l + 1):
                                k = l * (l + 1) // 2 + m
                                cache[rr, corner, m] += Y[rr, k, None] * (
                                    field[cc, 10 + k] if sh_source is None else sh_source[cc, :, k]
                                )
                                if strat and not combine:
                                    scache[rr, corner, m] += Y[rr, k, None] * field[cc, 10 + count(c.get('L', 32)) + k]
                        tags[rr, corner] = cc
                        statistics['angular_pairs'] += len(rr)
                    if strat:
                        saero = warp_sum(q['cm'][:, :, None] * scache[:, corner]) * c['strato_density'][ir, None]
                    aero = warp_sum(q['cm'][:, :, None] * cache[:, corner])
                    if not combine:
                        aero *= c['ha'][ir, None]
                    rm = field[cell, :4].transpose(0, 2, 1)
                    density = c['hr'][ir]
                trace = (rm[:, :, 0] + rm[:, :, 1]) + rm[:, :, 2]
                tensor = np.zeros_like(total)
                for k in range(4):
                    rotated = np.zeros((n, W), 'f4')
                    for u in range(4):
                        rotated += q['T'][:, k, u, None] * rm[:, :, u]
                    tensor[:, :, k] = density[:, None] * (
                        mix * rotated + ((F(1) - mix) / F(3) * trace) * pu[:, k, None]
                    )
                    tensor[:, :, k] += (aero * pu[:, k, None]) / F(2)
                    if strat and not first:
                        tensor[:, :, k] += (saero * pu[:, k, None]) / F(2)
                    val[:, :, k] += q['sw'][:, corner, None] * tensor[:, :, k]
                if not first:
                    tr = (tensor[:, :, 0] + tensor[:, :, 1]) + tensor[:, :, 2]
                    # Valid corner traces set the scale (G/A)^W, continuous in their total weight W (kernel.cu, same rule).
                    trace_accumulate(tr, q['sw'][:, corner, None], logs, wg, am)
            if not first:
                val = trace_scale(val, logs, wg, am)
            total, correction = kahan(total, correction, val * att[:, :, None])
        prefix.finish()
    if boundary is not None:
        E = (F(1) - g['end_fraction'].astype('f4'))[:, None] * boundary[g['end_index']].astype('f4')
        E += g['end_fraction'].astype('f4')[:, None] * boundary[g['end_index'] + 1].astype('f4')
        optical = np.zeros((n, W), 'f4')
        endcol = g['end_columns']
        if 'density_optics' in c:
            from .density import correct_columns

            opt = c['density_optics']
            endcol = correct_columns(g['p'], g['d'], endcol, opt['old'], opt, 24, end=g['edges'][:, -1])
        for k in range(NCOMP):
            optical += endcol[:, k, None].astype('f4') * c['ext'][k]
        if strat:
            from .stratosphere import columns

            stcol = columns(g['p'], g['d'], c['strato_old'], c['strato_profile'], 24).astype('f4')
            optical += stcol[:, None] * c['strato_ext']
        E *= np.exp(-optical) * g['hits_ground'][:, None]
        for k in range(4):
            total[:, :, k] += (E * pu[:, k, None]) / F(2)
    return (total, Y) if return_basis else total


def project(g, radiance, c, Y=None, ground_only=False):
    """Device mirror: _project in the FP32 flush-to-zero mode of the CUDA build (flush_to_zero)."""
    with flush_to_zero():
        return _project(g, radiance, c, Y, ground_only)


def _project(g, radiance, c, Y=None, ground_only=False):
    """FP32 weighted products, deterministic ray-order FP64 cell sums."""
    Y = basis(g['d'][:, 2], c.get('L', 32)) if Y is None else Y
    st = np.sqrt(np.maximum(0.0, 1 - g['d'][:, 2] ** 2))
    cp = np.where(st > 1e-12, g['d'][:, 0] / np.maximum(st, 1e-30), 1.0).astype('f4')
    cm = cosine(cp)
    weight = g['weight'].astype('f4')
    trace = (radiance[:, :, 0] + radiance[:, :, 1]) + radiance[:, :, 2]
    weighted = trace * weight[:, None]
    out = np.zeros((4 + count(c.get('L', 32)), c['W']), 'f8')
    if not ground_only:
        for k in range(4):
            out[k] = np.sum(radiance[:, :, k] * weight[:, None], axis=0, dtype='f8')
        for l in range(c.get('L', 32) + 1):
            for m in range(l + 1):
                k = l * (l + 1) // 2 + m
                out[4 + k] = np.sum(weighted * (Y[:, k] * cm[:, m])[:, None], axis=0, dtype='f8')
    ground = (
        np.sum((trace * (weight * np.maximum(F(0), g['dloc'][:, 2].astype('f4')))[:, None]).astype('f8'), axis=0)
        * c['albedo']
        / math.pi
    )
    return out, ground
