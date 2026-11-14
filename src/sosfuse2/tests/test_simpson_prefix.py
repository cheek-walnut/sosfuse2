"""Simpson optical-depth prefix in backends/numpy_reference.py.

Run: python -m pytest -q -s src/sosfuse2/tests/test_simpson_prefix.py   (-s prints the E4 closure numbers)

The prefix under test is numpy_reference.SimpsonPrefix driven exactly as transport() drives it (begin / node(sample()) / finish) with the
production sample() and profile_at(); nothing is replaced by a synthetic beta. Truth is an independent FP64 integral of the same
profile model (test-local, every panel split at the profile knots it crosses, Gauss-Legendre 32 per piece).
  - node match: profile_at at the node heights == sample() prof (and the historical inline expressions, copied below) within tolerance,
    4 components, density table / two-layer aerosol / stratosphere on and off; extinction * ds == the old dt/ds; the DS height
    helper reproduces sample() h within tolerance.
  - constant and exponential beta, each component alone (air, tropospheric aerosol incl. AEROSOL_2L, ozone, stratosphere):
    FP64 replica of the Simpson rule (same panels and midpoints) to FP32 rounding, truth within the Simpson error bound, and the
    old GL-weight rule (copied below) shown to fail the same check.
  - E4 closure: sum ds beta exp(-tau_k) vs 1 - exp(-tau_end) for the old rule and Simpson, synthetic slant rays
    and the frozen E1 production rays (when present).
"""

import json
import math

import numpy as np
import pytest

from sosfuse2.backends import numpy_reference as ref  # noqa: E402
from sosfuse2.atmosphere import tropo_profile, density  # noqa: E402
from sosfuse2.backends.arithmetic import F, split, kahan  # noqa: E402

from sosfuse2.checks.numerical_checks import within_tol

RE = 6371.0
from sosfuse2.checks import testkit  # noqa: E402

FROZEN = testkit.sky_sos_optional(
    'results/taskB_simpson_20261022/e1_frozen_small.npz'
)  # Optional research record.
Z05 = np.arange(161) * 0.5


# ------------------------------------------------------------------------------------------------ geometry and constants


def ray(h0, el_deg, az_deg=0.0):
    """Start height h0 km, local elevation el; ends at the ground or at 80 km."""
    p = np.array([0.0, 0.0, RE + h0])
    e, a = math.radians(el_deg), math.radians(az_deg)
    d = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    pd, pp = float(p @ d), float(p @ p)
    g = pd * pd - pp + RE * RE
    end = -pd - math.sqrt(g) if (pd < 0 and g > 0) else -pd + math.sqrt(pd * pd - pp + (RE + 80.0) ** 2)
    return p, d, end


RAYS = [
    ray(0.1, 90.0),
    ray(0.1, 2.0),
    ray(0.1, 10.0, 40.0),
    ray(2.0, 5.0, 120.0),
    ray(5.0, 30.0, 200.0),
    ray(30.0, -3.0, 80.0),
    ray(40.0, -60.0, 10.0),
    ray(1.0, -1.0),
]


def geometry(rays=RAYS, cuts=(0.03, 0.15, 0.4, 0.7), lower=(True, False, True, False, True)):
    P = np.array([r[0] for r in rays])
    D = np.array([r[1] for r in rays])
    T = np.array([r[2] for r in rays])
    n = len(rays)
    edges = np.concatenate([np.zeros((n, 1)), T[:, None] * np.asarray(cuts)[None, :], T[:, None]], 1)
    # one degenerate segment (b - a <= 1e-8) on the first ray: skipped by the kernel, must not change tau
    edges[0, 2] = edges[0, 1]
    low = np.tile(np.asarray(lower[: edges.shape[1] - 1]), (n, 1))
    return dict(p=P, d=D, edges=edges, lower=low, target=np.tile(np.eye(3), (n, 1, 1)), dloc=D.copy())


def constants(nq, comp=None, kind='exp'):
    """Two lanes. comp None: every component with production-like options; else component comp (0 air, 1 aerosol, 2 ozone,
    3 stratosphere) alone, kind 'const' / 'exp' / 'aer2' (two-layer aerosol) / 'triangle' (default ozone)."""
    u, wu = np.polynomial.legendre.leggauss(nq)
    c = dict(
        W=2,
        u=(u + 1) / 2,
        wu=wu / 2,
        h=np.linspace(0, 80, 21).astype('f4'),
        chi=np.linspace(0, math.pi, 13).astype('f4'),
        height=F(1.2),
        ext=np.zeros((5, 2), 'f4'),
    )
    ext = {0: [0.045, 0.012], 1: [0.5, 0.4], 2: [0.02, 0.03], 3: [0.02, 0.018]}
    tab = np.stack([np.exp(-Z05 / 8.1), np.exp(-np.abs(Z05 - 24.0) / 6.0), density.no2_profile(Z05)], 1).astype('f4')
    strato = dict(center=20.0, sigma=3.0, norm=3.0 * math.sqrt(2 * math.pi))
    a2 = np.asarray(tropo_profile.parameters(dict(tropo_profile.DEFAULTS, aod550=0.6))['vector'], 'f4')
    if comp is None:
        for k in range(3):
            c['ext'][k] = ext[k]
        c.update(
            density_optics=dict(table=tab, mask=3), aer2=a2, strato_ext=np.asarray(ext[3], 'f4'), strato_profile=strato
        )
        return c
    if comp < 3:
        c['ext'][comp] = ext[comp]
    else:
        c.update(strato_ext=np.asarray(ext[3], 'f4'), strato_profile=strato)
    if kind == 'const':
        if comp == 0:
            c['density_optics'] = dict(table=np.ones((161, 3), 'f4'), mask=1)
        if comp == 1:
            c['height'] = F(1e30)  # exp(-h/1e30) == 1 in FP32
        if comp == 2:
            c['density_optics'] = dict(table=np.ones((161, 3), 'f4'), mask=2)
        if comp == 3:
            c['strato_profile'] = dict(center=20.0, sigma=1e30, norm=1.0)
    elif kind == 'exp':
        if comp == 2:
            c['density_optics'] = dict(
                table=np.stack([np.ones(161), np.exp(-Z05 / 6.0), density.no2_profile(Z05)], 1).astype('f4'), mask=2
            )
    elif kind == 'aer2':
        c['aer2'] = a2
    return c


# ------------------------------------------------------------------------------------------------ FP64 model and truth


def profile64(h, c):
    """[..., 4] FP64 components with the FP32 constants of c (same model as numpy_reference.profile_at)."""
    h = np.clip(np.asarray(h, 'f8'), 0.0, 80.0)
    dens = c.get('density_optics')
    mask = dens['mask'] if dens else 0
    tab = np.asarray(dens['table'], 'f4').astype('f8') if dens else None
    air = np.interp(h, Z05, tab[:, 0]) if mask & 1 else np.exp(-h / 8.0)
    aer = (
        tropo_profile.profile(h, np.asarray(c['aer2'], 'f4').astype('f8'))
        if 'aer2' in c
        else np.exp(-h / float(F(c['height'])))
    )
    oz = np.interp(h, Z05, tab[:, 1]) if mask & 2 else np.maximum(0.0, 1.0 - np.abs(h - 25.0) / 15.0)
    if 'strato_ext' in c:
        p = c['strato_profile']
        t = (h - float(F(p['center']))) / float(F(p['sigma']))
        st = np.exp(-0.5 * t * t) / float(F(p['norm']))
    else:
        st = np.zeros_like(h)
    return np.stack([air, aer, oz, st], -1)


def beta64(h, c):
    pr = profile64(h, c)
    out = pr[..., :3] @ np.asarray(c['ext'][:3], 'f4').astype('f8')
    return out + pr[..., 3:4] * np.asarray(c['strato_ext'], 'f4').astype('f8') if 'strato_ext' in c else out


def knots(c):
    k = [0.0, 80.0]
    dens = c.get('density_optics')
    if dens:
        k += list(Z05)
    if not (dens and dens['mask'] & 2):
        k += [10.0, 25.0, 40.0]
    if 'aer2' in c:
        v = np.asarray(c['aer2'], 'f4').astype('f8')
        k += [v[0], v[0] + 1 / v[1]]
    return np.unique(np.clip(k, 0, 80))


def height64(g, i, t):
    return np.linalg.norm(g['p'][i][None, :] + np.asarray(t, 'f8')[:, None] * g['d'][i][None, :], axis=-1) - RE


def node_t64(g, i, seg, u):
    a, b = g['edges'][i, seg], g['edges'][i, seg + 1]
    L = b - a
    return a + L * u * u if g['lower'][i, seg] else b - L * (1 - u) ** 2


def truth(g, c, gl=32):
    """FP64 tau at every node [n, nseg, nq, W] and segment end [n, nseg, W]: pieces between consecutive nodes / edges / knot crossings."""
    n, nseg, nq, W = len(g['d']), g['edges'].shape[1] - 1, len(c['u']), c['W']
    x, w = np.polynomial.legendre.leggauss(gl)
    x = (x + 1) / 2
    w = w / 2
    K = knots(c)
    nodes = np.zeros((n, nseg, nq, W))
    ends = np.zeros((n, nseg, W))
    for i in range(n):
        p, d = g['p'][i], g['d'][i]
        pd, pp = p @ d, p @ p
        disc = pd * pd - pp + (RE + K) ** 2
        roots = np.concatenate([-pd - np.sqrt(disc[disc > 0]), -pd + np.sqrt(disc[disc > 0])])
        tau = np.zeros(W)
        for seg in range(nseg):
            a, b = g['edges'][i, seg], g['edges'][i, seg + 1]
            if b - a <= 1e-8:
                ends[i, seg] = tau
                nodes[i, seg] = tau
                continue
            tk = np.array([node_t64(g, i, seg, u) for u in c['u']])
            pts = np.unique(np.concatenate([[a, b], tk, roots[(roots > a) & (roots < b)]]))
            lo, hi = pts[:-1], pts[1:]
            tt = lo[:, None] + (hi - lo)[:, None] * x[None, :]
            piece = np.einsum(
                'pq,pqw->pw', (hi - lo)[:, None] * w[None, :], beta64(height64(g, i, tt.ravel()).reshape(tt.shape), c)
            )
            cum = tau + np.cumsum(piece, 0)
            for k, t in enumerate(tk):
                nodes[i, seg, k] = cum[np.searchsorted(hi, t)]
            tau = cum[-1]
            ends[i, seg] = tau
    return nodes, ends


def simpson64(g, c):
    """FP64 replica of the rule (same panels, midpoints, segment ends) on the FP64 model: the implementation check."""
    n, nseg, nq, W = len(g['d']), g['edges'].shape[1] - 1, len(c['u']), c['W']
    nodes = np.zeros((n, nseg, nq, W))
    ends = np.zeros((n, nseg, W))
    for i in range(n):
        tau = np.zeros(W)
        for seg in range(nseg):
            a, b = g['edges'][i, seg], g['edges'][i, seg + 1]
            if b - a <= 1e-8:
                nodes[i, seg] = tau
                ends[i, seg] = tau
                continue
            t = np.concatenate([[a], [node_t64(g, i, seg, u) for u in c['u']], [b]])
            bt = beta64(height64(g, i, t), c)
            bm = beta64(height64(g, i, (t[:-1] + t[1:]) / 2), c)
            pieces = (np.diff(t) / 6)[:, None] * (bt[:-1] + 4 * bm + bt[1:])
            cum = np.cumsum(pieces, 0)
            nodes[i, seg] = tau + cum[:nq]
            tau = tau + cum[-1]
            ends[i, seg] = tau
    return nodes, ends


def simpson_bound(g, c, comp):
    """Sum over panels of l^5 max|beta''''| / 2880 (smooth one-component profiles; d/dt along the path at most d/dh)."""
    n, nseg, nq = len(g['d']), g['edges'].shape[1] - 1, len(c['u'])
    out = np.zeros((n, nseg, nq + 1, c['W']))
    ext = np.asarray(c['strato_ext'] if comp == 3 else c['ext'][comp], 'f4').astype('f8')
    for i in range(n):
        for seg in range(nseg):
            a, b = g['edges'][i, seg], g['edges'][i, seg + 1]
            if b - a <= 1e-8:
                continue
            t = np.concatenate([[a], [node_t64(g, i, seg, u) for u in c['u']], [b]])
            ell = np.diff(t)
            hmin = np.minimum(height64(g, i, t[:-1]), height64(g, i, t[1:]))
            if comp in (0, 1):
                H = 8.0 if comp == 0 else float(F(c['height']))
                d4 = np.exp(-np.maximum(hmin - ell, 0) / H) / H**4
            else:  # gaussian: max |d^4/dx^4 exp(-x^2/2)| = 3
                s = c['strato_profile']
                d4 = np.full_like(ell, 3.0 / (float(s['sigma']) ** 4 * float(s['norm'])))
            # path curvature: d^2h/dt^2 <= 1/R adds lower-order terms; bound them with a factor 4 safety
            out[i, seg, :] = 4 * (ell**5 * d4 / 2880)[:, None] * ext[None, :]
    return np.cumsum(out.reshape(n, -1, c['W']), 1).reshape(out.shape)


# ------------------------------------------------------------------------------------------------ production traces


def production(g, c):
    """numpy_reference.SimpsonPrefix exactly as transport() drives it; also node ds and beta for the closure sums."""
    n, nseg, nq, W = len(g['d']), g['edges'].shape[1] - 1, len(c['u']), c['W']
    st = np.sqrt(np.maximum(0.0, 1 - g['d'][:, 2] ** 2)).astype('f4')
    P = ref.SimpsonPrefix(g, c)
    nodes = np.full((n, nseg, nq, W), np.nan)
    ends = np.zeros((n, nseg, W))
    ds = np.zeros((n, nseg, nq))
    beta = np.zeros((n, nseg, nq, W))
    for seg in range(nseg):
        P.begin(seg)
        for step in range(nq):
            q = ref.sample(g, seg, step, c, st)
            if not q['valid'].any():
                continue
            nodes[:, seg, step] = P.node(q)
            ds[:, seg, step] = q['ds']
            beta[:, seg, step] = ref.extinction_profile(q['prof'], c, q['st_prof'])
        P.finish()
        ends[:, seg] = P.tau - P.tc
    return nodes, ends, ds, beta


def old_gl(g, c):
    """Historical prefix: mid = (tau - tc) + dt/2, tau += dt (Kahan)."""
    n, nseg, nq, W = len(g['d']), g['edges'].shape[1] - 1, len(c['u']), c['W']
    st = np.sqrt(np.maximum(0.0, 1 - g['d'][:, 2] ** 2)).astype('f4')
    tau = np.zeros((n, W), 'f4')
    tc = np.zeros_like(tau)
    nodes = np.full((n, nseg, nq, W), np.nan)
    ends = np.zeros((n, nseg, W))
    for seg in range(nseg):
        for step in range(nq):
            q = ref.sample(g, seg, step, c, st)
            if not q['valid'].any():
                continue
            dt = np.zeros_like(tau)
            for k in range(3):
                dt += (q['prof'][:, k] * q['ds'])[:, None] * c['ext'][k]
            if 'strato_ext' in c:
                dt += (ref.strato_profile(q['h'], c) * q['ds'])[:, None] * c['strato_ext']
            nodes[:, seg, step] = (tau - tc) + dt / F(2)
            tau, tc = kahan(tau, tc, dt)
        ends[:, seg] = tau - tc
    return nodes, ends


def old_inline_profile(h, c):
    """Historical sample() profile block, copied verbatim."""
    prof = np.stack([np.exp(-h / F(8)), np.exp(-h / c['height']), np.maximum(F(0), F(1) - abs(h - F(25)) / F(15))], -1)
    if 'density_optics' in c:
        from sosfuse2.atmosphere.density import interpolate32

        optics = c['density_optics']
        tab = interpolate32(h, optics['table'])
        if optics['mask'] & 1:
            prof[:, 0] = tab[:, 0]
        if optics['mask'] & 2:
            prof[:, 2] = tab[:, 1]
    if 'aer2' in c:
        from sosfuse2.atmosphere.tropo_profile import profile32

        prof[:, 1] = profile32(h, c['aer2'])
    return prof


def errors(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.abs(np.asarray(a, 'f8')[m] - np.asarray(b, 'f8')[m]), np.abs(np.asarray(b, 'f8')[m])


# ------------------------------------------------------------------------------------------------ tests


@pytest.mark.parametrize('opt', ['full', 'plain', 'none', 'aer2_only'])
def test_node_profile_matches_sample(opt):
    """Node match: profile_at(node h) == sample() prof4 == old inline expressions (3 cols) and strato_profile (col 3)."""
    c = constants(12)
    if opt != 'full':
        for k in ('density_optics', 'aer2'):
            c.pop(k)
    if opt == 'none':
        c.pop('strato_ext')
        c.pop('strato_profile')
    if opt == 'aer2_only':
        c.pop('strato_ext')
        c.pop('strato_profile')
        c['aer2'] = constants(12)['aer2']
    g = geometry()
    st = np.sqrt(np.maximum(0.0, 1 - g['d'][:, 2] ** 2)).astype('f4')
    pp, dd = split(g['p']), split(g['d'])
    checked = 0
    for seg in range(g['edges'].shape[1] - 1):
        for step in range(len(c['u'])):
            q = ref.sample(g, seg, step, c, st)
            p4 = ref.profile_at(q['h'], c)
            assert p4.dtype == np.float32 and p4.shape == (len(q['h']), 5)
            assert within_tol(p4, q['prof'])
            assert within_tol(old_inline_profile(q['h'], c), q['prof'][:, :3])
            if 'strato_ext' in c:
                assert within_tol(ref.strato_profile(q['h'], c), q['st_prof'])
            else:
                assert not q['st_prof'].any()
            # extinction * ds == the old dt (sum (prof_k ds) ext_k + (st ds) strato_ext) up to FP32 reassociation
            dt = np.zeros((len(q['h']), c['W']), 'f4')
            for k in range(3):
                dt += (q['prof'][:, k] * q['ds'])[:, None] * c['ext'][k]
            if 'strato_ext' in c:
                dt += (ref.strato_profile(q['h'], c) * q['ds'])[:, None] * c['strato_ext']
            bds = ref.extinction_profile(p4, c, q['st_prof']) * q['ds'][:, None]
            assert np.all(np.abs(bds.astype('f8') - dt) <= 4e-7 * np.abs(dt) + 1e-37)
            # the DS height helper of the prefix gives sample()'s node height within tolerance
            assert within_tol(ref.path_height(ref.path_point(pp, dd, q['t_DS'])[2]), q['h'])
            checked += int(q['valid'].sum())
    assert checked > 300


CASES = [
    (0, 'const'),
    (1, 'const'),
    (2, 'const'),
    (3, 'const'),
    (0, 'exp'),
    (1, 'exp'),
    (1, 'aer2'),
    (2, 'exp'),
    (2, 'triangle'),
    (3, 'exp'),
]


@pytest.mark.parametrize('comp,kind', CASES)
@pytest.mark.parametrize('nq', [12, 24])
def test_component_prefix(comp, kind, nq):
    g = geometry()
    c = constants(nq, comp, kind if kind != 'triangle' else 'exp')
    if kind == 'triangle':
        c.pop('density_optics', None)
    nodes, ends, _, _ = production(g, c)
    tn, te = truth(g, c)
    rn, re_ = simpson64(g, c)
    # degenerate segment: no change of tau across it
    assert np.array_equal(ends[0, 1], ends[0, 0])
    assert np.isfinite(nodes[np.isfinite(nodes)]).all() and np.all(np.diff(ends, axis=1) >= -1e-6 * np.abs(ends[:, 1:]))
    # (a) implementation == rule: FP64 replica of the same panels, FP32 rounding only
    d, r = errors(np.concatenate([nodes.ravel(), ends.ravel()]), np.concatenate([rn.ravel(), re_.ravel()]))
    assert np.all(d <= 3e-6 * r + 1e-9), (d.max(), (d / np.maximum(r, 1e-30)).max())
    # (b) rule vs truth, elementwise: nodes [n, nseg, nq, W] then segment ends [n, nseg, W]
    got = np.concatenate([nodes.reshape(-1), ends.reshape(-1)])
    want = np.concatenate([tn.reshape(-1), te.reshape(-1)])
    ok = np.isfinite(got)
    da = np.abs(got[ok] - want[ok])
    ra = np.abs(want[ok])
    gn, ge = old_gl(g, c)
    gg = np.concatenate([gn.reshape(-1), ge.reshape(-1)])
    dg = np.abs(gg[ok] - want[ok])
    if kind == 'const':
        assert np.all(da <= 2e-6 * ra + 1e-9), (comp, kind, da.max())
        assert (
            dg / np.maximum(ra, 1e-9)
        ).max() > 0.1  # the old rule is wrong at the nodes even for a constant beta (+156 %)
    elif kind == 'exp' and comp in (0, 1, 3):  # smooth: Simpson error bound (cumulative over panels)
        bound = simpson_bound(g, c, comp)
        tb = np.concatenate([bound[:, :, :nq].reshape(-1), bound[:, :, nq].reshape(-1)])[ok]
        assert np.all(da <= tb + 3e-6 * ra + 1e-9), (comp, kind, (da - tb).max())
        assert dg.max() > 30 * da.max()
    else:  # kinks inside panels (density table, ozone triangle, two-layer aerosol): O(l^2) panels, accepted risk (E1_REPORT)
        assert da.max() <= 2e-3 * max(ra.max(), 1.0) and dg.max() > 2 * da.max(), (comp, kind, da.max(), dg.max())
    print(
        f'PREFIX comp={comp} {kind:<8} nq={nq}: simpson max|dtau| {da.max():.2e} (tau max {ra.max():.2f}), old GL {dg.max():.2e}'
    )


def closure(nodes, ends, ds, beta, tau_end):
    S = np.einsum('isk,iskw->iw', ds, beta * np.exp(-np.where(np.isfinite(nodes), nodes, 0.0)))
    C = -np.expm1(-tau_end)
    ok = C >= 1e-6
    return np.where(ok, S / np.where(ok, C, 1.0) - 1, 0.0)


def frozen_sets():
    with np.load(FROZEN) as Z:
        sets = json.loads(str(Z['sets']))
        out = []
        for s in sets:
            g = {k: Z[f"{s['name']}__{k}"] for k in ('p', 'd', 'edges', 'lower', 'target', 'dloc', 'hits_ground')}
            pre = s['prefix']
            nq = s['nq']
            c = dict(
                W=int(Z[pre + 'W']),
                u=Z[pre + f'u{nq}'],
                wu=Z[pre + f'wu{nq}'],
                h=Z[pre + 'h'],
                chi=Z[pre + 'chi'],
                height=np.float32(Z[pre + 'height']),
                ext=Z[pre + 'ext'],
            )
            c['ext'] = np.pad(c['ext'], ((0, 2), (0, 0)))
            if pre + 'dens_table' in Z:
                c['density_optics'] = dict(
                    table=np.column_stack([Z[pre + 'dens_table'], density.no2_profile(Z05)]),
                    mask=int(Z[pre + 'dens_mask']),
                )
            if pre + 'aer2' in Z:
                c['aer2'] = Z[pre + 'aer2']
            if pre + 'strato_ext' in Z:
                sp = Z[pre + 'strato_profile']
                c.update(
                    strato_ext=Z[pre + 'strato_ext'],
                    strato_profile=dict(center=float(sp[0]), sigma=float(sp[1]), norm=float(sp[2])),
                )
            out.append((s['name'], g, c))
    return out


def slant(g):
    """E1 slant class (e1_tau.classify): |start elevation| < 60, start above 2 km, no tangent point, no ground hit."""
    p, d = np.asarray(g['p'], 'f8'), np.asarray(g['d'], 'f8')
    r0 = np.linalg.norm(p, axis=-1)
    pd = np.sum(p * d, -1)
    end = g['edges'][:, -1]
    el = np.degrees(np.arcsin(np.clip(pd / r0, -1, 1)))
    ground = np.asarray(g.get('hits_ground', np.zeros(len(p), bool)), bool)
    tangent = (pd < 0) & (-pd < end) & ~ground
    return ~tangent & (np.abs(el) < 60) & (r0 - RE > 2) & ~ground


def test_e4_closure():
    """E4: stop if the Simpson closure error is >= 1e-3 and more than twice the old rule's, on the
    slant class and on all rays. The closure also carries the GL source quadrature error, so both rules can
    exceed 1e-3 on under-resolved slant segments; the rule compares the two prefixes on the same nodes and weights."""
    rows = []
    g = geometry(
        [ray(h, el, az) for h in (0.1, 2.0, 5.0) for el, az in ((2.0, 0.0), (5.0, 60.0), (10.0, 120.0), (30.0, 200.0))]
    )
    c = constants(12)
    sets = [('synthetic', g, c)] + (frozen_sets() if FROZEN is not None else [])
    for name, g, c in sets:
        tn, te = truth(g, c) if name == 'synthetic' else (None, None)
        nodes, ends, ds, beta = production(g, c)
        gn, ge = old_gl(g, c)
        t_end = te[:, -1] if te is not None else ends[:, -1].astype('f8')
        es = closure(nodes, ends, ds, beta, t_end)
        eg = closure(gn, ge, ds, beta, t_end)
        sl = slant(g)
        row = dict(
            set=name,
            rays=len(g['d']),
            slant=int(sl.sum()),
            simpson_max=float(np.abs(es).max()),
            gl_max=float(np.abs(eg).max()),
            simpson_slant_max=float(np.abs(es[sl]).max()) if sl.any() else None,
            gl_slant_max=float(np.abs(eg[sl]).max()) if sl.any() else None,
            end_tau='truth' if te is not None else 'simpson',
        )
        rows.append(row)
        print('E4_CLOSURE', json.dumps(row))
        assert not (row['simpson_max'] >= 1e-3 and row['simpson_max'] > 2 * row['gl_max'])
        if sl.any():
            assert not (row['simpson_slant_max'] >= 1e-3 and row['simpson_slant_max'] > 2 * row['gl_slant_max'])
    assert rows[0]['slant'] > 0
