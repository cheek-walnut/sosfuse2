"""Extended-domain (80 < top_km <= 200 km) MLX transport unit tests."""

import copy
from functools import partial
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sosfuse2.atmosphere import density as density_mod
from sosfuse2.atmosphere.stratosphere import cell_density as strato_cell_density
from sosfuse2.atmosphere.tropo_profile import profile32
from sosfuse2.backends import mlx_ops
from sosfuse2.backends import numpy_reference as ref
from sosfuse2.backends.sh_layout import count
from sosfuse2.foundation import top as top_mod
from sosfuse2.rays import shell_segments as shells
from sosfuse2.solve.engine_core import basis_at
from sosfuse2.solve.ray_geometry import normal_state


R = shells.R


def _require_metal():
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    return mx


def _extended_constants(
    *,
    top_km=200.0,
    step_km=0.5,
    upper_air='us76_neff',
    W=1,
    L=16,
    nq=4,
    heights=None,
    strato=True,
    combine=False,
    skip_thin=False,
    albedo=0.0,
    aer2=False,
    with_optics=True,
    ng=0,
):
    if top_km > 80.0:
        domain = top_mod.resolve(dict(top_km=top_km, upper_heights_km=[120.0, top_km], upper_air=upper_air), step=step_km)
    else:
        domain = top_mod.Domain()
    state = copy.deepcopy(normal_state())
    state['density'] = dict(density_mod.DEFAULTS, profile_step_km=step_km)
    optics = density_mod.build(state, domain) if with_optics else None
    if heights is None:
        h = np.array([0.0, 8.0, 40.0, 80.0, 90.0, top_km] if top_km > 80.0 else [0.0, 8.0, 40.0, 80.0], 'f4')
    else:
        h = np.asarray(heights, 'f4')
    chi = np.deg2rad([0.0, 85.0, 95.0, 105.0, 180.0]).astype('f4')
    ns = len(h) * len(chi)
    u, wu = shells.gl(nq)
    if optics is not None:
        hr = density_mod.interpolate32(h, optics['table'], domain)[:, 0].astype('f4')
    else:
        hr = np.exp(-h / 8.0).astype('f4')
        if ref.support_extended(domain):
            hr = np.where(h > domain.air_end, np.float32(0.0), hr)
    aer2_vec = np.array([1.0, 0.7, 0.9, 1.0, 0.2, 0.35, 1.0], 'f4')
    if aer2:
        ha = profile32(h, aer2_vec).astype('f4')
    else:
        ha = np.exp(-h / 1.2).astype('f4')
    if ref.support_extended(domain):
        ha = np.where(h > domain.lower_end, np.float32(0.0), ha)
    sun_h = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 140.0, top_km] if top_km > 80.0 else [0.0, 20.0, 40.0, 60.0, 80.0], 'f4')
    sun_col = np.zeros((len(sun_h), 5, 5), 'f4')
    sun_col[:, :, 0] = np.linspace(1.0, 0.005, len(sun_h), dtype='f4')[:, None] * np.linspace(4.0, 0.3, 5, dtype='f4')[None, :]
    rng = np.random.default_rng(1200 + W + L + ng + int(top_km))
    if ng > 0:
        dirs = rng.normal(size=(ns, ng, 3)).astype('f4')
        dirs[:, :, 1] = np.abs(dirs[:, :, 1])
        dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
        gv = rng.uniform(0.0, 0.03, (ns, ng, W)).astype('f4')
        gm = rng.uniform(0.0, 0.02, (ns, 4, W)).astype('f4')
    else:
        dirs = np.zeros((ns, 0, 3), 'f4')
        gv = np.zeros((ns, 0, W), 'f4')
        gm = np.zeros((ns, 4, W), 'f4')
    ext = np.broadcast_to(np.array([0.05, 0.08, 0.01, 0.004, 0.002], 'f4')[:, None], (5, W)).copy()
    ext *= np.linspace(1.0, 0.6, W, dtype='f4')[None, :]
    c = dict(
        domain=domain,
        L=L,
        W=W,
        u=u,
        wu=wu,
        h=h,
        chi=chi,
        height=np.float32(1.2),
        mix=np.float32(0.97),
        g=np.float32(0.65),
        albedo=float(albedo),
        hr=hr,
        ha=ha,
        ext=ext,
        asca=np.linspace(0.05, 0.08, W, dtype='f4'),
        sun_h=sun_h,
        sun_col=sun_col,
        gm=gm,
        gd=dirs,
        gv=gv,
    )
    old_ns = NS(domain=domain, probe=NS(boundaries=partial(shells.boundaries, domain=domain), gl=shells.gl))
    if optics is not None:
        optics['old'] = old_ns
        c['density_optics'] = optics
    if aer2:
        c['aer2'] = aer2_vec
    if strato:
        z = np.linspace(0.0, 80.0, 8001)
        center, sigma = 21.0, 3.0
        norm = float(np.trapezoid(np.exp(-0.5 * ((z - center) / sigma) ** 2), z))
        profile = dict(center=center, sigma=sigma, norm=norm)
        sden, _ = strato_cell_density(h.astype('f8'), profile)
        if ref.support_extended(domain):
            sden = np.where(h > domain.lower_end, 0.0, sden)
        l_idx = np.arange(33, dtype='f4')[:, None]
        st_chi = ((2.0 * l_idx + 1.0) * (0.62 ** l_idx) * np.linspace(1.0, 0.9, W, dtype='f4')[None, :]).astype('f4')
        st_chi[0] = 1.0
        st_ext = np.linspace(0.03, 0.018, W, dtype='f4')
        c.update(
            strato_ext=st_ext,
            strato_sca=st_ext * np.float32(0.98),
            strato_chi=st_chi,
            strato_density=sden.astype('f4'),
            strato_sun_col=(np.linspace(1.0, 0.0, len(sun_h), dtype='f4')[:, None] * np.linspace(4.0, 0.4, 5, dtype='f4')[None, :]).astype('f4'),
            strato_profile=profile,
            strato_old=old_ns,
        )
        if combine:
            c.update(sh_combine=True, sh_combine_skip=skip_thin)
    return c


def _ray_group(p_list, d_list, edges_list, lower_list, nq=4):
    p = np.asarray(p_list, 'f8')
    d = np.asarray(d_list, 'f8')
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    target = basis_at(p)
    nr = len(d)
    return dict(
        p=p,
        d=d,
        dloc=np.einsum('nij,ni->nj', target, d),
        target=target,
        weight=np.linspace(0.5, 1.2, nr, dtype='f8'),
        edges=np.asarray(edges_list, 'f8'),
        lower=np.asarray(lower_list, bool),
        hits_ground=np.zeros(nr, bool),
        end_index=np.zeros(nr, 'i4'),
        end_fraction=np.zeros(nr, 'f8'),
        end_columns=np.zeros((nr, 5), 'f8'),
        nq=nq,
    )


def _assert_close_rel(got, expected, *, wave_axis=-1, rtol=1e-3, err_msg=''):
    assert got.shape == expected.shape
    assert np.all(np.isfinite(got))
    axes = tuple(i for i in range(expected.ndim) if i != wave_axis % expected.ndim)
    scale = np.max(np.abs(expected), axis=axes)
    error = np.max(np.abs(got - expected), axis=axes)
    np.testing.assert_array_less(error, np.nextafter(rtol * scale.astype('f8'), np.inf), err_msg=err_msg)


@pytest.mark.parametrize('upper_air', ['us76_neff', 'vacuum'])
def test_extended_80km_ulp_boundary_and_o4(upper_air):
    """Simpson endpoints within +/-1 FP32 ULP of 80 km keep one-sided support; O4/lower species vanish above 80 km."""
    _require_metal()
    from sosfuse2.backends import mlx_warm as warm

    h_below = float(np.nextafter(np.float32(80.0), np.float32(-np.inf)))
    h_above = float(np.nextafter(np.float32(80.0), np.float32(np.inf)))

    # 1. Upper segment starting at 80 - 1 ULP going upward to 120 km (both lower=True and lower=False),
    #    plus tangent ray starting at 80 km going outward into h > 80 km.
    g_upper = _ray_group(
        [[R + h_below, 0.0, 0.0], [R + h_below, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 40.0], [0.0, 40.0]],
        [[True], [False]],
        nq=4,
    )
    g_tangent = _ray_group(
        [[R + 80.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0]],
        [[0.0, 50.0]],
        [[True]],
        nq=4,
    )
    # 2. Lower segment starting at 70 km and ending at 80 + 1 ULP (must not be truncated).
    g_lower = _ray_group(
        [[R + 70.0, 0.0, 0.0], [R + 70.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, h_above - 70.0], [0.0, h_above - 70.0]],
        [[True], [False]],
        nq=4,
    )

    c = _extended_constants(top_km=200.0, upper_air=upper_air, W=1, L=16, nq=4, strato=True, aer2=True)
    ns = len(c['h']) * len(c['chi'])
    K = count(16)
    rng = np.random.default_rng(51)
    field = rng.uniform(0.01, 0.05, (ns, 10 + 2 * K, 1)).astype('f4')
    field[:, 10:, :] *= 0.01

    # Exaggerate lower-atmosphere and O4 extinctions so any leakage above 80 km shows up strongly.
    # For upper-only rays with sun overhead (+z), solar paths never dip below 80 km so lower-species sun columns vanish.
    c['strato_sun_col'] = np.where(c['sun_h'][:, None] >= 80.0, np.float32(0.0), c['strato_sun_col'])
    c_heavy = copy.deepcopy(c)
    c_heavy['ext'] = np.array([[0.05], [50.0], [50.0], [50.0], [50.0]], 'f4')
    c_heavy['strato_ext'] = np.array([50.0], 'f4')

    for g in (g_upper, g_tangent, g_lower):
        for first in (True, False):
            exp_q = ref.transport(g, c, field, first)
            (got_q,) = mlx_ops.query([g], [c], field, first)
            _assert_close_rel(got_q, exp_q, wave_axis=1)

            val_ref, y_ref = ref.transport(g, c, field, first, return_basis=True)
            exp_m, _ = ref.project(g, val_ref, c, y_ref)
            got_m, _ = warm.evaluate([g], [c], field, None, first=first)
            _assert_close_rel(got_m[0], exp_m)

    # Both rays have only upper scattering nodes. The tangent's first Simpson
    # midpoint, however, rounds to 80 km and contributes lower extinction.
    for g in (g_upper, g_tangent):
        for first in (True, False):
            (q_base,) = mlx_ops.query([g], [c], field, first)
            (q_heavy,) = mlx_ops.query([g], [c_heavy], field, first)
            m_base, _ = warm.evaluate([g], [c], field, None, first=first)
            m_heavy, _ = warm.evaluate([g], [c_heavy], field, None, first=first)
            if upper_air == 'vacuum':
                np.testing.assert_array_equal(q_base, 0.0)
                np.testing.assert_array_equal(m_base, 0.0)
            else:
                assert float(np.max(np.abs(q_base))) > 0.0
                assert float(np.max(np.abs(m_base))) > 0.0
                if g is g_upper:
                    np.testing.assert_array_equal(q_base, q_heavy)
                    np.testing.assert_array_equal(m_base, m_heavy)
                else:
                    q0, y = ref.transport(g, c, field, first, return_basis=True)
                    q1 = ref.transport(g, c_heavy, field, first)
                    assert (q0 > q1).any()
                    _assert_close_rel(q_base - q_heavy, q0 - q1, wave_axis=1)
                    m1, _ = ref.project(g, q1, c_heavy, y)
                    _assert_close_rel(m_heavy[0], m1)
                    assert (m_base[0] != m_heavy[0]).any()


@pytest.mark.parametrize('upper_air', ['us76_neff', 'vacuum'])
@pytest.mark.parametrize('W,combine', [(1, False), (15, True)])
def test_extended_boundary_cell_no_leakage(upper_air, W, combine):
    """Nodes with h <= 80 and h > 80 sharing cell [80, 90] must not leak hr/ha/hs across 80 km."""
    _require_metal()
    from sosfuse2.backends import mlx_warm as warm

    c = _extended_constants(
        top_km=200.0,
        upper_air=upper_air,
        W=W,
        L=16,
        nq=4,
        heights=[0.0, 40.0, 80.0, 90.0, 200.0],
        strato=True,
        combine=combine,
        skip_thin=combine,
    )
    # Give the 80 km level large densities so any corner leak into h > 80 nodes is unmistakable.
    c['ha'][2] = np.float32(0.5)
    c['strato_density'][2] = np.float32(0.4)
    c['hr'][2] = np.float32(0.3)

    # Mix h <= 80 and h > 80 nodes along the same ray, including an unsplit segment crossing 80 km
    # and a return back to h <= 80 to stress the per-cell register cache (prev_base / previous).
    g = _ray_group(
        [[R + 79.0, 0.0, 0.0], [R + 79.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.99, 0.0, 0.14106735979665885]],
        [[0.0, 1.0, 6.0, 10.0], [0.0, 9.0, 10.0, 10.0]],
        [[True, True, False], [True, True, True]],
        nq=4,
    )

    ns = len(c['h']) * len(c['chi'])
    K = count(16)
    rng = np.random.default_rng(701 + W)

    # Case A: aerosol + stratospheric SH only (Rayleigh zero) -> verifies ha/hs non-leakage in W=1 STRATO_SEP and W=15 combine.
    field_sh = np.zeros((ns, 10 + 2 * K, W), 'f4')
    field_sh[:, 10 : 10 + K, :] = rng.uniform(0.01, 0.05, (ns, K, W)).astype('f4')
    field_sh[:, 10 + K :, :] = rng.uniform(0.01, 0.05, (ns, K, W)).astype('f4')

    # Case B: mixed Rayleigh + SH with near-cancelling Rayleigh trace to exercise INTEGRATE REFINE path.
    field_mix = rng.uniform(0.01, 0.05, (ns, 10 + 2 * K, W)).astype('f4')
    field_mix[:, 0, :] = 0.05
    field_mix[:, 1, :] = -0.05
    field_mix[:, 2, :] = 1e-7
    field_mix[:, 10:, :] *= 0.02

    for field in (field_sh, field_mix):
        exp_q = ref.transport(g, c, field, False)
        (got_q,) = mlx_ops.query([g], [c], field, False)
        _assert_close_rel(got_q, exp_q, wave_axis=1)

        val_ref, y_ref = ref.transport(g, c, field, False, return_basis=True)
        exp_m, _ = ref.project(g, val_ref, c, y_ref)
        got_m, _ = warm.evaluate([g], [c], field, None, first=False)
        _assert_close_rel(got_m[0], exp_m)

    # Explicitly exercise the angular-table (ANG) path in warm._tile on the boundary cell when combined.
    if combine:
        packed = warm.CompactRays([g])
        dc = warm._constants(c)
        raw_src = warm._device(field_mix)
        src = warm._pack_source(raw_src, c['L'], c['W'], c, dc)
        ang_bins = 6 * c['L']
        dc_ang = dict(dc, angular=warm._angular_field(src[2], c, ang_bins), ang_bins=ang_bins)
        got_ang = warm._tile(packed, 0, 1, c, src, None, dc_ang, warm._device([0.0]))
        val_ref, y_ref = ref.transport(g, c, field_mix, False, return_basis=True)
        exp_m, _ = ref.project(g, val_ref, c, y_ref)
        _assert_close_rel(got_ang[0], exp_m)


def test_extended_domain_switch_cache_keys():
    """Consecutive calls switching upper_air (air_end) and top_km (200 <-> 80) must never reuse stale cache entries."""
    _require_metal()
    from sosfuse2.backends import mlx_warm as warm

    # Same ray geometry and wavelength count; ray spans 75..140 km so air_end change alters the signal.
    g = _ray_group(
        [[R + 82.0, 0.0, 0.0], [R + 82.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 30.0, 60.0], [0.0, 20.0, 50.0]],
        [[True, False], [True, True]],
        nq=4,
    )
    g['hits_ground'] = np.array([True, False], bool)
    g['end_index'] = np.array([1, 0], 'i4')
    g['end_fraction'] = np.array([0.25, 0.0], 'f8')
    g['end_columns'] = np.array([[0.2, 0.05, 0.01, 0.005, 0.002], [0.0, 0.0, 0.0, 0.0, 0.0]], 'f8')

    c_neff = _extended_constants(top_km=200.0, upper_air='us76_neff', W=3, L=16, nq=4, strato=True, albedo=0.24)
    c_vac = _extended_constants(top_km=200.0, upper_air='vacuum', W=3, L=16, nq=4, strato=True, albedo=0.24)
    # Also test opt is None branch in _boundary_key with different air_end.
    c_neff_noopt = _extended_constants(top_km=200.0, upper_air='us76_neff', W=3, L=16, nq=4, strato=False, albedo=0.24, with_optics=False)
    c_vac_noopt = _extended_constants(top_km=200.0, upper_air='vacuum', W=3, L=16, nq=4, strato=False, albedo=0.24, with_optics=False)
    assert warm._boundary_key(c_neff_noopt) != warm._boundary_key(c_vac_noopt)

    K = count(16)
    rng = np.random.default_rng(808)
    ns_ext = len(c_neff['h']) * len(c_neff['chi'])
    field_ext = rng.uniform(0.01, 0.05, (ns_ext, 10 + 2 * K, 3)).astype('f4')
    field_ext[:, 10:, :] *= 0.01
    boundary = np.linspace(0.05, 0.25, len(c_neff['chi']) * 3, dtype='f8').reshape(-1, 3)

    packed = warm.CompactRays([g], include_empty=True)
    # 1. Switch upper_air: us76_neff -> vacuum -> us76_neff on the same CompactRays and query plan.
    q_res, m_res = [], []
    for c_cur in (c_neff, c_vac, c_neff):
        for first in (True, False):
            exp_q = ref.transport(g, c_cur, field_ext, first)
            (got_q,) = mlx_ops.query([g], [c_cur], field_ext, first)
            _assert_close_rel(got_q, exp_q, wave_axis=1)
            if not first:
                q_res.append(got_q)

            val_ref, y_ref = ref.transport(g, c_cur, field_ext, first, boundary=boundary, return_basis=True)
            exp_m, exp_f = ref.project(g, val_ref, c_cur, y_ref)
            got_m, got_f = warm.evaluate([g], [c_cur], field_ext, boundary, packed, first=first)
            _assert_close_rel(got_m[0], exp_m)
            _assert_close_rel(got_f[0], exp_f)
            if not first:
                m_res.append(got_m[0])
    # Confirm us76_neff and vacuum actually produced distinct outputs and switching back restored us76_neff.
    assert float(np.max(np.abs(q_res[0]))) > 1e-6 and float(np.max(np.abs(q_res[1]))) == 0.0
    assert float(np.max(np.abs(m_res[0] - m_res[1]))) > 1e-6
    np.testing.assert_allclose(q_res[0], q_res[2], rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(m_res[0], m_res[2], rtol=1e-6, atol=0.0)

    # 2. Switch top_km: 200 km -> 80 km -> 200 km on shared low-altitude ray geometry.
    g_low = _ray_group(
        [[R + 20.0, 0.0, 0.0], [R + 20.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.6, 0.0, 0.8]],
        [[0.0, 20.0, 40.0], [0.0, 15.0, 30.0]],
        [[True, False], [True, True]],
        nq=4,
    )
    c_80 = _extended_constants(top_km=80.0, W=3, L=16, nq=4, strato=True, albedo=0.24)
    ns_80 = len(c_80['h']) * len(c_80['chi'])
    field_80 = rng.uniform(0.01, 0.05, (ns_80, 10 + 2 * K, 3)).astype('f4')
    field_80[:, 10:, :] *= 0.01
    packed_low = warm.CompactRays([g_low])
    for c_cur, f_cur in ((c_neff, field_ext), (c_80, field_80), (c_neff, field_ext)):
        for first in (True, False):
            exp_q = ref.transport(g_low, c_cur, f_cur, first)
            (got_q,) = mlx_ops.query([g_low], [c_cur], f_cur, first)
            _assert_close_rel(got_q, exp_q, wave_axis=1)
            val_ref, y_ref = ref.transport(g_low, c_cur, f_cur, first, return_basis=True)
            exp_m, exp_f = ref.project(g_low, val_ref, c_cur, y_ref)
            got_m, got_f = warm.evaluate([g_low], [c_cur], f_cur, None, packed_low, first=first)
            _assert_close_rel(got_m[0], exp_m)
            _assert_close_rel(got_f[0], exp_f)


@pytest.mark.parametrize('order', [16, 24, 32])
def test_extended_200km_orders_albedo_twilight(order):
    """Exercise L in {16, 24, 32}, albedo > 0, aer2, strato, and deep-twilight -15 deg rays at 200 km."""
    _require_metal()
    from sosfuse2.backends import mlx_warm as warm

    W = 15 if order == 16 else 3
    c = _extended_constants(
        top_km=200.0,
        upper_air='us76_neff',
        W=W,
        L=order,
        nq=6,
        strato=True,
        combine=(W == 15),
        skip_thin=(W == 15),
        albedo=0.24,
        aer2=True,
        ng=20 if order == 16 else 4,
    )
    # Rays starting at 200 km and 40 km at chi = 105 deg (-15 deg solar elevation), crossing Earth shadow and 80 km shell.
    chi = math.radians(105.0)
    up = np.array([math.sin(chi), 0.0, math.cos(chi)])
    ex = np.array([-math.cos(chi), 0.0, math.sin(chi)])
    ey = np.array([0.0, 1.0, 0.0])
    p0 = (R + 40.0) * up
    dirs = []
    for el_deg, az_deg in ((5.0, 0.0), (20.0, 45.0), (-5.0, 15.0)):
        el, az = math.radians(el_deg), math.radians(az_deg)
        dirs.append(math.sin(el) * up + math.cos(el) * (math.cos(az) * ex + math.sin(az) * ey))
    p = np.broadcast_to(p0, (len(dirs), 3)).copy()
    d = np.asarray(dirs, 'f8')
    edges = shells.boundaries(p, d, domain=c['domain'])
    end = edges[:, -1]
    b = np.sum(p * d, -1)
    pp = np.sum(p * p, -1)
    more = []
    for height in c['domain'].extra_shells:
        disc = b * b - pp + (R + height) ** 2
        for sign in (-1, 1):
            root = -b + sign * np.sqrt(np.maximum(disc, 0.0))
            more.append(np.where((disc > 0) & (root > 0) & (root < end), root, end))
    edges = np.sort(np.column_stack([edges, *more]), axis=-1)
    pts = p[:, None, :] + edges[:, :, None] * d[:, None, :]
    rad = np.linalg.norm(pts, axis=2)
    target = basis_at(p)
    g = dict(
        p=p,
        d=d,
        dloc=np.einsum('nij,ni->nj', target, d),
        target=target,
        weight=np.array([0.8, 1.1, 0.6]),
        edges=edges,
        lower=rad[:, :-1] <= rad[:, 1:],
        hits_ground=np.zeros(len(dirs), bool),
        end_index=np.zeros(len(dirs), 'i4'),
        end_fraction=np.zeros(len(dirs), 'f8'),
        end_columns=np.zeros((len(dirs), 5), 'f8'),
        nq=6,
    )
    ns = len(c['h']) * len(c['chi'])
    K = count(order)
    rng = np.random.default_rng(1500 + order)
    field = rng.uniform(0.005, 0.04, (ns, 10 + 2 * K, W)).astype('f4')
    field[:, 10:, :] *= 0.01

    for first in (True, False):
        exp_q = ref.transport(g, c, field, first)
        (got_q,) = mlx_ops.query([g], [c], field, first)
        _assert_close_rel(got_q, exp_q, wave_axis=1)
        val_ref, y_ref = ref.transport(g, c, field, first, return_basis=True)
        exp_m, exp_f = ref.project(g, val_ref, c, y_ref)
        got_m, got_f = warm.evaluate([g], [c], field, None, first=first)
        _assert_close_rel(got_m[0], exp_m)
        _assert_close_rel(got_f[0], exp_f)


def _check_independent_rays(cells, c, field, packed=None):
    from sosfuse2.backends import mlx_warm as warm

    assert all(len(cell['d']) == 1 for cell in cells)
    g = {k: np.concatenate([cell[k] for cell in cells]) for k in cells[0] if k != 'nq'}
    g['nq'] = cells[0]['nq']
    constants = [c] * len(cells)
    for first in (True, False):
        value, y = ref.transport(g, c, field, first, return_basis=True)
        assert np.all(np.max(np.abs(value), axis=2) > 0)
        queries = mlx_ops.query(cells, constants, field, first)
        m, f = warm.evaluate(cells, constants, field, None, packed, first=first)
        if not first:
            _, ground = warm.evaluate(cells, constants, field, None, first=False, ground_only=True)
        for i, cell in enumerate(cells):
            ray = value[i:i + 1]
            context = f"direction={cell['d'][0]}, first={first}"
            _assert_close_rel(queries[i], ray, wave_axis=1, err_msg=context)
            moments, flux = ref.project(cell, ray, c, y[i:i + 1])
            _assert_close_rel(m[i], moments, err_msg=context)
            _assert_close_rel(f[i], flux, err_msg=context)
            if not first:
                _assert_close_rel(ground[i], flux, err_msg=context)


def _rayleigh_field(c):
    field = np.zeros((len(c['h']) * len(c['chi']), 10 + count(c['L']), c['W']), 'f4')
    field[:, :3, :] = np.geomspace(0.03, 3e-5, c['W'])
    return field


@pytest.mark.parametrize('upper_air', ['vacuum', 'us76_neff'])
def test_extended_tangent_nq24(upper_air):
    _require_metal()
    c = _extended_constants(upper_air=upper_air, W=3, nq=24, strato=False, albedo=0.24)
    g = _ray_group([[R + 80, 0, 0]], [[0, 0, 1]], [[0, 100]], [[True]], nq=24)
    for k in (0, 1):
        assert ref.sample(g, 0, k, c)['h'].item() == 80.0
    _check_independent_rays([g], c, _rayleigh_field(c))


_RADIAL_ANGLES = [(0, 0), (14, 0), (37, 0), (73, 0), (0, 37), (14, 37)]


def _radial_ray(height, angles, edges, lower):
    az, el = np.deg2rad(angles)
    radial = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    return _ray_group([(R + height) * radial], [radial], [edges], [lower])


@pytest.mark.parametrize('edge_name', ['air_end', 'lower_end'])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('offsets', [(-1.5, 1.5), (-0.5, 1), (-1.5, 10)],
                         ids=['symmetric', 'midpoint-on-edge', 'asymmetric'])
def test_extended_short_support_interval(offsets, lower, edge_name):
    _require_metal()
    from sosfuse2.backends import mlx_warm as warm

    c = _extended_constants(W=3, nq=4, strato=False, albedo=0.24)
    edge = getattr(c['domain'], edge_name)
    ulp = float(np.spacing(np.float32(edge)))
    left, right = offsets
    width = (right - left) * ulp
    cells = [_radial_ray(edge + left * ulp, angles, [0, width, width + 4 * ulp], [lower, True])
             for angles in _RADIAL_ANGLES]
    table = c['density_optics']['table'].copy()
    strengths = np.array([0.25, 1, 3]) / width
    if edge_name == 'lower_end':
        # A constant lower absorber changes transmission into the second interval.
        table[:, 2] = 1
        c['ext'][3] = strengths
    else:
        # A steep last cell resolves individual ULPs even after clipping to the top.
        table[-2:, 0] = [1, 2**-16]
        c['sun_col'][:] = 0
        c['ext'][0] = strengths / (ulp / c['domain'].step_km)
    c['density_optics'] = dict(c['density_optics'], table=table)
    packed = warm.CompactRays(cells)
    _check_independent_rays(cells, c, _rayleigh_field(c), packed)

    # Check the actual three FP32 inputs to endpoint support, including cases
    # where clipping or a zero extinction would hide a geometry error.
    records = np.asarray(packed.ext_seg_tile(0, len(cells), c['domain'])).reshape(len(cells), 2, 11)
    p, d = (ref.split(np.concatenate([cell[k] for cell in cells])) for k in ('p', 'd'))
    edges = np.concatenate([cell['edges'] for cell in cells])
    for seg, (a, b) in enumerate(zip(edges.T[:-1], edges.T[1:])):
        expected = [ref.path_height(ref.path_point(p, d, ref.split(t))[2], c['domain'])
                    for t in (a, (a + b) / 2, b)]
        np.testing.assert_array_equal(records[:, seg, 8:11], np.column_stack(expected))


def test_extended_air_endpoint():
    _require_metal()
    c = _extended_constants(W=3, nq=4, strato=False, albedo=0.24)
    top = c['domain'].air_end
    ulp = float(np.spacing(np.float32(top)))
    cells = [_radial_ray(top - ulp, angles, [0, ulp], [True]) for angles in _RADIAL_ANGLES]
    # A steep final table cell distinguishes air_end from its lower neighbor.
    tab = c['density_optics']['table'].copy()
    tab[-2:, 0] = [1.0, 1e-6]
    c['density_optics'] = dict(c['density_optics'], table=tab)
    c['sun_col'][:] = 0
    c['ext'][0] = np.array([0.25, 1, 3]) / (ulp * (ulp / c['domain'].step_km))
    _check_independent_rays(cells, c, _rayleigh_field(c))


@pytest.mark.parametrize('invalid', ['missing', 'rows', 'domain'])
def test_extended_invalid_profile_query(invalid):
    _require_metal()
    c = _extended_constants(with_optics=invalid != 'missing', strato=False)
    if invalid == 'rows':
        c['density_optics'] = dict(c['density_optics'], table=c['density_optics']['table'][:-1])
    elif invalid == 'domain':
        c['domain'] = _extended_constants(upper_air='vacuum', with_optics=False)['domain']
    g = _ray_group([[R + 80, 0, 0]], [[1, 0, 0]], [[0, 1]], [[True]])
    for lookup in (lambda: ref.transport(g, c, None, True), lambda: mlx_ops.query([g], [c], None, True)):
        with pytest.raises(RuntimeError, match='E_PROFILE_N_MISMATCH'):
            lookup()


def test_extended_plan_and_segment_domain_switch():
    _require_metal()
    from sosfuse2.backends import mlx_query, mlx_warm as warm

    g = _ray_group([[R + 79.875, 0, 0]], [[1, 0, 0]], [[0, 0.125, 4]], [[True, True]])
    packed = warm.CompactRays([g])
    states = [_extended_constants(step_km=step, upper_air=air, W=3, strato=False, albedo=0.24)
              for step, air in ((0.5, 'us76_neff'), (0.25, 'us76_neff'), (0.25, 'vacuum'))]
    misses = mlx_query._plan.cache_info().misses
    for i, c in enumerate([*states, states[0]]):
        _check_independent_rays([g], c, _rayleigh_field(c), packed)
        assert mlx_query._plan.cache_info().misses - misses == min(i + 1, 3)
        assert len(packed._ext_seg_cache) == min(i + 1, 3)
