"""Exercise packed query order, source updates and partial-ray tiles."""

import math
import numpy as np
import pytest

from sosfuse2.backends import mlx_ops
from sosfuse2.backends import numpy_reference as ref
from sosfuse2.tests.test_mlx_trace_shadow import single_ray


def test_query_tiles_and_orders(monkeypatch):
    pytest.importorskip('mlx.core')
    a, c, field = single_ray([6371.01, 0.0, -3.0], 19.0, [0.0, 80.0])
    b, _, _ = single_ray([6370.99, 4.0, -6.0], 9.0, [0.0, 80.0])
    groups = []
    constants = []
    for nq in (1, 5):
        g = {key: np.concatenate([b[key], a[key], a[key]]) for key in a if key != 'nq'}
        g.update(
            nq=nq,
            weight=np.array([0.25, 3.0, 0.0]),
            edges=np.array([[0.0, 1.0, 9.0], [0.0, 5.0, 19.0], [0.0, 0.0, 0.0]]),
            lower=np.array([[True, False], [False, True], [True, True]]),
        )
        u, wu = np.polynomial.legendre.leggauss(nq)
        groups.append(g)
        constants.append(dict(c, u=(u + 1) / 2, wu=wu / 2, ext=np.full((5, 1), 0.01, 'f4')))
    empty = {k: v[:0] if isinstance(v, np.ndarray) else v for k, v in groups[0].items()}
    groups.insert(0, empty)
    constants.insert(0, constants[0])
    rng = np.random.default_rng(42)
    field[:] = rng.uniform(0.01, 0.2, field.shape)
    field[:, 10:] *= 0.0002
    monkeypatch.setenv('SOSFUSE2_MLX_TILE_SAMPLES', '2')
    for first, scale in ((False, 1.0), (True, 2.0), (False, 3.0)):
        current = field * np.float32(scale)
        actual = mlx_ops.query(groups, constants, current, first)
        assert [v.shape for v in actual] == [(0, 1, 4), (3, 1, 4), (3, 1, 4)]
        for g, cc, got in zip(groups[1:], constants[1:], actual[1:]):
            expected = ref.transport(g, cc, current, first)
            assert np.max(abs(got - expected)) <= 1e-3 * np.max(abs(expected))
            np.testing.assert_array_equal(got[2], 0.0)
    assert mlx_ops.query(groups, [dict(constants[0], tropo_lut=np.ones(4, 'f4')), *constants[1:]], field, False) is None
    pos = [dict(constants[0], albedo=0.4), *constants[1:]]
    actual = mlx_ops.query(groups, pos, field, False)
    for g, cc, got in zip(groups[1:], pos[1:], actual[1:]):
        expected = ref.transport(g, cc, field, False)
        assert np.max(abs(got - expected)) <= 1e-3 * np.max(abs(expected))


def _strato_case(W=1, L=16, *, combine=False, skip_thin=False, albedo=0.0, ng=0,
                 center=21.0, sigma=3.0, tau=0.03, chi_scale=1.0):
    from sosfuse2.atmosphere.stratosphere import cell_density
    from sosfuse2.backends.sh_layout import count
    from sosfuse2.foundation.top import Domain
    from sosfuse2.solve.engine_core import basis_at

    # Two rays from the same cell origin so both query() and cell evaluate() share one frame.
    p = np.array([[6371.0 + 18.0, 0.0, 100.0], [6371.0 + 18.0, 0.0, 100.0]], dtype='f8')
    d = np.array([[1.0, 0.0, 0.0], [0.6, 0.5, -math.sqrt(1.0 - 0.6**2 - 0.5**2)]], dtype='f8')
    target = basis_at(p)
    g = dict(
        p=p,
        d=d,
        dloc=np.einsum('nij,ni->nj', target, d),
        target=target,
        weight=np.array([1.0, 0.75]),
        edges=np.array([[0.0, 12.0, 25.0], [0.0, 10.0, 20.0]]),
        lower=np.array([[True, False], [False, True]]),
        hits_ground=np.array([False, False]),
        end_index=np.array([0, 0], 'i4'),
        end_fraction=np.array([0.0, 0.0]),
        end_columns=np.zeros((2, 5)),
        nq=4,
    )
    _, c_base, _ = single_ray([6371.0 + 18.0, 0.0, 100.0], 25.0, [0.0, 10.0, 20.0, 30.0, 80.0])
    u, wu = np.polynomial.legendre.leggauss(4)
    h = np.array([0.0, 10.0, 20.0, 30.0, 80.0], 'f4')
    chi = np.array([0.0, np.pi / 2, np.pi], 'f4')
    ns = len(h) * len(chi)
    K = count(L)
    z = np.linspace(0.0, 80.0, 8001)
    norm = float(np.trapezoid(np.exp(-0.5 * ((z - center) / sigma) ** 2), z))
    profile = dict(center=center, sigma=sigma, norm=norm)
    sden, _ = cell_density(h.astype('f8'), profile)
    rng = np.random.default_rng(700 + W + ng)
    g_st = np.clip(0.65 * chi_scale, 0.0, 0.76)
    l_idx = np.arange(33, dtype='f4')[:, None]
    st_chi = ((2.0 * l_idx + 1.0) * (g_st ** l_idx) * np.linspace(1.0, 0.92, W, dtype='f4')[None, :]).astype('f4')
    st_chi[0] = 1.0
    st_ext = np.linspace(tau, tau * 0.6, W, dtype='f4')
    st_sca = st_ext * np.float32(0.98)
    sun_h = np.array([0.0, 20.0, 80.0], 'f4')
    sun_col = np.zeros((3, 4, 5), 'f4')
    sun_col[:, :, 0] = np.linspace(1.0, 0.1, 3, dtype='f4')[:, None] * np.linspace(4.0, 0.5, 4, dtype='f4')[None, :]
    st_sun_col = (np.linspace(1.0, 0.02, 3, dtype='f4')[:, None] * np.linspace(5.0, 0.8, 4, dtype='f4')[None, :]).astype('f4')
    if ng > 0:
        dirs = rng.normal(size=(ns, ng, 3)).astype('f4')
        dirs[:, :, 1] = np.abs(dirs[:, :, 1])
        dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
        gv = rng.uniform(0.0, 0.04, (ns, ng, W)).astype('f4')
        gm = rng.uniform(0.0, 0.03, (ns, 4, W)).astype('f4')
    else:
        dirs = np.zeros((ns, 0, 3), 'f4')
        gv = np.zeros((ns, 0, W), 'f4')
        gm = np.zeros((ns, 4, W), 'f4')
    c = dict(
        c_base,
        domain=Domain(),
        L=L,
        W=W,
        u=(u + 1) / 2,
        wu=wu / 2,
        h=h,
        chi=chi,
        hr=np.exp(-h / 8.0).astype('f4'),
        ha=np.exp(-h / 1.2).astype('f4'),
        albedo=float(albedo),
        g=np.float32(0.65),
        ext=np.broadcast_to(np.array([0.04, 0.08, 0.005, 0.001, 0.0005], 'f4')[:, None], (5, W)).copy(),
        asca=np.linspace(0.05, 0.09, W, dtype='f4'),
        sun_h=sun_h,
        sun_col=sun_col,
        gm=gm,
        gd=dirs,
        gv=gv,
        strato_ext=st_ext,
        strato_sca=st_sca,
        strato_chi=st_chi,
        strato_density=sden.astype('f4'),
        strato_sun_col=st_sun_col,
        strato_profile=profile,
    )
    if combine:
        c.update(sh_combine=True, sh_combine_skip=skip_thin)
    field = np.zeros((ns, 10 + 2 * K, W), 'f4')
    return g, c, field


def test_strato_query_and_unsupported_fallbacks():
    pytest.importorskip('mlx.core')
    from types import SimpleNamespace as NS
    from sosfuse2.backends.sh_layout import count
    from sosfuse2.foundation.top import Domain

    g, c, field = _strato_case(W=15, L=16, combine=True, skip_thin=True, albedo=0.24, ng=20)
    rng = np.random.default_rng(123)
    K = count(16)
    field[:, :3, :] = rng.uniform(0.01, 0.05, (field.shape[0], 3, 15)).astype('f4')
    field[:, 10 : 10 + K, :] = rng.normal(0.0, 0.002, (field.shape[0], K, 15)).astype('f4')
    field[:, 10 + K :, :] = rng.normal(0.0, 0.005, (field.shape[0], K, 15)).astype('f4')

    for first in (True, False):
        expected = ref.transport(g, c, field, first)
        (got,) = mlx_ops.query([g], [c], field, first)
        scale = np.abs(expected).max(axis=(0, 2))
        assert np.all(scale > 0)
        assert np.all(np.abs(got - expected).max(axis=(0, 2)) <= 1e-3 * scale)

    # Supported 200 km extended-domain STRATO input must succeed and match NumPy reference;
    # unsupported tropo_lut must still return None for both evaluate and query.
    def fake_geo_and_s(cc, gg=g):
        s = NS(
            geo=NS(chi=cc['chi'], ns=len(cc['h']) * len(cc['chi'])),
            kernel=NS(K=K, setup=lambda _s, _nq: cc),
            W=cc['W'],
        )
        geo = NS(cell=lambda _i: gg)
        return geo, s

    from sosfuse2.atmosphere import density as density_mod
    from sosfuse2.atmosphere.stratosphere import cell_density as strato_cell_density
    from sosfuse2.foundation import top as top_mod
    from sosfuse2.solve.ray_geometry import normal_state

    ext_dom = top_mod.resolve(dict(top_km=200.0, upper_heights_km=[120.0, 200.0], upper_air='us76_neff'))
    ext_state = dict(normal_state(), density=dict(density_mod.DEFAULTS, profile_step_km=0.5))
    ext_opt = density_mod.build(ext_state, ext_dom)
    h_ext = np.array([0.0, 10.0, 20.0, 30.0, 80.0, 120.0, 200.0], 'f4')
    ns_ext = len(h_ext) * len(c['chi'])
    sden_ext, _ = strato_cell_density(h_ext.astype('f8'), c['strato_profile'])
    sden_ext = np.where(h_ext > 80.0, 0.0, sden_ext).astype('f4')
    hr_ext = density_mod.interpolate32(h_ext, ext_opt['table'], ext_dom)[:, 0].astype('f4')
    ha_ext = np.where(h_ext > 80.0, 0.0, np.exp(-h_ext / 1.2)).astype('f4')
    sun_h_ext = np.array([0.0, 20.0, 80.0, 200.0], 'f4')
    sun_col_ext = np.zeros((4, 4, 5), 'f4')
    sun_col_ext[:, :, 0] = np.linspace(1.0, 0.01, 4, dtype='f4')[:, None] * np.linspace(4.0, 0.5, 4, dtype='f4')[None, :]
    st_sun_col_ext = (np.linspace(1.0, 0.0, 4, dtype='f4')[:, None] * np.linspace(5.0, 0.8, 4, dtype='f4')[None, :]).astype('f4')
    c_ext = dict(
        c,
        domain=ext_dom,
        density_optics=ext_opt,
        h=h_ext,
        hr=hr_ext,
        ha=ha_ext,
        strato_density=sden_ext,
        sun_h=sun_h_ext,
        sun_col=sun_col_ext,
        strato_sun_col=st_sun_col_ext,
        gm=np.zeros((ns_ext, 4, c['W']), 'f4'),
        gd=np.zeros((ns_ext, 0, 3), 'f4'),
        gv=np.zeros((ns_ext, 0, c['W']), 'f4'),
    )
    field_ext = np.zeros((ns_ext, 10 + 2 * K, c['W']), 'f4')
    field_ext[:, :3, :] = rng.uniform(0.01, 0.05, (ns_ext, 3, c['W'])).astype('f4')
    field_ext[:, 10 : 10 + K, :] = rng.normal(0.0, 0.002, (ns_ext, K, c['W'])).astype('f4')
    field_ext[:, 10 + K :, :] = rng.normal(0.0, 0.005, (ns_ext, K, c['W'])).astype('f4')
    for first in (True, False):
        expected_ext = ref.transport(g, c_ext, field_ext, first)
        got_ext_list = mlx_ops.query([g], [c_ext], field_ext, first)
        assert got_ext_list is not None
        (got_ext,) = got_ext_list
        scale_ext = np.abs(expected_ext).max(axis=(0, 2))
        assert np.all(scale_ext > 0)
        assert np.all(np.abs(got_ext - expected_ext).max(axis=(0, 2)) <= 1e-3 * scale_ext)
        geo_ext, s_ext = fake_geo_and_s(c_ext)
        got_m, _ = mlx_ops.evaluate(geo_ext, s_ext, field_ext, first, 0, False, None)
        assert got_m is not None
        val_ref, y_ref = ref.transport(g, c_ext, field_ext, first, return_basis=True)
        exp_m, _ = ref.project(g, val_ref, c_ext, y_ref)
        scale_m = np.abs(exp_m).max(axis=0)
        assert np.all(np.abs(got_m[0] - exp_m).max(axis=0) <= 1e-3 * scale_m)

    bad = dict(c, tropo_lut=np.ones((5, c['W']), 'f4'), tropo_bins=4)
    geo, s = fake_geo_and_s(bad)
    assert mlx_ops.query([g], [bad], field, False) is None
    assert mlx_ops.query([g], [bad], field, True) is None
    assert mlx_ops.evaluate(geo, s, field, False, 0, False, None) is None
    assert mlx_ops.evaluate(geo, s, field, True, 0, False, None) is None


@pytest.mark.parametrize('order', [16, 24])
def test_strato_w1_second_sh_plane_only(order):
    pytest.importorskip('mlx.core')
    from sosfuse2.backends import mlx_warm as warm
    from sosfuse2.backends.sh_layout import count

    # W=1 has sh_combine=False and reads the second SH plane directly.
    g, c, field = _strato_case(W=1, L=order, combine=False, albedo=0.0)
    assert not c.get('sh_combine', False)
    K = count(order)
    rng = np.random.default_rng(909 + order)
    # Zero Rayleigh and zero tropospheric SH plane; ONLY the second (stratospheric) SH plane is nonzero.
    field[:, : 10 + K, :] = 0.0
    field[:, 10 + K :, :] = rng.uniform(0.01, 0.05, (field.shape[0], K, 1)).astype('f4')

    expected_q = ref.transport(g, c, field, False)
    assert np.max(np.abs(expected_q)) > 0.0
    (got_q,) = mlx_ops.query([g], [c], field, False)
    assert np.max(np.abs(got_q - expected_q)) <= 1e-3 * np.max(np.abs(expected_q))

    val_ref, y = ref.transport(g, c, field, False, return_basis=True)
    expected_m, _ = ref.project(g, val_ref, c, y)
    assert np.max(np.abs(expected_m)) > 0.0
    got_m, _ = warm.evaluate([g], [c], field, None, first=False)
    assert np.max(np.abs(got_m[0] - expected_m)) <= 1e-3 * np.max(np.abs(expected_m))


def test_strato_cache_keys_single_dependency():
    pytest.importorskip('mlx.core')
    from sosfuse2.backends import mlx_warm as warm
    from sosfuse2.backends.sh_layout import count

    # Keep geometry and wavelengths fixed while varying ONE stratospheric dependency at a time.
    g0, c0, _ = _strato_case(W=15, L=16, combine=True, skip_thin=True, albedo=0.24, ng=20,
                             center=21.0, sigma=3.0, tau=0.03, chi_scale=1.0)
    K = count(16)
    rng = np.random.default_rng(404)
    field = rng.uniform(0.005, 0.03, (len(c0['h']) * len(c0['chi']), 10 + 2 * K, 15)).astype('f4')
    field[:, 10:, :] *= 0.01

    variants = [
        c0,
        _strato_case(W=15, L=16, combine=True, skip_thin=True, albedo=0.24, ng=20, center=21.0, sigma=3.0, tau=0.09, chi_scale=1.0)[1],
        _strato_case(W=15, L=16, combine=True, skip_thin=True, albedo=0.24, ng=20, center=25.0, sigma=3.0, tau=0.03, chi_scale=1.0)[1],
        _strato_case(W=15, L=16, combine=True, skip_thin=True, albedo=0.24, ng=20, center=21.0, sigma=5.0, tau=0.03, chi_scale=1.0)[1],
        _strato_case(W=15, L=16, combine=True, skip_thin=True, albedo=0.24, ng=20, center=21.0, sigma=3.0, tau=0.03, chi_scale=0.6)[1],
    ]
    # Keep identical gd/gv/gm across variants so only the single stratospheric parameter changes.
    for v in variants[1:]:
        v['gd'] = c0['gd']
        v['gv'] = c0['gv']
        v['gm'] = c0['gm']

    for v in variants:
        for first in (True, False):
            exp_q = ref.transport(g0, v, field, first)
            (got_q,) = mlx_ops.query([g0], [v], field, first)
            sc = np.abs(exp_q).max(axis=(0, 2))
            assert np.all(np.abs(got_q - exp_q).max(axis=(0, 2)) <= 1e-3 * sc)

            val_ref, y = ref.transport(g0, v, field, first, return_basis=True)
            exp_m, exp_f = ref.project(g0, val_ref, v, y)
            got_m, got_f = warm.evaluate([g0], [v], field, None, first=first)
            scm = np.abs(exp_m).max(axis=0)
            assert np.all(np.abs(got_m[0] - exp_m).max(axis=0) <= 1e-3 * scm)
            scf = np.abs(exp_f)
            if np.any(scf > 0):
                assert np.all(np.abs(got_f[0] - exp_f) <= 1e-3 * np.maximum(scf, 1e-30))


@pytest.mark.parametrize('ng', [4, 24])
def test_query_first_with_ground_and_shadow(ng):
    pytest.importorskip('mlx.core')
    # Shadowed ray plus sunlit ray with nonzero gm and ground angular pool (both direct CSR and HG table).
    g_sh, c, src = single_ray([6370.0, 0.0, -300.0], 20.0, [0.0, 80.0])
    g_sun, _, _ = single_ray([6371.5, 2.0, 100.0], 15.0, [0.0, 80.0])
    g = {key: np.concatenate([g_sh[key], g_sun[key]]) for key in g_sh if key != 'nq'}
    g['nq'] = 4
    u, wu = np.polynomial.legendre.leggauss(4)
    W = 3
    ns = len(c['h']) * len(c['chi'])
    rng = np.random.default_rng(100 + ng)
    dirs = rng.normal(size=(ns, ng, 3)).astype('f4')
    dirs[:, :, 1] = np.abs(dirs[:, :, 1])
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    gv = rng.uniform(0.0, 0.05, (ns, ng, W)).astype('f4')
    gm = rng.uniform(0.0, 0.04, (ns, 4, W)).astype('f4')
    gm[:, 3, :] *= 0.2
    c.update(
        W=W,
        u=(u + 1) / 2,
        wu=wu / 2,
        albedo=0.24,
        g=np.float32(0.68),
        ext=np.full((5, W), 0.015, 'f4'),
        asca=np.linspace(0.05, 0.15, W, dtype='f4'),
        gd=dirs,
        gv=gv,
        gm=gm,
    )
    src = np.zeros((ns, 163, W), 'f4')
    for scale, new_g in ((1.0, 0.68), (1.5, 0.68), (1.5, 0.80)):
        cc = dict(c, albedo=0.24 * scale, g=np.float32(new_g), gv=(gv * np.float32(scale)), gm=(gm * np.float32(scale)))
        expected = ref.transport(g, cc, src, True)
        (got,) = mlx_ops.query([g], [cc], src, True)
        scale_ref = np.abs(expected).max(axis=(0, 2))
        assert np.all(np.abs(got - expected).max(axis=(0, 2)) <= 1e-3 * scale_ref)
        # Ensure shadowed ray itself receives nonzero ground contribution.
        assert np.max(np.abs(got[0])) > 0.0
        assert np.max(np.abs(got[0] - expected[0])) <= 1e-3 * np.max(np.abs(expected[0]))


@pytest.mark.parametrize('delta', [-1e-7, 1e-7])
def test_query_shadow_crossing(delta):
    pytest.importorskip('mlx.core')
    g, c, src = single_ray([6371.0 - (5.0 + delta), 0.0, -300.0], 20.0, [0.0, 80.0])
    expected = ref.transport(g, c, src, True)
    (actual,) = mlx_ops.query([g], [c], src, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=0.0)


@pytest.mark.parametrize('order', [24, 32])
def test_query_rebuilds_changed_optics(order):
    pytest.importorskip('mlx.core')
    from sosfuse2.atmosphere.tropo_profile import profile32
    from sosfuse2.backends.sh_layout import count
    from sosfuse2.rays import shell_segments
    from sosfuse2.solve.engine_core import basis_at

    nr, waves, nq = 7, 3, 4
    elev = np.deg2rad(np.linspace(-1.0, 60.0, nr))
    az = np.deg2rad(np.linspace(1.0, 137.0, nr))
    start, up, ex = shell_segments.frame(np.full(nr, 6371.2), np.deg2rad(np.linspace(88.0, 107.0, nr)))
    ray = np.sin(elev)[:, None] * up + np.cos(elev)[:, None] * (
        np.cos(az)[:, None] * ex + np.sin(az)[:, None] * np.cross(up, ex)
    )
    target = basis_at(start)
    edges = shell_segments.boundaries(start, ray)
    points = start[:, None, :] + edges[:, :, None] * ray[:, None, :]
    rad = np.linalg.norm(points, axis=2)
    g = {
        'p': start,
        'd': ray,
        'target': target,
        'dloc': np.einsum('nij,ni->nj', target, ray),
        'edges': edges,
        'lower': rad[:, :-1] <= rad[:, 1:],
        'nq': nq,
    }
    _, c, _ = single_ray([6371.0, 0.0, 0.0], 5.0, [0.0, 1.0, 10.0, 80.0])
    h = c['h']
    chi = np.deg2rad([0.0, 86.0, 93.0, 104.0, 180.0]).astype('f4')
    u, wu = shell_segments.gl(nq)
    cells = len(h) * len(chi)
    c.update(
        L=order,
        W=waves,
        u=u,
        wu=wu,
        chi=chi,
        gm=np.zeros((cells, 4, waves), 'f4'),
        gd=np.empty((cells, 0, 3), 'f4'),
        gv=np.empty((cells, 0, waves), 'f4'),
        ext=np.full((5, waves), 0.011, 'f4'),
        asca=np.ones(waves, 'f4'),
    )
    rng = np.random.default_rng(order)
    field = rng.normal(0.0, 0.001, (cells, 10 + count(order), waves)).astype('f4')
    field[:, :3] += np.float32(0.04)
    field *= np.geomspace(1e-37, 1.0, cells).astype('f4')[:, None, None]
    for _ in range(2):
        expected = ref.transport(g, c, field)
        (got,) = mlx_ops.query([g], [c], field, False)
        scale = np.abs(expected).max(axis=(0, 2))
        assert np.all(np.abs(got - expected).max(axis=(0, 2)) <= 1e-3 * scale)
        c['ext'][:, 1:] *= np.float32(2.5)
        c['aer2'] = np.array([1.0, 0.7, 0.9, 1.0, 0.2, 0.35, 1.0], 'f4')
        c['ha'] = profile32(h, c['aer2'])
        field[:, :, 0] *= np.float32(0.75)
