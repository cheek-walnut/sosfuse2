"""Per-execution upper boundary: CPU contracts and independent support integrals."""

import copy
import ctypes
from functools import partial
import json
import platform
import shutil
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sosfuse2.foundation import top, paths
from sosfuse2.atmosphere import density
from sosfuse2.backends import numpy_reference as nr
from sosfuse2.jobs import run_state as rs
from sosfuse2.rays import shell_segments as shell, column_table as ct
from sosfuse2.solve import engine, source_solver, ray_geometry


def state(step=.5):
    result = copy.deepcopy(ray_geometry.normal_state())
    result['density'] = dict(density.DEFAULTS, profile_step_km=step)
    return result


def domain(T=120, step=.5):
    return top.resolve(dict(top_km=T, upper_air='vacuum') if T > 80 else {}, step)


def config(tmp_path, raw=None, reduced=False):
    p = tmp_path / 'settings.json'
    p.write_text(json.dumps(raw or {}))
    return rs.settings_for(NS(settings_json=p, reduced=reduced))


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    for name in ('SUN', 'GROUND', 'COLUMN', 'GEOMETRY'):
        monkeypatch.setenv(f'SOSFUSE2_{name}_WORKERS', '1')
    monkeypatch.setenv('SOSFUSE2_HOST_CACHE', '0')
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'exact')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')


def test_settings_and_chi(tmp_path):
    base = config(tmp_path)
    assert not set(base) & {'top_km', 'upper_heights_km', 'upper_air', 'export_first_order'}
    assert config(tmp_path, {'export_first_order': False}) == base
    extended = config(tmp_path, dict(top_km=120, upper_air='vacuum', chi_refine_deg=[[88, 100], [90, 95]]))
    assert extended['nr'] == base['nr'] + 4
    assert extended['upper_heights_km'] == [90, 100, 110, 120]
    ch = np.rad2deg(rs.refined_chi(shell.grids(1)[1], .5, extended['chi_refine_deg']))
    assert np.min(np.diff(ch)) == pytest.approx(.125)
    assert len(ch) == 119
    for raw in ({'top_km': 79}, {'top_km': 201}, {'top_km': 10**500},
                {'top_km': 120, 'upper_heights_km': [10**500, 120]}, {'top_km': True}, {'top_km': float('nan')},
                {'upper_air': 'vacuum'},
                {'top_km': 120, 'upper_air': 'unknown'},
                {'top_km': 121, 'upper_air': 'vacuum'},
                {'top_km': 120, 'upper_air': 'vacuum', 'upper_heights_km': [90, 90, 120]},
                {'top_km': 120, 'upper_air': 'vacuum', 'upper_heights_km': [90, 110]}):
        with pytest.raises(ValueError, match='E_SETTINGS_TOP'):
            config(tmp_path, raw)
    with pytest.raises(ValueError, match='E_SETTINGS_TOP'):
        config(tmp_path, dict(top_km=120, upper_air='vacuum'), reduced=True)
    with pytest.raises(ValueError, match='E_SETTINGS_GRID'):
        config(tmp_path, dict(top_km=120, upper_air='vacuum', nr=20))
    with pytest.raises(ValueError, match='E_SETTINGS_GRID'):
        config(tmp_path, dict(chi_refine_deg=[[88, 100], [90.1, 95]]))
    for bad in ({}, dict(density=density.LEGACY), dict(density=dict(density.DEFAULTS, external_profile={}))):
        with pytest.raises(ValueError, match='E_SETTINGS_TOP'):
            rs.resolve_settings(extended, bad)
    assert rs.execution_domain(extended, state()) == domain()


def test_domains_and_cache_identity(tmp_path):
    from sosfuse2.checks import gate
    from sosfuse2.precompute import endpoint_contract

    keys, columns, suns, endpoints, folders, tools = [], [], [], [], [], []
    for T in (120, 80, 120):
        d = domain(T)
        k = source_solver.Kernel(backend='numpy', L=16, domain=d)
        keys.append(k.kernel_key((), 1, 1, 0, 0, 0))
        assert f'-DTOP_KM={T}.0f' in d.defines()
        ctx = engine.EngineContext(NS(), 1.2, .8, None, False, d)
        assert ctx.Z == T
        optical = density.build(state(), d)
        columns.append(ct.table_key(ct.molecular(optical, 0)))
        suns.append(shell.SunGeometry.identity(1, 1.2, {}, d))
        fake = NS(kernel=k, geo=NS(h=np.array([0., 80., T]), chi=np.array([0., 1.]), cfg={}), fuse_state=state())
        endpoints.append(endpoint_contract.identity(fake))
        folders.append(engine.cell_folder(tmp_path, [0, 80], [0, 1], {}, d))
        tools.append(gate.toolkey(dict(device={}, config=dict(flags=[], stage1=[], async_=False, L=16, block=128,
                                                             profile_n=d.n, domain=d.identity()))))
    for seq in (keys, columns, suns, endpoints, folders, tools):
        assert seq[0] == seq[2] and seq[0] != seq[1]
    a, b = top.Domain(80., .25), domain(160, .5)
    assert a.n == b.n and a.defines() != b.defines()
    assert source_solver.Kernel(domain=a).kernel_key((), 1, 1, 0, 0, 0) != source_solver.Kernel(domain=b).kernel_key((), 1, 1, 0, 0, 0)


@pytest.mark.parametrize('air', ['vacuum', 'us76_neff'])
def test_native_profiles_and_one_sided_simpson(tmp_path, air):
    d = top.resolve(dict(top_km=120, upper_air=air))
    optics = density.build(state(), d)
    h = np.array([np.nextafter(np.float32(80), np.float32(-np.inf)), 80,
                  np.nextafter(np.float32(80), np.float32(np.inf)), 80.25,
                  np.nextafter(np.float32(120), np.float32(-np.inf)), 120,
                  np.nextafter(np.float32(120), np.float32(np.inf))], 'f4')
    c = dict(domain=d, density_optics=optics, height=np.float32(1.2), W=1,
             ext=np.ones((5, 1), 'f4'), strato_ext=np.ones(1, 'f4'),
             strato_profile=dict(center=21., sigma=3., norm=1.))
    c['aer2'] = np.array([1., 1., 1., .7, 1., .1, 2.], 'f4')
    prof = nr.profile_at(h, c)
    start_component = 0 if air == 'vacuum' else 1
    assert np.all(prof[2:, start_component:] == 0) and np.all(nr.strato_profile(h, c)[2:] == 0)
    # A segment outside the support must add exactly zero in either direction.
    panel_values = []
    for p, direction, edges in (([0, 0, shell.R + 80], [0, 0, 1], [0, 40]),
                                ([0, 0, shell.R + 120], [0, 0, -1], [0, 40]),
                                ([0, 0, shell.R + 80], [1, 0, 0], [0, 50])):
        g = dict(p=np.array([p], 'f8'), d=np.array([direction], 'f8'), edges=np.array([edges], 'f8'))
        prefix = nr.SimpsonPrefix(g, c)
        prefix.begin(0)
        prefix.finish()
        if air == 'vacuum':
            assert np.all(prefix.tau == 0), (p, direction, prefix.tau)
        else:
            assert np.all(prefix.tau > 0)
        panel_values.append(float(prefix.tau[0, 0]))
    compiler = shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        pytest.skip('C++ compiler unavailable; NumPy support and Simpson checks above executed')
    src = paths.KERNEL_CU.read_text()
    excerpt = src[src.index('struct ProfilePolicy {'):src.index('__device__ int ix(')]
    excerpt += src[src.index('struct ProfileParams {'):src.index('#define POINT_INLINE')]
    cpp = tmp_path / 'profile.cpp'
    cpp.write_text('#include <cmath>\n#include <algorithm>\n#define __device__\n#define __forceinline__ inline\n'
                   '#define TRACE_MIN 1e-30f\n#define FW 1\n#define NCOMP 5\n#define STRATO\n#define AEROSOL_2L\n'
                   'using std::min;using std::max;\n' + excerpt +
                   'extern "C" float sample(float h,const float*t,int k){return profile_value(h,t,k);}\n'
                   'extern "C" void assembled(float h,const float*t,const float*a,float*out){'
                   'ProfileParams profile{1.2f,a,t,3,t[0],21.f,3.f,1.f};profile_at(h,profile,out,out[5]);}\n'
                   'extern "C" float panel(const double*p,const double*d,double end,const float*t,const float*a){'
                   'float ext[5]={1,1,1,1,1},st[1]={1};DS lo=ds(0.),hi=ds(end),mid=ds(end/2);'
                   'ExtinctionParams optics{{1.2f,a,t,3,t[0],21.f,3.f,1.f},ext,st};'
                   'float b0=beta_at_side(p,d,lo,mid,optics,0);'
                   'float b1=beta_at_side(p,d,hi,mid,optics,0);'
                   'return simpson_piece(p,d,lo,hi,b0,b1,optics,0);}\n')
    lib = tmp_path / 'profile.so'
    subprocess.run([compiler, '-std=c++17', '-ffp-contract=off', '-fPIC',
                    '-dynamiclib' if platform.system() == 'Darwin' else '-shared', *d.defines(),
                    str(cpp), '-o', str(lib)], check=True, capture_output=True)
    native = ctypes.CDLL(str(lib))
    f = native.sample
    f.argtypes, f.restype = [ctypes.c_float, ctypes.c_void_p, ctypes.c_int], ctypes.c_float
    table = np.ascontiguousarray(optics['table'], 'f4')
    got = np.array([[f(z, table.ctypes.data, k) for k in range(3)] for z in h])
    np.testing.assert_allclose(got, density.interpolate32(h, table, d), rtol=1e-6, atol=0)
    assert np.all(got[2:, start_component:] == 0)
    native.assembled.argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    assembled = np.empty((len(h), 6), 'f4')
    for i, z in enumerate(h):
        native.assembled(float(z), table.ctypes.data, c['aer2'].ctypes.data, assembled[i].ctypes.data)
    np.testing.assert_allclose(assembled[:, :5], prof, rtol=1e-5, atol=0)
    np.testing.assert_allclose(assembled[:, 5], nr.strato_profile(h, c), rtol=1e-5, atol=0)
    assert np.all(assembled[2:, start_component:] == 0)
    native.panel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
    native.panel.restype = ctypes.c_float
    for expected, (start, ray, end) in zip(panel_values, (([0, 0, shell.R + 80], [0, 0, 1], 40),
                            ([0, 0, shell.R + 120], [0, 0, -1], 40),
                            ([0, 0, shell.R + 80], [1, 0, 0], 50))):
        p, v = np.array(start, 'f8'), np.array(ray, 'f8')
        actual = native.panel(p.ctypes.data, v.ctypes.data, end, table.ctypes.data, c['aer2'].ctypes.data)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


def test_support_integrals_and_shells(monkeypatch):
    from sosfuse2.rays import ray_columns, incoming_rings
    from sosfuse2.tests.test_profile_step import ref_integral

    d = domain()
    optics = density.build(state(), d)
    old = NS(domain=d, probe=NS(boundaries=partial(shell.boundaries, domain=d), gl=shell.gl))
    p = np.array([[0., 0., shell.R + 79], [0., 0., shell.R + 120], [0., 0., shell.R + 80]])
    rays = np.array([[0., 0., 1], [0., 0., -1], [1., 0., 0.]])
    end = np.array([41., 41., 100.])
    spec = ct.molecular(optics, 0)
    z = np.arange(161) * .5
    f = lambda h: np.where(h > 80, 0, np.interp(h, z, optics['table'][:161, 0]))
    ref = np.array([ref_integral(f, pp, dd, 0, ee, z) for pp, dd, ee in zip(p, rays, end)])
    exact = density.exact_profile_columns(p, rays, end, optics)[:, 0]
    np.testing.assert_allclose(exact, ref, rtol=1e-6, atol=1e-12)
    for method in ('exact', 'table-v1'):
        monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', method)
        got = ct.column(spec, p, rays, end)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-12)
    # Let the GL walk derive its own boundary cuts, including the support sphere.
    _, gl, _ = ray_columns.integrate(p[:1], rays[:1], old, 24, optics=optics)
    np.testing.assert_allclose(gl[:, 0], ref[:1], rtol=1e-3, atol=0)
    assert incoming_rings.RING_HEIGHTS[-1] == 80
    assert d.ring_heights[-2:] == (80, 120)
    for T in (120, 200):
        dom = domain(T)
        bound = shell.boundaries(p, rays, domain=dom)
        cuts = [bound]
        for height in dom.extra_shells:
            dot, pp = np.sum(p*rays, -1), np.sum(p*p, -1)
            disc = dot*dot-pp+(shell.R+height)**2
            cuts.extend([np.clip((-dot+sgn*np.sqrt(np.maximum(disc, 0)))[:, None], 0, bound[:, -1, None]) for sgn in (-1, 1)])
        allcuts = np.sort(np.column_stack(cuts), axis=1)
        assert allcuts.shape[1] - 1 <= 16
        assert np.any(np.isclose(allcuts[0], 1.))


def test_precompile_settings_conflict(tmp_path):
    from sosfuse2.jobs.precompile import resolve_inputs

    p = tmp_path / 'settings.json'
    p.write_text(json.dumps(dict(top_km=120, upper_air='vacuum', L=24, wavelengths=1, spectral_mode='legacy')))
    args = NS(settings_json=p, L=None, W=None, profile_step_km=.25)
    L, W, step, dom = resolve_inputs(args)
    assert (L, W, step, dom.top_km) == (24, 1, .25, 120)
    args.L = 16
    with pytest.raises(ValueError, match='E_PRECOMPILE_CONFLICT: L'):
        resolve_inputs(args)
    p.write_text(json.dumps(dict(top_km=80.125, upper_heights_km=[80.125])))
    args.L = args.profile_step_km = None
    with pytest.raises(ValueError, match='E_SETTINGS_TOP'):
        resolve_inputs(args)
    args.profile_step_km = .125
    assert resolve_inputs(args)[3].top_km == 80.125


def test_worker_bindings_roundtrip(monkeypatch):
    from sosfuse2.precompute import column_parallel as cp, sun_parallel

    d = domain(step=.25)
    old = NS(domain=d, profiles=partial(shell.profiles, engine.ExponentialProfile(1.2, False), domain=d),
             probe=NS(boundaries=partial(shell.boundaries, domain=d), gl=shell.gl))
    frozen = cp.freeze(old, ['profiles', 'probe'])
    restored = cp.thaw(frozen)
    assert restored.domain == d
    p, ray = np.array([[0., 0., shell.R + 80]]), np.array([[0., 0., 1.]])
    assert restored.probe.boundaries(p, ray)[0, -1] == pytest.approx(40)
    assert np.all(restored.profiles(np.array([80.25, 120.])) == 0)
    monkeypatch.setattr(sun_parallel, '_STATE', {})
    sun_parallel._init(frozen, np.array([80., 120.]), np.array([.5, 1.]), shell.R, 24, None, 'gl', cp.SOURCE_HASHES)
    i, value = sun_parallel._run((1, None))
    assert i == 1 and np.isfinite(value).all()
    assert sun_parallel._STATE['old'].domain == d


def small_solver(path, T=120, upper_air='vacuum', backend='numpy', step=.5, wavelengths=None):
    """Small full-domain fixture; no production reduced-grid setting is overridden."""
    d = top.resolve(dict(top_km=T, upper_air=upper_air) if T > 80 else {}, step)

    def grid(mode, h, ch, nmu, nphi, path_n):
        return np.array([0., .1, 8., 40., 80.] + ([120.] if T > 80 else [])), np.deg2rad([0., 85., 95., 105., 180.]), 1, 2, 12

    inputs = state(step)
    if wavelengths is not None:
        inputs['stratosphere'] = dict(status='enabled')
        inputs['parameters']['stratosphere_tau550'] = .01
    s, ctx, _ = engine.create(inputs, path, 'full', 1, domain=d, grid=grid,
                              kernel_factory=lambda *a: source_solver.Kernel(1, a[1], a[2], backend, L=16, domain=d),
                              prepare_ground=False)
    if wavelengths is not None:
        s = ctx.Solver(s.geo, s.kernel, wavelengths, 1, .03)
        s.active[1::2] = 0
    s.levels, s.af = s.levels[:s.kernel.K], s.af[:s.kernel.K]
    s.fuse_state = inputs
    density.attach(s, ctx, defer_ground=wavelengths is not None)
    if wavelengths is not None:
        from sosfuse2.atmosphere import stratosphere

        stratosphere.attach(s, ctx)
    s.geo.pack = lambda p, rays, n: ray_geometry.pack(p, rays, n, s, ctx, ground=False)
    return s, ctx


@pytest.mark.parametrize('fixed_grid', [True, False], ids=['fixed-discretization', 'native-diagnostic'])
def test_small_vacuum_and_first_order(tmp_path, monkeypatch, fixed_grid):
    from sosfuse2.outputs import radiance_table as rt
    from sosfuse2.atmosphere.stratosphere import source_size
    from sosfuse2.rays import incoming_rings

    original_grid, original_incoming = shell.sun_grid, incoming_rings.incoming

    def fixed_sun_grid(scale, top=None):
        h, x = original_grid(scale, 80)
        return (np.r_[h, top] if top is not None and top > 80 else h), x

    def fixed_incoming(*args, **kw):
        kw['domain'] = top.Domain()
        return original_incoming(*args, **kw)

    if fixed_grid:
        monkeypatch.setattr(shell, 'sun_grid', fixed_sun_grid)
        monkeypatch.setattr(incoming_rings, 'incoming', fixed_incoming)

    all_values = []
    for T in (80, 120):
        s, ctx = small_solver(tmp_path / str(T), T)
        geo = ray_geometry.TransientGeometry(s, ctx)
        wrapper = source_solver.Solver(s, geo, prepared=True)
        x = np.zeros((s.geo.ns, source_size(s), s.W), 'f4')
        observed = []
        s.export_first_order = True
        for iteration in range(4):
            table, _ = rt.make_table(s, x, [-3., -6.], [10., 45.], [0., 180.], log=lambda _: None)
            observed.append(table.copy())
            if iteration == 0:
                np.testing.assert_allclose(s.first_order_table, table, rtol=1e-6, atol=0)
                meta = rt.save_first_order(tmp_path / str(T), [s.first_order_table], *map(np.array, ([550.], [-3., -6.], [10., 45.], [0., 180.])))
                with np.load(tmp_path / str(T) / 'first_order.npz') as saved:
                    np.testing.assert_allclose(saved['radiance'], table, rtol=1e-6, atol=0)
                assert rt.first_order_complete(tmp_path / str(T), dict(settings=dict(export_first_order=True), **meta))
                assert not rt.first_order_complete(tmp_path / 'missing', dict(settings=dict(export_first_order=True), **meta))
            if iteration < 3:
                x = wrapper.apply(x)
        all_values.append(np.array(observed))
    rel = np.abs(all_values[1] - all_values[0]) / np.maximum(np.abs(all_values[0]), 1e-30)
    location = np.unravel_index(np.argmax(rel), rel.shape)
    print('VACUUM_MAX_REL', float(rel[location]), 'LOCATION', location, flush=True)
    (tmp_path / 'vacuum.json').write_text(json.dumps(dict(fixed_grid=fixed_grid,
        max_rel=float(rel[location]), location=[int(i) for i in location],
        max_by_iteration=np.max(rel, axis=(1, 2, 3, 4)).tolist())))
    assert np.isfinite(all_values).all()
    if fixed_grid:
        np.testing.assert_allclose(all_values[1], all_values[0], rtol=1e-5, atol=0)


@pytest.mark.parametrize('method', ['table-v1', 'exact', 'gl24'])
@pytest.mark.parametrize('resumed', [False, True])
def test_reused_aerosol_column_matches_fresh_pack(tmp_path, monkeypatch, method, resumed):
    from sosfuse2.atmosphere.tropo_profile import ExponentialProfile, TwoLayerProfile
    from sosfuse2.solve.ray_geometry_cpu import ensure_column

    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', method)
    d = domain()
    starts = np.array([[0., 0., shell.R + 90], [0., 0., shell.R + 79.5]])
    rays = np.array([[1., 0., 0.], [0., 0., 1.]])
    s = NS(kernel=NS(height=3., domain=d), geo=NS(chi=np.array([0., np.pi])))
    old = NS(domain=d, probe=NS(boundaries=partial(shell.boundaries, domain=d), gl=shell.gl))
    scope = NS(b=NS(sh=NS(old=old)), basis_at=lambda p: np.broadcast_to(np.eye(3), (len(p), 3, 3)))

    def fresh(profile):
        old.profiles = partial(shell.profiles, profile, domain=d)
        old.aerosol_descriptor = profile.descriptor()
        return ray_geometry.pack(starts, rays, 1, s, scope)

    common = fresh(ExponentialProfile(1., True))
    target = ExponentialProfile(3., True)
    if resumed:
        target = TwoLayerProfile(np.array([1., 1., 1., 1., 1., .2, 2.]))
        s.aerosol_profile = dict(tag='two_layer', vector=target.vector)
    expected = fresh(target)['end_columns'][:, 1]
    assert expected[0] == 0 and expected[1] > 0
    counts, edges, _ = ray_geometry._csr_vector(common['edges'], common['lower'])
    host = dict(starts=starts, dirs=rays, cell_off=np.arange(3), edges=edges,
                edge_off=np.r_[0, counts.cumsum()], end_col=common['end_columns'].astype('f4'))
    geometry = ray_geometry.TwoLayerGeometry.__new__(ray_geometry.TwoLayerGeometry)
    from sosfuse2.precompute import endpoint_contract

    s.geo.h, s.geo.cfg = np.array([0., 80., 120.]), {}
    with monkeypatch.context() as before:
        before.setattr(endpoint_contract, 'VERSION', 'sosfuse2-geometry-contract-4')
        old_key = endpoint_contract.identity(s)[1]
    key = endpoint_contract.identity(s)[1]
    assert key != old_key
    cache = tmp_path / ('endpoints_' + key)
    cache.mkdir()
    geometry.cache, geometry.s, geometry.bundle, geometry.stats = cache, s, None, {}
    geometry.host_arrays = lambda: (host, [])
    installed = {}
    geometry.install = lambda host, records, columns: installed.update(columns)
    (cache / 'source_height.json').write_text(json.dumps(dict(height=None if resumed else 1.)))
    path = geometry.column_path(3.)
    stale = tmp_path / ('endpoints_' + old_key) / path.relative_to(cache)
    stale.parent.mkdir(parents=True)
    np.ones(2, 'f4').tofile(stale)
    assert ensure_column(geometry) == 'computed'
    assert ensure_column(geometry) == 'cached'
    geometry.prepare()
    np.testing.assert_allclose(installed['end_col'][:, 1], expected, rtol=1e-3, atol=0)
    path.unlink()
    geometry.prepare()
    assert geometry.stats['aerosol_column'] == 'computed'
    np.testing.assert_allclose(installed['end_col'][:, 1], expected, rtol=1e-3, atol=0)
