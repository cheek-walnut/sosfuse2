"""Boundary propagation, independent policies and cache identity on CPU."""

import shutil
import subprocess
import sys
import textwrap

import pytest


def run_with_top(body, *args, top=96.0):
    code = (
        'import sosfuse2.foundation.top as top\n'
        f'domain = top.Domain({top!r}, .5, ({top!r},) if {top!r} > 80 else (), "vacuum" if {top!r} > 80 else None)\n'
        + textwrap.dedent(body)
    )
    result = subprocess.run([sys.executable, '-B', '-c', code, *map(str, args)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_boundary_propagates_to_engine_grids_and_observer(tmp_path):
    run_with_top(
        '''
        import hashlib, json, sys
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        import pytest
        from sosfuse2.atmosphere import components, density
        from sosfuse2.checks import testkit
        from sosfuse2.jobs import run_state
        from sosfuse2.rays import shell_segments, column_table, incoming_quadrature
        from sosfuse2.solve import engine, ray_geometry

        assert (shell_segments.Z, column_table.Z, incoming_quadrature.Z, engine.Z) == (80,) * 4
        assert not hasattr(engine.EngineContext, 'Z')
        assert testkit.sun_scope().Z == 80
        assert shell_segments.grids(1)[0][-1] == shell_segments.small_heights()[-1] == 80
        p = np.array([[0., 0., shell_segments.R]])
        d = np.array([[0., 0., 1.]])
        assert shell_segments.boundaries(p, d, domain=domain)[0, -1] == pytest.approx(96, rel=1e-3)

        work = Path(sys.argv[1])
        settings = work / 'settings.json'
        raw = dict(top_km=96, upper_heights_km=[96], upper_air='vacuum', observer_km=96)
        settings.write_text(json.dumps(raw))
        args = SimpleNamespace(reduced=False, settings_json=settings)
        cfg = run_state.settings_for(args)
        expected = hashlib.sha256(np.ascontiguousarray(np.r_[run_state.refined_heights(shell_segments.grids(1)[0], cfg['h_refine']), 96]).tobytes()).hexdigest()
        assert run_state.grid_metadata(cfg)['h_sha256'] == expected
        settings.write_text(json.dumps(dict(raw, observer_km=96.1)))
        with pytest.raises(ValueError, match='E_SETTINGS_OBSERVER'):
            run_state.settings_for(args)

        # A verified, synthetic extended NO2 loader isolates grid construction from the fixed v1 asset.
        z = np.arange(193) * .5
        state = dict(ray_geometry.normal_state(), density=dict(density.DEFAULTS))
        with patch.object(density, 'no2_asset', return_value=(z, 1 + z / 100, 'synthetic')), components._s1_reference():
            for mode in ('full', 'reduced', 'selftest'):
                solver, ctx, _ = engine.create(state, work / mode, mode, 1, kernel_factory=lambda *a: None,
                                                sun_geometry=lambda *a: None, prepare_ground=False, domain=domain,
                                                grid=lambda mode,h,ch,nmu,nphi,n: (np.r_[h,96],ch,nmu,nphi,n))
                assert solver.geo.h[-1] == 96
                assert ctx.Z == 96
                height = state['parameters']['scale_height_km']
                expected_column = height * (-np.expm1(-80 / height))
                assert ctx.b.sh.NORMAL['aer'] == pytest.approx(state['parameters']['aod550'] / expected_column, rel=1e-3)
        optics = dict(table=np.ones((193, 3)), mask=0, scale=1.0, domain=domain, meta=dict(profile_step_km=0.5))
        assert density.profile_identity(optics)['top_km'] == 96
        ''',
        tmp_path,
    )


def test_profile_grid_derivation_and_invalid_division():
    run_with_top(
        '''
        import numpy as np
        import pytest
        from sosfuse2.atmosphere import density
        for step, n, inv in ((.5, 193, 2), (.25, 385, 4), (.125, 769, 8)):
            assert top.profile_n(step, domain.top_km) == n
            assert density.profile_n(step) == top.profile_n(step, 80)
            assert top.profile_kernel_constants(step, domain.top_km) == (inv, n - 2)
            assert density.KERNEL_LITERALS[density.profile_n(step)] == (inv, density.profile_n(step) - 2)
            assert density.grid(n, step=step, top_km=96)[-1] == 96
            assert len(density.grid(n, step=step, top_km=96)) == n
        for step, boundary in ((.7, 96), (0, 96), (.5, float('nan'))):
            with pytest.raises(ValueError, match='E_PROFILE_GRID'):
                top.profile_n(step, boundary)
        '''
    )


def test_kernel_and_cpu_clip_and_interpolation(tmp_path):
    if not shutil.which('clang++'):
        pytest.skip('clang++ is required to compile the extracted CUDA functions for the host')
    run_with_top(
        r'''
        import ctypes, platform, re, subprocess, sys
        from pathlib import Path
        import numpy as np
        from sosfuse2.atmosphere import density
        from sosfuse2.backends import numpy_reference
        from sosfuse2.backends.arithmetic import split
        from sosfuse2.foundation import paths
        from sosfuse2.tests.test_kernel_static import STUB
        from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA
        from sosfuse2.tests.test_strato_precision import extract_function
        src = paths.KERNEL_CU.read_text()
        clip = extract_function(src, 'clip') + '\n'
        start = src.index('#ifndef PROFILE_N')
        profile = src[start:src.index('\n}\n', src.index('float profile_value(', start)) + 3]
        path = src[src.index('struct RayGeometry {'):src.index('struct PathPoint {')]
        cpp = Path(sys.argv[1]) / 'profile.cpp'
        cpp.write_text('#include <algorithm>\n#include <cmath>\n#define __device__\n#define __forceinline__ inline\n'
                       'using std::min; using std::fminf; using std::fmaxf;\n' + clip + profile + path
                       + 'extern "C" float pv(float h,const float*t,int k){return profile_value(h,t,k);}\n'
                       + 'extern "C" float ph(float h){PathSegment path{0,1,h,6371.f+h,{},true,h>80.f};return path.height(0);}\n')
        stub = Path(sys.argv[1]) / 'cuda_stub.h'
        stub.write_text(STUB + SYNTAX_STUB_EXTRA + 'float __expf(float);float __sinf(float);float __cosf(float);\n')
        link = '-dynamiclib' if platform.system() == 'Darwin' else '-shared'
        for step in (.5, .25, .125):
            domain = top.Domain(96., step, (96.,), 'vacuum')
            subprocess.run(['clang++', '-std=c++17', '-ffp-contract=off', '-fsyntax-only', '-x', 'c++', '-include',
                            str(stub), '-DFW=15', '-DK_FIRST=0', '-DK_PHASE=0', '-DK_BOUNDARY=0',
                            *domain.defines(), str(paths.KERNEL_CU)], check=True)
            library = Path(sys.argv[1]) / f'profile-{step}.so'
            subprocess.run(['clang++', '-std=c++17', '-O1', '-ffp-contract=off', '-fPIC', link,
                            *domain.defines(), str(cpp), '-o', str(library)], check=True)
            native = ctypes.CDLL(str(library))
            native.pv.argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
            native.pv.restype = ctypes.c_float
            native.ph.argtypes = [ctypes.c_float]
            native.ph.restype = ctypes.c_float
            z = density.grid(domain.n, step=step, top_km=96)
            table = np.ascontiguousarray(np.stack([1 + z/100, 2 + z/200, 3 + z/400], axis=1), 'f4')
            samples = np.array([-1, 0, 80, 95.75, 96, 97, np.nextafter(np.float32(96), np.float32(0)),
                                np.nextafter(np.float32(96), np.float32(100))], 'f4')
            got = np.array([[native.pv(h, table.ctypes.data, k) for k in range(3)] for h in samples])
            np.testing.assert_allclose(got, density.interpolate32(samples, table, domain), rtol=1e-3, atol=0)
            np.testing.assert_allclose(got[:, 0], np.where(samples > 80, 0, 1 + np.clip(samples, 0, 96)/100), rtol=1e-3, atol=0)
            assert got[3, 0] == 0 and got[2, 0] > 0
            np.testing.assert_allclose(got[-1, 0], got[4, 0], rtol=1e-3, atol=0)
            actual_h = np.array([native.ph(h) for h in samples])
            expected_h = numpy_reference.path_height(split(6371.0 + samples.astype('f8')), domain)
            np.testing.assert_allclose(actual_h, expected_h, rtol=1e-3, atol=0)
            optics = dict(table=table, mask=3, scale=1., domain=domain, meta=dict(profile_step_km=step))
            np.testing.assert_allclose(density.profiles(samples, optics), got, rtol=1e-3, atol=0)
        ''',
        tmp_path,
    )


def test_ray_columns_rings_and_boundary_margin():
    run_with_top(
        '''
        from unittest.mock import patch
        import numpy as np
        from sosfuse2.atmosphere import density, stratosphere
        from sosfuse2.backends import numpy_reference
        from sosfuse2.backends.arithmetic import split
        from sosfuse2.rays import shell_segments as s, column_table as ct, incoming_rings as ir, incoming_quadrature
        h = np.array([94.99, 95.01])
        p = np.column_stack([np.zeros(2), np.zeros(2), s.R + h])
        d = np.tile([1., 0., 0.], (2, 1))
        c = ct.coordinates(p, d, np.ones(2), top_km=domain.top_km)
        np.testing.assert_allclose(c['a'], np.sqrt(h / 96), rtol=1e-3, atol=0)
        assert c['direct'].tolist() == [False, True]
        assert 'top79' in ct.BOUNDARY and domain.top_km - top.COLUMN_TOP_MARGIN_KM == 95
        z = density.grid(193, step=.5, top_km=96)
        spec = ct.descriptor('usstd76', values=(1 + z/100).tolist(), grid=dict(step_km=.5, top_km=96))
        p = np.array([[0., 0., s.R + 82.]])
        d = np.array([[0., 0., 1.]])
        integral = ct.exact_columns(spec, p, d, np.array([10.]))[0]
        np.testing.assert_allclose(integral, 10 * (1 + 87/100), rtol=1e-3, atol=0)
        table = np.column_stack([1 + z/100, 2 + z/100, 3 + z/100])
        optics = dict(table=table, mask=3, scale=1., domain=domain, meta=dict(profile_step_km=.5))
        p = np.array([[0., 0., s.R + 95.]])
        measured = density.exact_profile_columns(p, d, np.array([3.]), optics, selected=(0,))[0, 0]
        assert measured == 0  # vacuum support replaces extrapolation beyond the lower component end
        p_lower = np.array([[0., 0., s.R + 79.]])
        measured_lower = density.exact_profile_columns(p_lower, d, np.array([3.]), optics, selected=(0,))[0, 0]
        np.testing.assert_allclose(measured_lower, 1 + 79.5/100, rtol=1e-3, atol=0)

        heights = np.array([80., 95.75, 96., 97.])
        r = s.R + heights
        q = np.column_stack([np.zeros(len(r)), np.zeros(len(r)), r])
        sh = np.array([0., 80., 96.])
        sx = np.array([0., 1.])
        values = np.broadcast_to((1 + sh/100)[:, None, None], (3, 2, 5)).copy()
        sun = s.SunGeometry.__new__(s.SunGeometry)
        sun.h, sun.x, sun.col = sh, sx, values
        sun.domain = domain
        expected = 1 + np.clip(heights, 0, 96)/100
        np.testing.assert_allclose(sun.lookup(q, r)[0][:, 0], expected, rtol=1e-3, atol=0)
        np.testing.assert_allclose(stratosphere.sun_lookup(q, r, sun, values[:, :, 0])[0], expected, rtol=1e-3, atol=0)
        ds = split(r)
        packed = dict(h=heights.astype('f4'), r_DS=ds, qz_DS=ds)
        cols, _, _ = numpy_reference.sun_columns(packed, dict(domain=domain, sun_h=sh.astype('f4'), sun_col=values.astype('f4')))
        np.testing.assert_allclose(cols[:, 0], expected, rtol=1e-3, atol=0)
        dirs = np.tile([0., 0., 1.], (len(r), 1))
        _, ah, _, _, _ = incoming_quadrature.spatial_lookup(q, r, dirs, sh, np.array([0., 1.]), top_km=96)
        np.testing.assert_allclose(ah, [0., 15.75/16, 1., 1.], rtol=1e-3, atol=0)

        r, chi, phi = s.R + 20, 2.0, .2
        pp, up, ex = s.frame(np.array(r), np.array(chi))
        ey = np.cross(up, ex)
        levels = [-np.sqrt(max(0, (s.R + v)**2 - s.R**2)) for v in domain.ring_heights]
        assert ir.RING_HEIGHTS[-1] == 80 and domain.ring_heights[-2:] == (80, 96)
        np.testing.assert_allclose(ir.ring_edges(r, chi, phi, domain.ring_heights),
                                   ir.ring_edges(r, chi, phi, domain.ring_heights, (pp, up, ex, ey, levels)),
                                   rtol=1e-3, atol=0)
        with patch.object(ir, 'ring_edges', wraps=ir.ring_edges) as spy:
            ir.incoming(r, chi, 1, 2, domain=domain)
        assert spy.call_count and all(call.args[3][-1] == 96 and call.args[4][-1][-1] == levels[-1]
                                      for call in spy.call_args_list)
        '''
    )


def test_sun_grid_serial_parallel_and_cache_shape(tmp_path):
    run_with_top(
        '''
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from sosfuse2.rays import shell_segments as s
        from sosfuse2.precompute import sun_parallel, host_cache
        from sosfuse2.solve.engine import SunGeometryBinder
        root = Path(sys.argv[1])
        serial, parallel = root / 'serial', root / 'parallel'
        serial.mkdir(); parallel.mkdir()
        parallel_table = parallel / 'host_sun' / float(1.2).hex()
        parallel_table.mkdir(parents=True)
        aerosol = lambda h: np.exp(-h / 1.2)
        descriptor = dict(kind='exp', height=1.2)
        binder = SunGeometryBinder(aerosol, 1.2, descriptor, domain)
        old = SimpleNamespace(R=s.R, Z=domain.top_km, domain=domain, SunGeometry=binder)
        with patch.object(s, 'total_columns', side_effect=lambda aerosol, p, d, n, **kw: np.zeros((len(d), 5))):
            direct = binder(1, serial)
        with patch.object(sun_parallel, 'rows', side_effect=lambda old, h, x, radius, n: np.zeros((len(h), len(x), 5))) as rows:
            sun_parallel.prebuild(old, 1, parallel_table)
        h, x = rows.call_args.args[1:3]
        np.testing.assert_allclose(h, direct.h, rtol=1e-3, atol=0)
        np.testing.assert_allclose(x, direct.x, rtol=1e-3, atol=0)
        assert len(h) == 81 and len(x) == 401
        np.testing.assert_allclose(h[-1], 96, rtol=1e-3, atol=0)
        with patch.object(sun_parallel, 'prebuild', return_value=None):
            cached = host_cache.sun_geometry(old, 1, parallel, 1.2)
        np.testing.assert_allclose(cached.h, h, rtol=1e-3, atol=0)
        np.testing.assert_allclose(cached.x, x, rtol=1e-3, atol=0)
        assert len(s.sun_grid(2)[0]) == 161
        ''',
        tmp_path,
    )


def test_asset_ranges_and_extended_synthetic_inputs():
    run_with_top(
        '''
        from unittest.mock import patch
        import numpy as np
        import pytest
        from sosfuse2.atmosphere import density, ozone_temperature as oz
        z, _, _ = density.no2_asset()
        assert len(z) == 161 and z[-1] == 80 and density.no2_profile(80) >= 0
        with pytest.raises(ValueError, match='E_ASSET_RANGE: no2_afglus_profile.json'):
            density.no2_profile(np.nextafter(80., np.inf))
        with pytest.raises(ValueError, match='E_ASSET_RANGE: no2_afglus_profile.json: asset end 80 km, requested end 96 km'):
            density.no2_profile([0, 96])
        with pytest.raises(ValueError, match='E_SETTINGS_TOP'):
            density.build(dict(density=dict(density.LEGACY)), domain)
        # Load the real asset through its version and digest checks before exercising its range.
        data, za, air, ozone = density.load_asset()
        assert data['version'] == 1 and za[-1] == 80
        with pytest.raises(ValueError, match='E_SETTINGS_TOP'):
            density.build({}, domain)
        supported = density.build(dict(density=dict(density.DEFAULTS)), domain)
        assert np.all(supported['table'][161:] == 0)
        model = oz.ProfileMean.__new__(oz.ProfileMean)
        model.shape, model.q_usstd, model.t = 'usstd76', np.ones(161), np.full(161, 220.)
        for fn, name in ((model.q, 'usstd76_profile.json'), (model.temp, 'temperature_usstd76.json')):
            assert fn(80) > 0
            with pytest.raises(ValueError, match='E_ASSET_RANGE: ' + name):
                fn([80, 96])
        model.shape = 'triangle'
        assert model.q(96) == 0

        # Test-only loader results are finite, ordered, nonnegative and extend to the requested boundary.
        extended = np.arange(193) * .5
        extended_air = np.exp(-extended / 8)
        extended_ozone = 1 + extended / 100
        assert np.all(np.diff(extended) > 0) and np.isfinite(extended_air).all() and np.all(extended_ozone > 0)
        with (patch.object(density, 'no2_asset', return_value=(extended, 1 + extended/100, 'synthetic')),
              patch.object(density, 'load_asset', return_value=(data, extended, extended_air, extended_ozone))):
            optics = density.build(dict(density=dict(density.DEFAULTS)), domain)
            external = density.with_profile({}, z_km=extended[:161], table=np.column_stack([extended_air[:161], extended_ozone[:161]]),
                                            mask=3, normalize_ozone=True, air_surface_number_density_m3=2.5e25)
            injected = density.build(external)
            with pytest.raises(ValueError, match='E_SETTINGS_TOP'):
                density.build(external, domain)
        assert optics['table'].shape == (193, 3)
        assert optics['table'][-1, 2] == 0 and optics['table'][160, 2] > 0
        np.testing.assert_allclose(np.trapezoid(optics['table'][:161, 0], extended[:161]), density.LEGACY_COLUMN, rtol=1e-3, atol=0)
        np.testing.assert_allclose(np.trapezoid(injected['table'][:161, 1], extended[:161]), 15, rtol=1e-3, atol=0)
        assert injected['meta']['normalization'] == 'ozone column scaled to 15 on the injected grid'
        np.testing.assert_allclose(injected['meta']['no2']['H_f_km'], np.trapezoid((1 + extended/100)[:161], extended[:161]), rtol=1e-3, atol=0)
        from sosfuse2.atmosphere.rayleigh import coefficient
        expected_tau = np.trapezoid(extended_air[:161], extended[:161]) * coefficient(550.)
        np.testing.assert_allclose(injected['meta']['rayleigh_tau550'], expected_tau, rtol=1e-3, atol=0)
        '''
    )


def test_cache_sources_and_kernel_keys_separate_grids(tmp_path):
    run_with_top(
        '''
        import json, sys
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from sosfuse2.atmosphere import density, stratosphere
        from sosfuse2.foundation import content_identity, paths
        from sosfuse2.precompute import column_parallel, endpoint_contract, host_cache, sun_parallel
        from sosfuse2.rays import column_table, shell_segments
        from sosfuse2.solve import engine, source_solver

        assert 'foundation/top' in paths.PRODUCTION_MODULES
        assert 'foundation/top.py' in engine.SOURCE_FILES and 'foundation/top.py' in endpoint_contract.UPSTREAM
        path = paths.PKG / 'foundation' / 'top.py'
        z = np.arange(193) * .5
        sun = SimpleNamespace(h=np.array([0., 96.]), x=np.array([0., 1.]), col=np.zeros((2, 2, 5)))
        s = SimpleNamespace(geo=SimpleNamespace(h=np.array([0., 96.]), chi=np.array([0., 1.]), cfg={}, sun=sun),
                            fuse_state={})
        sun_factory = lambda scale, folder: SimpleNamespace(h=np.empty(81), x=np.empty(401))
        old = SimpleNamespace(R=6371., Z=96., SunGeometry=sun_factory)
        root = Path(sys.argv[1])
        reads = Path.read_bytes
        def altered(p):
            raw = reads(p)
            return raw.replace(b'TOP_KM = 80.0', b'TOP_KM = 96.0', 1) if p == path else raw
        def keys():
            column_table.source_hash.cache_clear()
            with patch.object(sun_parallel, 'prebuild', return_value=None):
                folders = []
                def sun(scale, folder):
                    folders.append(str(folder)); return sun_factory(scale, folder)
                old.SunGeometry = sun
                host_cache.sun_geometry(old, 1, root / 'local', 1.2, shared=root / 'shared')
            return dict(column=column_table.source_hash(),
                        sun=json.loads(shell_segments.SunGeometry.identity(1, 1.2, {'kind':'exp'}))['sources']['top.py'],
                        endpoint=endpoint_contract.identity(s)[1], worker=column_parallel.source_hashes()['foundation/top.py'],
                        strato=stratosphere._geometry_key(s, None, {'center': 21., 'sigma': 3., 'norm': 1.}),
                        package=content_identity.package_hashes()['sosfuse2/foundation/top.py'],
                        engine=engine.fingerprint()['sosfuse2/foundation/top.py'], folder=folders[0])
        with patch.object(density, 'no2_asset', return_value=(z, 1 + z/100, 'synthetic')):
            before = keys()
            expected = column_parallel.source_hashes()
            with patch.object(Path, 'read_bytes', altered):
                after = keys()
                assert column_parallel.source_drift(expected) == 'foundation/top.py'
            assert all(before[name] != after[name] for name in before), (before, after)

        class Raw:
            def __init__(self, code, symbol, options, **kw): self.options, self.attributes = options, {}
            def compile(self): pass
        runtime = SimpleNamespace(getDevice=lambda: 0, getDeviceProperties=lambda _: dict(sharedMemPerBlock=1 << 30))
        kernel = source_solver.Kernel(backend='numpy', L=16, profile_step_km=.25, domain=top.Domain(96., .25, (96.,), 'vacuum'))
        kernel.xp = SimpleNamespace(RawKernel=Raw, cuda=SimpleNamespace(runtime=runtime))
        kernel.device = lambda: ('capture', {'stage1_defaults': ()}, 'test')
        raw = kernel.raw_kernel(15)
        assert all(option in raw.options for option in kernel.domain.defines())
        assert ('TOP_KM', 96.) in next(iter(kernel.raw))
        assert ('PROFILE_STEP', .25) in next(iter(kernel.raw))
        assert '_top' in kernel.attribute_suffix() and '_step' in kernel.attribute_suffix()
        '''
        , tmp_path,
    )
    key_code = '''
        import json
        from sosfuse2.solve.source_solver import Kernel
        k = Kernel(backend='numpy', L=16, profile_step_km=STEP,
                   domain=top.Domain(domain.top_km, STEP, domain.upper_heights_km, domain.upper_air))
        print(json.dumps([k.profile_n, k.kernel_key((), 15, 15, 0, 0, 0), k.attribute_suffix()]))
    '''
    first = run_with_top(key_code.replace('STEP', '.25'), top=80.0)
    second = run_with_top(key_code.replace('STEP', '.5'), top=160.0)
    import json

    a, b = json.loads(first), json.loads(second)
    assert a[0] == b[0] == 321 and a[1] != b[1] and a[2] != b[2]


def test_historical_and_independent_policies():
    run_with_top(
        '''
        import numpy as np
        import pytest
        from sosfuse2.atmosphere import density, ozone_temperature, stratosphere, tropo_profile
        from sosfuse2.checks import gate
        assert (top.NORMALIZATION_TOP_KM, density.ASSET_TOP_KM, density.ASSET_N, top.SUN_HEIGHT_INTERVALS) == (80, 80, 161, 80)
        assert gate.LEGACY_PROFILE_N == 161
        assert ozone_temperature.Z[-1] == 80
        assert tropo_profile.NORMALIZATION_TOP_KM == stratosphere.NORMALIZATION_TOP_KM == 80
        assert density.LEGACY_COLUMN == pytest.approx(8 * (-np.expm1(-80 / 8)), rel=1e-3)
        with pytest.raises(ValueError, match='E_STRATO_PROFILE_UNSUPPORTED'):
            stratosphere.profile_config(dict(stratosphere=dict(center_km=81)))
        meta = dict(device={}, config=dict(flags=[], stage1=[], async_=False, L=16, block=128))
        assert gate.toolkey(meta) == gate.toolkey(dict(meta, config=dict(meta['config'], profile_n=161)))
        assert gate.toolkey(meta)[0] != gate.toolkey(dict(meta, config=dict(meta['config'], profile_n=193)))[0]
        F = gate.load_fixtures()
        want = gate.expected_config(F, 'FX-C3')
        assert want['profile_n'] == 161
        with pytest.raises(gate.GateError, match='profile_n requested 161 measured 193'):
            gate.check_config(F, 'FX-C3', dict(backend='numpy', config=dict(want, profile_n=193)))
        '''
    )
