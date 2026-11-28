"""Exercise tile boundaries and compare a nonzero incident field with the reference."""

from collections import defaultdict
from time import perf_counter
from types import SimpleNamespace as NS
import warnings

import numpy as np
import pytest

from sosfuse2.backends import numpy_reference as ref
from sosfuse2.backends.sh_layout import count
from sosfuse2.foundation.top import Domain
from sosfuse2.rays import incoming_rings, shell_segments as shells
from sosfuse2.solve.engine_core import basis_at
from sosfuse2.solve.ray_geometry import pack, subset


def _case(nmu=12, nphi=16, nq=12):
    h = shells.small_heights()
    chi = np.deg2rad([0, 60, 85, 90, 93, 96, 105, 110, 140, 180])
    domain = Domain()
    old = NS(probe=shells, domain=domain)
    scope = NS(b=NS(sh=NS(old=old)), basis_at=basis_at)
    s = NS(geo=NS(chi=chi))
    cells = []
    for z in h:
        for ch in chi:
            r = 6371.0 + z
            loc, w = incoming_rings.incoming(r, ch, nmu, nphi)
            p, up, ex = shells.frame(np.array(r), ch)
            d = loc[:, 0, None]*ex + loc[:, 1, None]*np.cross(up, ex) + loc[:, 2, None]*up
            cells.append(pack(p, d, nq, s, scope, w, ground=False))
    W, L = 15, 16
    u, wu = shells.gl(nq)
    c = dict(
        L=L, W=W, u=u, wu=wu, domain=domain, h=h.astype('f4'), chi=chi.astype('f4'),
        height=np.float32(1.2), mix=np.float32(.97), g=np.float32(.65), albedo=0.,
        hr=np.exp(-h / 8).astype('f4'), ha=np.exp(-h / 1.2).astype('f4'),
        ext=np.broadcast_to(np.asarray([.1, .3, .01, .002, .001], 'f4')[:, None], (5, W)).copy())
    rng = np.random.default_rng(5)
    field = rng.uniform(0, 0.1, (len(cells), 10+count(L), W)).astype('f4')
    field[:, 10:] *= .002
    return c, field, cells


def test_compact_warm_tiles(monkeypatch):
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends import mlx_warm as warm, mlx_backend

    c, field, cells = _case()
    small = subset(cells[25], np.arange(0, len(cells[25]['d']), 128))
    rays, y = ref.transport(small, c, field, return_basis=True)
    reference, _ = ref.project(small, rays, c, y)
    value, _ = warm.evaluate([small], [c], field, None)
    assert np.linalg.norm(value[0]-reference)/np.linalg.norm(reference) < 1e-4
    original = warm._launch
    stages = defaultdict(float)

    def timed(name, *args, **kw):
        start = perf_counter()
        out = original(name, *args, **kw)
        mx.eval(*out)
        mx.synchronize()
        stages[name] += perf_counter()-start
        return out

    monkeypatch.setattr(warm, '_launch', timed)
    packed = warm.CompactRays(cells)
    results = []
    for cap in (2**18, 2**20):
        monkeypatch.setenv(mlx_backend.TILE_ENV, str(cap))
        stages.clear()
        mx.reset_peak_memory()
        start = perf_counter()
        value, flux = warm.evaluate(cells, [c]*len(cells), field, None, packed)
        elapsed = perf_counter()-start
        results.append(value)
        warnings.warn(f'cap={cap} wall={elapsed:.5f} peak={mx.get_peak_memory()} kernels={dict(stages)}')
        assert np.all(np.isfinite(value)) and not np.any(flux)
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6, atol=1e-12)


def test_native_underflow():
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    sample = mx.array(np.asarray([np.finfo('f4').tiny, np.finfo('f4').tiny / 2], 'f4'))
    k = mx.fast.metal_kernel(name='sos_test_underflow', input_names=['x'], output_names=['out'],
                             source='out[0] = x[0]*0.5f; out[1] = x[1]*4.f;')
    a, = k(inputs=[sample], grid=(1,1,1), threadgroup=(1,1,1), output_shapes=[(2,)],
           output_dtypes=[mx.float32])
    warnings.warn(f'native underflow output words {np.asarray(a).view("u4").tolist()}')


def test_integrate_group_shapes(monkeypatch):
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends import mlx_warm as warm
    c, field, groups = _case()
    packed = warm.CompactRays(groups)
    original = warm._launch
    def measured(name, inputs, templates, grid, group, outputs, **kw):
        if name != 'integrate':
            return original(name, inputs, templates, grid, group, outputs, **kw)
        mx.eval(*inputs)
        measurements = defaultdict(list)
        for _ in range(3):
            for rows in (8, 2, 16, 4, 32, 8):
                begin = perf_counter()
                out = original(name, inputs, templates, grid, (grid[0], rows, 1), outputs)
                mx.eval(*out)
                measurements[rows].append(perf_counter()-begin)
        warnings.warn(f'integrate group median ms {[(k, round(1000*np.median(v), 3)) for k,v in measurements.items()]}')
        return out
    monkeypatch.setattr(warm, '_launch', measured)
    lo, hi = list(packed.tiles(2**20))[1]
    actual = warm._tile(packed, lo, hi, c, warm._device(field), None, warm._constants(c), warm._device([0.]))
    assert np.isfinite(actual).all()


def test_long_path_and_boundary():
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends import mlx_warm as warm
    c, field, groups = _case()
    g = subset(groups[21], np.arange(0, len(groups[21]['d']), 157))
    g['nq'] = 36
    c['u'], c['wu'] = shells.gl(g['nq'])
    boundary = np.linspace(0., 1., len(c['chi']) * c['W']).reshape(-1, c['W'])
    for empty in (False, True):
        if empty:
            g = dict(g, edges=np.zeros((len(g['d']), 2)), lower=np.ones((len(g['d']), 1), bool),
                     hits_ground=np.ones(len(g['d']), bool))
        y, basis = ref.transport(g, c, field, boundary=boundary, return_basis=True)
        expected, _ = ref.project(g, y, c, basis)
        value, _ = warm.evaluate([g], [c], field, boundary)
        assert np.linalg.norm(value[0]-expected) / np.linalg.norm(expected) < 1e-4


def test_tile_peak_model(monkeypatch):
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends import mlx_warm as warm, mlx_backend
    peaks = []
    for nmu, nphi, nq, caps in ((12, 16, 12, [2**18, 2**20]), (1, 2, 4, [2**20])):
        c, field, cells = _case(nmu, nphi, nq)
        packed = warm.CompactRays(cells)
        for cap in caps:
            monkeypatch.setenv(mlx_backend.TILE_ENV, str(cap))
            peak = 0
            times = []
            sampled_peaks = []
            for _ in range(12):
                mx.synchronize()
                mx.reset_peak_memory()
                start=perf_counter()
                value, _ = warm.evaluate(cells, [c]*len(cells), field, None, packed)
                times.append(perf_counter()-start)
                assert np.isfinite(value).all()
                mx.synchronize()
                sampled_peaks.append(mx.get_peak_memory())
                peak = max(peak, sampled_peaks[-1])
            peaks.append(peak)
            warnings.warn(f'peak model samples={sum(packed.samples)} cap={cap} peak={peak} wall={times}')
            warnings.warn(f'barrier peaks cap={cap}: {sampled_peaks}')
    w = (peaks[1]-peaks[0]) / (2**20 - 2**18)
    rate = (peaks[1]-peaks[2]-w*(2**20-12288)) / (2462436-12288)
    pred = (peaks[1]+max(0,rate)*(236311392-2462436))/(1024**3)
    warnings.warn(f'synthetic peak extrapolation R={rate:.5f} w={w:.5f} product={pred:.5f} GiB')


def test_two_layer_dark_source():
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends import mlx_warm as warm
    from sosfuse2.atmosphere.tropo_profile import profile32
    c, field, cells = _case()
    c['aer2'] = np.array([1., .7, .9, 1., .2, .35, 1.], 'f4')
    c['ha'] = profile32(c['h'], c['aer2'])
    c['ext'] = c['ext'] * np.linspace(.25, 2., c['W'], dtype='f4')
    field *= np.geomspace(1e-35, 1., len(cells)).astype('f4')[:, None, None]
    group = subset(cells[24], np.arange(0, len(cells[24]['d']), 121))
    y, basis = ref.transport(group, c, field, return_basis=True)
    target, _ = ref.project(group, y, c, basis)
    actual, _ = warm.evaluate([group], [c], field, None)
    error = np.linalg.norm(actual[0]-target, axis=0) / np.linalg.norm(target, axis=0)
    assert np.max(error) < 1e-3


def test_first_light_projection():
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends import mlx_warm as warm
    from sosfuse2.backends.arithmetic import basis

    c, _, cells = _case(nmu=4, nphi=8, nq=4)
    g = cells[25]
    rng = np.random.default_rng(32)
    radiance = rng.normal(size=(len(g['d']), c['W'], 4)).astype('f4')
    y = basis(g['d'][:, 2], c['L'])
    expected = ref.project(g, radiance, c, y)[0]
    actual = warm.project_cell(g, c, mx.array(radiance), y)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, actual.astype('f4').astype('f8'))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def _black_state(**params):
    import copy
    from sosfuse2.solve.ray_geometry import normal_state

    state = copy.deepcopy(normal_state())
    state['parameters']['albedo'] = 0.0
    state['parameters'].update(params)
    return state


def test_first_light_tiled(tmp_path, monkeypatch):
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends import mlx_backend, mlx_warm as warm
    from sosfuse2.solve.ray_geometry import TransientGeometry, create

    waves = np.linspace(380.0, 760.0, 15).tolist()
    s, scope, _ = create(tmp_path / 'first15', 'selftest', waves, 'numpy', 128, state=_black_state())
    geo = TransientGeometry(s, scope)
    cells = [geo.cell(i) for i in range(s.geo.ns)]
    c = s.kernel.setup(s, cells[0]['nq'])
    expected = np.zeros((len(cells), 4 + count(c['L']), c['W']), 'f8')
    for i, g in enumerate(cells):
        val, y = ref.transport(g, c, None, first=True, return_basis=True)
        expected[i], _ = ref.project(g, val, c, y)

    packed = warm.CompactRays(cells)
    results = []
    worst = 0.0
    for cap in (2**14, 2**20):
        monkeypatch.setenv(mlx_backend.TILE_ENV, str(cap))
        actual, flux = warm.evaluate(cells, [c] * len(cells), None, None, packed, first=True)
        assert np.all(np.isfinite(actual)) and not np.any(flux)
        results.append(actual)
        for i in range(len(cells)):
            ref_norm = np.linalg.norm(expected[i], axis=0)
            diff_norm = np.linalg.norm(actual[i] - expected[i], axis=0)
            rel = np.divide(diff_norm, ref_norm, out=np.where(diff_norm == 0, 0.0, np.inf), where=ref_norm > 0)
            worst = max(worst, float(np.max(rel)))
            assert np.max(rel) < 1e-3, f'cell {i} max rel {np.max(rel)}'
    warnings.warn(f'first_light_tiled max_rel={worst:.6g}')
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6, atol=1e-12)


def test_first_light_long_and_boundary(tmp_path):
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends import mlx_warm as warm
    from sosfuse2.solve.ray_geometry import TransientGeometry, create

    # Exercise NQ > 16 (NQ = 36), judge-like aerosol, and a nonzero boundary.
    waves = np.linspace(370.0, 810.0, 15).tolist()
    state = _black_state(aod550=0.22, angstrom=1.4, ssa=0.96, scale_height_km=2.0, g=0.74)
    s, scope, _ = create(tmp_path / 'first36', 'selftest', waves, 'numpy', 128, state=state)
    geo = TransientGeometry(s, scope)
    g = dict(geo.cell(25))
    g['nq'] = 36
    c = dict(s.kernel.setup(s, 36))
    boundary = np.linspace(0.0, 0.2, len(c['chi']) * c['W']).reshape(-1, c['W'])
    val, y = ref.transport(g, c, None, first=True, boundary=boundary, return_basis=True)
    expected, _ = ref.project(g, val, c, y)
    actual, flux = warm.evaluate([g], [c], None, boundary, first=True)
    assert np.all(np.isfinite(actual)) and not np.any(flux)
    rel = np.linalg.norm(actual[0] - expected, axis=0) / np.linalg.norm(expected, axis=0)
    warnings.warn(f'first_light_long_and_boundary max_rel={float(np.max(rel)):.6g}')
    assert np.max(rel) < 1e-3, f'NQ36 max rel {np.max(rel)}'


def test_product_tile_profile(monkeypatch):
    """Profile one 1M-sample tile on the 44x75 product grid dimensions (L=16, W=15)."""
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends import mlx_warm as warm

    h = np.linspace(0.0, 80.0, 44)
    chi = np.linspace(0.0, np.pi, 75)
    domain = Domain()
    old = NS(probe=shells, domain=domain)
    scope = NS(b=NS(sh=NS(old=old)), basis_at=basis_at)
    s = NS(geo=NS(chi=chi))
    # Build 14 cells at low/mid altitudes to fill one ~1M-sample product tile.
    cells = []
    for ih in (0, 2, 4, 6, 8, 10, 12):
        for ic in (30, 42):
            r = 6371.0 + h[ih]
            ch = chi[ic]
            loc, w = incoming_rings.incoming(r, ch, 12, 16)
            p, up, ex = shells.frame(np.array(r), ch)
            d = loc[:, 0, None] * ex + loc[:, 1, None] * np.cross(up, ex) + loc[:, 2, None] * up
            cells.append(pack(p, d, 12, s, scope, w, ground=False))
    W, L = 15, 16
    u, wu = shells.gl(12)
    c = dict(
        L=L, W=W, u=u, wu=wu, domain=domain, h=h.astype('f4'), chi=chi.astype('f4'),
        height=np.float32(1.2), mix=np.float32(0.97), g=np.float32(0.65), albedo=0.0,
        hr=np.exp(-h / 8.0).astype('f4'), ha=np.exp(-h / 1.2).astype('f4'),
        ext=np.broadcast_to(np.asarray([0.1, 0.3, 0.01, 0.002, 0.001], 'f4')[:, None], (5, W)).copy(),
    )
    rng = np.random.default_rng(7)
    field = rng.uniform(0, 0.1, (44 * 75, 10 + count(L), W)).astype('f4')
    field[:, 10:] *= 0.002
    packed = warm.CompactRays(cells)
    dc = warm._constants(c)
    raw_src = warm._device(field)
    src = warm._pack_source(raw_src, L, W)
    mx.eval(*src)
    bd = warm._device([0.0])
    # Warm up JIT
    warm._tile(packed, 0, len(cells), c, src, None, dc, bd)
    mx.synchronize()

    # Measure total uninstrumented tile time
    t_total = []
    for _ in range(5):
        t0 = perf_counter()
        warm._tile(packed, 0, len(cells), c, src, None, dc, bd)
        mx.synchronize()
        t_total.append(perf_counter() - t0)

    # Measure per-kernel breakdown
    original = warm._launch
    stages = defaultdict(list)
    stats = {}

    def timed(name, inputs, templates, grid, group, outputs, **kw):
        mx.synchronize()
        t0 = perf_counter()
        out = original(name, inputs, templates, grid, group, outputs, **kw)
        mx.eval(*out)
        mx.synchronize()
        stages[name].append(perf_counter() - t0)
        if name == 'vp_blocks':
            bstart = np.asarray(out[0])
            inc = np.asarray(inputs[1])
            stats['live_recs'] = int(inc[-1])
            stats['nblks'] = int(bstart[-1])
        return out

    monkeypatch.setattr(warm, '_launch', timed)
    for _ in range(3):
        warm._tile(packed, 0, len(cells), c, src, None, dc, bd)
    med = {k: round(1000.0 * float(np.median(v)), 3) for k, v in stages.items()}
    warnings.warn(
        f'PROD_TILE samples={sum(packed.samples)} rays={int(packed.cell_off[-1])} '
        f'wall_ms={round(1000.0 * float(np.median(t_total)), 3)} stats={stats} kernels_ms={med}'
    )


def test_albedo_first_ground_and_warm(tmp_path):
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    import copy
    from sosfuse2.backends import mlx_ops
    from sosfuse2.solve.ray_geometry import TransientGeometry, create, normal_state

    waves = np.linspace(380.0, 760.0, 15).tolist()
    for albedo in (0.24, 0.36):
        state = copy.deepcopy(normal_state())
        state['parameters']['albedo'] = albedo
        s, scope, _ = create(tmp_path / f'alb_{albedo}', 'selftest', waves, 'numpy', 128, state=state)
        geo = TransientGeometry(s, scope)
        cells = [geo.cell(i) for i in range(s.geo.ns)]
        c = s.kernel.setup(s, cells[0]['nq'])
        zero = np.zeros((s.geo.ns, 10 + count(c['L']), c['W']), 'f4')

        # Phase 0 first scattering (moments + first_flux)
        ref_m1 = np.zeros((s.geo.ns, 4 + count(c['L']), c['W']), 'f8')
        ref_f1 = np.zeros((s.geo.ns, c['W']), 'f8')
        for i, g in enumerate(cells):
            val, y = ref.transport(g, c, zero, first=True, return_basis=True)
            ref_m1[i], ref_f1[i] = ref.project(g, val, c, y)
        got_m1, got_f1 = mlx_ops.evaluate(geo, s, zero, True, 0, False, None)
        assert np.any(got_f1 > 0)
        for got, exp, axes in ((got_m1, ref_m1, (0, 1)), (got_f1, ref_f1, (0,))):
            sc = np.abs(exp).max(axis=axes)
            err = np.abs(got - exp).max(axis=axes)
            assert float((err / sc).max()) < 1e-3

        # Build a realistic nonzero source field from first_moments
        rng = np.random.default_rng(int(albedo * 100))
        field = rng.uniform(0.0, 0.05, zero.shape).astype('f4')
        field[:, 10:] *= 0.002

        # Phase 1 ground_only
        nchi = len(s.geo.chi)
        ref_fg = np.zeros((nchi, c['W']), 'f8')
        for i in range(nchi):
            g = cells[i]
            val, y = ref.transport(g, c, field, first=False, return_basis=True)
            _, ref_fg[i] = ref.project(g, val, c, y, ground_only=True)
        got_mg, got_fg = mlx_ops.evaluate(geo, s, field, False, 1, True, None)
        assert not np.any(got_mg)
        assert np.any(got_fg > 0)
        assert float((np.abs(got_fg - ref_fg).max(axis=0) / np.abs(ref_fg).max(axis=0)).max()) < 1e-3

        # Phase 0 warm with nonzero boundary
        boundary = got_fg + got_f1[:nchi]
        ref_mw = np.zeros_like(ref_m1)
        ref_fw = np.zeros_like(ref_f1)
        for i, g in enumerate(cells):
            val, y = ref.transport(g, c, field, first=False, boundary=boundary, return_basis=True)
            ref_mw[i], ref_fw[i] = ref.project(g, val, c, y)
        got_mw, got_fw = mlx_ops.evaluate(geo, s, field, False, 0, False, boundary)
        assert np.any(got_fw > 0)
        for got, exp, axes in ((got_mw, ref_mw, (0, 1)), (got_fw, ref_fw, (0,))):
            sc = np.abs(exp).max(axis=axes)
            err = np.abs(got - exp).max(axis=axes)
            assert float((err / sc).max()) < 1e-3


def test_strato_albedo_first_ground_and_warm(tmp_path):
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    import copy
    from sosfuse2.backends import mlx_ops
    from sosfuse2.solve.ray_geometry import TransientGeometry, create, normal_state

    waves = np.linspace(380.0, 760.0, 15).tolist()
    for albedo, tau in ((0.0, 0.03), (0.24, 0.03)):
        state = copy.deepcopy(normal_state())
        state['parameters']['albedo'] = albedo
        state['parameters']['stratosphere_tau550'] = tau
        state['stratosphere'] = {'status': 'enabled', 'profile': 'gaussian', 'center_km': 21.0, 'sigma_km': 3.0}
        s, scope, _ = create(tmp_path / f'st_alb_{albedo}', 'selftest', waves, 'numpy', 128, state=state)
        geo = TransientGeometry(s, scope)
        cells = [geo.cell(i) for i in range(s.geo.ns)]
        c = s.kernel.setup(s, cells[0]['nq'])
        K = count(c['L'])
        zero = np.zeros((s.geo.ns, 10 + 2 * K, c['W']), 'f4')

        # Phase 0 first scattering
        ref_m1 = np.zeros((s.geo.ns, 4 + K, c['W']), 'f8')
        ref_f1 = np.zeros((s.geo.ns, c['W']), 'f8')
        for i, g in enumerate(cells):
            val, y = ref.transport(g, c, zero, first=True, return_basis=True)
            ref_m1[i], ref_f1[i] = ref.project(g, val, c, y)
        got_m1, got_f1 = mlx_ops.evaluate(geo, s, zero, True, 0, False, None)
        assert got_m1 is not None
        sc = np.abs(ref_m1).max(axis=(0, 1))
        assert float((np.abs(got_m1 - ref_m1).max(axis=(0, 1)) / sc).max()) < 1e-3
        if albedo > 0:
            assert np.any(got_f1 > 0)
            scf = np.abs(ref_f1).max(axis=0)
            assert float((np.abs(got_f1 - ref_f1).max(axis=0) / scf).max()) < 1e-3
        else:
            assert not np.any(got_f1)

        rng = np.random.default_rng(77 + int(albedo * 100))
        field = rng.uniform(0.0, 0.05, zero.shape).astype('f4')
        field[:, 10:, :] *= 0.003

        # Phase 1 ground_only
        nchi = len(s.geo.chi)
        got_mg, got_fg = mlx_ops.evaluate(geo, s, field, False, 1, True, None)
        assert not np.any(got_mg)
        if albedo > 0:
            ref_fg = np.zeros((nchi, c['W']), 'f8')
            for i in range(nchi):
                val, y = ref.transport(cells[i], c, field, first=False, return_basis=True)
                _, ref_fg[i] = ref.project(cells[i], val, c, y, ground_only=True)
            assert np.any(got_fg > 0)
            assert float((np.abs(got_fg - ref_fg).max(axis=0) / np.abs(ref_fg).max(axis=0)).max()) < 1e-3
            boundary = got_fg + got_f1[:nchi]
        else:
            assert not np.any(got_fg)
            boundary = None

        # Phase 0 warm (with strato boundary end columns when albedo > 0)
        ref_mw = np.zeros_like(ref_m1)
        ref_fw = np.zeros_like(ref_f1)
        for i, g in enumerate(cells):
            val, y = ref.transport(g, c, field, first=False, boundary=boundary, return_basis=True)
            ref_mw[i], ref_fw[i] = ref.project(g, val, c, y)
        got_mw, got_fw = mlx_ops.evaluate(geo, s, field, False, 0, False, boundary)
        scw = np.abs(ref_mw).max(axis=(0, 1))
        assert float((np.abs(got_mw - ref_mw).max(axis=(0, 1)) / scw).max()) < 1e-3
        if albedo > 0:
            assert np.any(got_fw > 0)
            scfw = np.abs(ref_fw).max(axis=0)
            assert float((np.abs(got_fw - ref_fw).max(axis=0) / scfw).max()) < 1e-3
