"""Host checks for CUDA variant selection, launch envelopes, layout and angular interpolation."""

from dataclasses import replace
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest

from sosfuse2.backends import vram_budget as budget
from sosfuse2.backends.arithmetic import basis
from sosfuse2.backends.flags import resolve_config
from sosfuse2.backends.sh_layout import FastTransport, count, kernel_source
from sosfuse2.checks import testkit
from sosfuse2.solve import source_solver as solver
from sosfuse2.solve.source_iteration import _use_gpu_history
from sosfuse2.tests.test_kernel_static import STUB
from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA


@pytest.mark.parametrize('order', [16, 24, 32])
@pytest.mark.parametrize('width', [1, 15])
def test_compiled_layout(order, width):
    compiler = testkit.require_tool('clang++')
    k = solver.Kernel(L=order, block_size=256)
    cfg = k.config(width)
    warm = FastTransport.select(order, cfg, False, 0)
    first = FastTransport.select(order, cfg, True, 0, False, True)
    field = np.arange(3 * (10 + k.K) * width, dtype='f4').reshape(3, 10 + k.K, width)
    packed = solver.sh_aux_waves(field, warm)
    np.testing.assert_array_equal(packed[:, :, :width], field[:, 10:])
    assert not packed[:, :, width:].any()
    common = STUB + SYNTAX_STUB_EXTRA + '\nvoid sincosf(float, float*, float*);\n' + kernel_source(order)
    for spec in (first, warm, replace(warm, role='angular'), replace(warm, role='direct'),
                 replace(warm, role='direct', stage_plane=False)):
        first_light = spec == first
        options = k.transport_options(spec, cfg.defines, (), first_light, 0, False)
        checks = [f'FAST_WAVE_STRIDE == {packed.shape[-1]}', f'K == {k.K}',
                  f'TransportPolicy::fast_first == {int(first_light)}',
                  f'TransportPolicy::fast_warm == {int(not first_light)}']
        if spec.role == 'first':
            size = k.shared_bytes(width, 0, True, ray_limit=True)
            checks += [f'HG_NODES == {spec.hg_nodes}',
                       f'{size} == 8*FW+16+8*(FAST_BATCH*sizeof(FirstPoint)+(FAST_BATCH+1)*(NCOMP+1)*4+(32+RAY_CONST_F+12)*4)']
        if spec.role == 'warm':
            size = k.shared_bytes(width, 0)
            checks += [f'{size} == 8*FW+12*4+8*((YPAD+4*FW+66+(SH_L==32 ? 4*FW : 0)+RAY_CONST_F)*4'
                       '+FAST_BATCH*sizeof(WarmPoint))+16+512']
        if spec.role == 'angular':
            checks.append(f'SH_NODES == {spec.sh_nodes}')
        if spec.role in ('direct', 'angular'):
            checks += [f'SP_THREADS == {spec.THREADS}', f'SP_MOMENTS == {spec.PROJ_MOMENTS}']
        if spec.role == 'direct':
            checks += [f'VP_STAGE_PLANE*K*FAST_WAVE_STRIDE*sizeof(float) == {spec.direct_shared}',
                       f'VP_VISITOR_TILE == {spec.visitor_tile}']
        check = '\n'.join(f'static_assert({c});' for c in checks)
        result = subprocess.run([compiler, *options[:1], '-fsyntax-only', '-x', 'c++',
                                 *(o for o in options if o.startswith('-D')), '-'],
                                input=common + '\n' + check, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    assert warm.defines() in k.kernel_key((), width, width, False, 0, False)
    assert first.defines() in k.kernel_key((), width, width, True, 0, False, ray_limit=True)


def test_role_and_ground_exclusions():
    for flags in (('STRATO',), ('EXP_SECTION_CLOCKS',)):
        cfg = resolve_config(flags)
        assert FastTransport.select(32, cfg, False, 0).role == 'none'
        assert FastTransport.select(32, cfg, True, 0, ray_limit=True).role == 'none'
    cfg = resolve_config(())
    for phase in (0, 1, 2):
        assert FastTransport.select(16, cfg, False, phase).role == 'warm'
    assert FastTransport.select(32, cfg, False, 0, split=True).role == 'none'
    assert FastTransport.select(32, resolve_config(['WARM_RAYSPLIT']), False, 0).role == 'none'
    for phase, boundary, ray_limit in ((1, False, True), (2, False, True), (0, True, True), (0, False, False)):
        assert FastTransport.select(32, cfg, True, phase, boundary, ray_limit).role == 'none'
    mie = FastTransport.select(32, resolve_config(['TROPO_MIE']), True, 0, ray_limit=True)
    assert mie.role == 'first' and not mie.hg
    first = FastTransport.select(32, cfg, True, 0, ray_limit=True)
    assert first.hg and solver._hg_fits(first, np.array([0, 512]), 1 << 30, 2.5)
    for counts, free, seconds in (([0], 1 << 30, 2.5), ([513], 1 << 30, 2.5),
                                  ([512], 128, 2.5), ([512], 1 << 30, 1e-8)):
        assert not solver._hg_fits(first, np.array(counts), free, seconds)
    with pytest.raises(ValueError, match='E_WAVELENGTHS'):
        resolve_config((), W=7)


def test_independent_buffer_maxima_and_probe_limits():
    co = np.array([0, 5000, 5001])
    pc = np.array([0, 1, 5001])
    def peak(p, r):
        return budget.transport_scratch(p, r, 64, count(24), 15, True, False)
    def chunks(free, resident=0, seconds=2.5, hard=None):
        return budget.transport_chunks(co, pc, 64, count(24), 15, True, False, 1, seconds, free, resident,
                                       budget=hard)
    headroom = 64 << 20
    independent = max(peak(1, 5000), peak(5000, 1))
    assert chunks(headroom + independent) is None
    free = headroom + peak(5000, 5000)
    assert chunks(free)[0] == [(0, 1), (1, 2)]
    assert chunks(free, 1) is None
    with pytest.raises(RuntimeError, match='E_LAUNCH_BUDGET'):
        chunks(free, seconds=.001)
    assert chunks(free, seconds=.001, hard=1)[0] == [(0, 1), (1, 2)]
    assert budget.transport_scratch(1 << 29, 1, 64, 153, 1, True, False) is None
    assert budget.transport_scratch(1, 1 << 31, 64, 153, 1, False, False) is None
    for cells in (64, budget.COUNT_SHARED_BYTES // 4 + 1):
        n = 40000
        cost = budget.transport_scratch(n // 4, 0, cells, 153, 15, True, False)
        _, nb = budget.count_segments(n, budget.PLAN_COUNT_BLOCKS)
        sort = 12 * nb * cells if 4 * cells <= budget.COUNT_SHARED_BYTES else 24 * n
        assert cost >= 8 * n + sort + budget.aerosol_bytes(n // 4, 15)


@pytest.mark.parametrize('order', [16, 24, 32])
def test_spline_poles_seams_and_modes(order):
    layout = FastTransport(order, 1, 'angular')
    polar, azimuth = solver.sh_spline_basis(layout)
    n = layout.bins
    # All knots and 15 interior samples per interval, including both end intervals.
    axis = np.linspace(0, n, n * 16 + 1)
    knot = np.minimum(axis.astype(int), n - 1)
    t = axis - knot
    weights = np.stack(((1-t)**3, 3*t**3-6*t**2+4, -3*t**3+3*t**2+3*t+1, t**3), axis=1)/6
    def sample(coeff):
        return np.einsum('ij,ijk->ik', weights, coeff[knot[:, None] + np.arange(4)])
    direct = basis(np.cos(axis*np.pi/n), order)
    exact_phi = np.cos(np.outer(axis*np.pi/n, np.arange(order + 1)))
    # Componentwise envelope of the original cubic grid. Relative error of
    # a single basis function is unbounded at its zeros.
    assert np.max(abs(sample(polar) - direct) / abs(direct).max(axis=0)) < 0.0017
    assert np.max(abs(sample(azimuth) - exact_phi)) < 0.0026
    np.testing.assert_allclose(sample(polar)[::16], direct[::16], atol=3e-7, rtol=1e-6)
    m = np.concatenate([np.arange(l + 1) for l in range(order + 1)])
    sign = np.where(m % 2, -1, 1)
    np.testing.assert_allclose(polar[0], sign * polar[2], atol=1e-6)
    np.testing.assert_allclose(polar[-1], sign * polar[-3], atol=1e-6)


def test_runtime_choices():
    k = solver.Kernel(L=32)
    k.backend = 'cupy'
    k.device = lambda: ('test', dict(launch_target_seconds=2.5, launch_budget_seconds=30), '')
    k.ksplit_memory = lambda: 1 << 30
    bins = FastTransport(32, 15, 'angular').bins
    co, pc = np.arange(5, dtype='i8') * bins, np.arange(5, dtype='i8') * 5000
    bundle = SimpleNamespace(_fast_vprod_meta=(co, pc))
    launch = SimpleNamespace(c=dict(h=np.arange(2), chi=np.arange(2)), W=15, nr=int(co[-1]), mode='fixed',
                             step=1, flux_only=False, xp=np, src=np.array([1.]))
    for w in (1, 15):
        assert solver._use_fast_vprod(k, False, 0, w, False)
        assert not solver._use_fast_vprod(k, False, 2, w, False)
        assert not solver._use_fast_vprod(k, True, 0, w, False)
        assert not solver._use_fast_vprod(k, False, 0, w, True)
    assert solver._prepare_fast_transport(k, launch, bundle, 0, True) and launch.angular and not launch.zero_volume
    # The loaded functions' actual local allocation is charged against the same live memory.
    launch.ks, launch.fast_vprod, launch.mode = False, True, 'fixed'
    k.fast_vprod_kernels = lambda *args: {'local_reserve_bytes': 1}
    solver._select_transport_kernels(k, launch, bundle, False, 0, None, None)
    assert launch.fast_chunks == [(i, i + 1) for i in range(4)]
    k.fast_vprod_kernels = lambda *args: {'local_reserve_bytes': 1 << 31}
    with pytest.raises(solver._FastResourceError, match='workspace'):
        solver._select_transport_kernels(k, launch, bundle, False, 0, None, None)
    launch.nr = 1
    assert solver._prepare_fast_transport(k, launch, bundle, 0, True) and not launch.angular
    # Same number of quadrature points, one fewer angular ray than the table threshold.
    sparse_co = co.copy()
    sparse_co[-1] -= 1
    sparse = SimpleNamespace(_fast_vprod_meta=(sparse_co, pc))
    launch.nr = int(sparse_co[-1])
    assert solver._prepare_fast_transport(k, launch, sparse, 0, True) and not launch.angular
    launch.src[:] = 0
    assert solver._prepare_fast_transport(k, launch, sparse, 0, True) and launch.zero_volume
    launch.src[:] = 1
    launch.nr = int(co[-1])
    cheap = SimpleNamespace(_fast_vprod_meta=(co, np.arange(5, dtype='i8')))
    assert solver._prepare_fast_transport(k, launch, cheap, 0, True) and not launch.angular
    launch.src[:] = 0
    assert solver._prepare_fast_transport(k, launch, bundle, 0, True) and launch.zero_volume
    launch.src[:] = 1
    k.deterministic = True
    assert solver._prepare_fast_transport(k, launch, bundle, 0, True) and not launch.angular
    k.deterministic = False
    assert solver._prepare_fast_transport(k, launch, bundle, 0, False) and not launch.angular
    need = budget.transport_scratch(5000, bins, 4, count(32), 15, True, False)
    k.ksplit_memory = lambda: (64 << 20) + need
    assert solver._prepare_fast_transport(k, launch, bundle, 0, True) and not launch.angular
    k.ksplit_memory = lambda: 0
    assert not solver._prepare_fast_transport(k, launch, bundle, 0, True)
    k.backend = 'numpy'
    assert not solver._use_fast_vprod(k, False, 0, 1, False)


def test_history_staging():
    for order in (16, 24, 32):
        for w in (1, 15):
            args = [True, True, False, False, order, w, 500, 1 << 28]
            assert _use_gpu_history(*args)
            for i, value in ((0, False), (1, False), (2, True), (5, 7), (7, 0)):
                assert not _use_gpu_history(*args[:i], value, *args[i+1:])
            args[3] = True
            assert _use_gpu_history(*args) == ((order, w) == (24, 15))


def test_compiled_resource_fallbacks_and_cached_capacity():
    props = dict(sharedMemPerBlock=48 << 10, sharedMemPerBlockOptin=96 << 10,
                 regsPerBlock=65536, maxThreadsPerMultiProcessor=1536, multiProcessorCount=1)
    raw_calls = []
    occupancy = [1]

    class Raw:
        def __init__(self, code, name, options, backend):
            raw_calls.append(name)
            staged = '-DVP_STAGE_PLANE=1' in options
            static = (35904 if staged else 17952) if name == 'sos_vprod_direct' else 0
            self.kernel = SimpleNamespace(ptr=1)
            self.attributes = dict(shared_size_bytes=static, max_threads_per_block=128, num_regs=100,
                                   local_size_bytes=32 if name == 'sos_fast_slim' else 0)

        def compile(self):
            pass

    k = solver.Kernel(L=32, flags=())
    cuda = SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0, getDeviceProperties=lambda _: props),
                           driver=SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=lambda *_: occupancy[0]))
    k.xp = SimpleNamespace(RawKernel=Raw, cuda=cuda)
    k.ksplit_memory = lambda: 1 << 30
    spec = FastTransport(32, 15, 'direct')
    selected = k.fast_vprod_kernels(spec, False)
    assert selected['shared_bytes'] == selected['sos_vprod_direct'].max_dynamic_shared_size_bytes == spec.direct_shared
    k.ksplit_memory = lambda: 10
    with pytest.raises(solver._FastResourceError, match='sos_fast_slim'):
        k.fast_vprod_kernels(spec, False)
    k.ksplit_memory = lambda: 1 << 30
    k.raw.clear()
    props['sharedMemPerBlockOptin'] = props['sharedMemPerBlock']
    assert k.fast_vprod_kernels(spec, False)['shared_bytes'] == 0
    assert raw_calls.count('sos_vprod_direct') == 3
    for prop, value in (('regsPerBlock', 100), ('sharedMemPerBlockOptin', 8192)):
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(props, prop, value)
            k.raw.clear()
            with pytest.raises(solver._FastResourceError):
                k.fast_vprod_kernels(spec, False)
    k.raw.clear()
    occupancy[0] = 0
    with pytest.raises(solver._FastResourceError):
        k.fast_vprod_kernels(spec, False)


def test_shared_program_can_compile_missing_sibling():
    from sosfuse2.tests.test_review_fixes import FakeRaw

    cuda = SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0,
                                                   getDeviceProperties=lambda _: dict(sharedMemPerBlock=1 << 20)))
    a, b = (solver.Kernel(L=16, flags=()) for _ in range(2))
    a.xp = b.xp = SimpleNamespace(RawKernel=FakeRaw, cuda=cuda)
    b.raw, b.attributes = a.raw, a.attributes
    raw = a.raw_kernel(15, first=True, ray_limit=True)
    assert b.raw_kernel(15, first=True, ray_limit=True) is raw
    assert b.fl_project_kernel(15).options == raw.options


def test_failure_after_a_chunk_restarts_from_first_cell(monkeypatch):
    k = solver.Kernel()
    rows = np.arange(4)
    # Inject a failure after writing a partial pass. Check the public return of launch().
    def prepare(kernel, launch, *args):
        launch.flux = np.empty(4)
        launch.moments = launch.query = None
        launch.section_buffer = None

    for name in ('_allocate_transport_outputs', '_assemble_transport_arguments', '_record_transport_clocks',
                 '_select_transport_kernels'):
        monkeypatch.setattr(solver, name, lambda *args: None)
    monkeypatch.setattr(solver, '_prepare_transport_constants', prepare)
    passes = []
    def order(kernel, launch, bundle, first, phase, fast, table):
        passes.append((fast, table))
        launch.fast_vprod, launch.angular = fast, table

    def run(kernel, launch, *args):
        if launch.fast_vprod:
            launch.flux[:2] = -99
            raise MemoryError('allocator capacity')
        launch.flux[:] = rows

    monkeypatch.setattr(solver, '_prepare_launch_order', order)
    monkeypatch.setattr(solver, '_dispatch_transport', run)
    with pytest.warns(RuntimeWarning, match='allocator capacity'):
        _, result, _ = k.launch(None, None, None)
    np.testing.assert_array_equal(result, rows)
    assert passes == [(True, True), (True, False), (False, False)]
    for status in (2, 701, 700):
        exc = type('CUDADriverError', (RuntimeError,), {'status': status})()
        assert solver._transport_resource_error(exc) == (status != 700)
    assert not solver._transport_resource_error(ValueError())


@pytest.mark.parametrize(('g', 'deterministic', 'boundary', 'expect'), [
    (.8, False, None, True), (-.8, False, None, True), (.8001, False, None, False),
    (.7, True, None, False), (.7, False, np.zeros(1), False),
])
def test_hg_dispatch_on_each_call(g, deterministic, boundary, expect):
    k = solver.Kernel(flags=(), L=16)
    k.deterministic = deterministic
    k.raw_kernel = lambda *args, **kw: None
    calls = []
    k.prepare_hg_table = lambda launch, layout: calls.append(layout.W)
    launch = SimpleNamespace(ks=False, mode='fixed', xp=np, ncell=3, nr=3, W=1, args=[],
                             fast_vprod=False, ground_sh_l=0, c={'g': g})
    solver._select_transport_kernels(k, launch, None, True, 0, boundary, None)
    assert bool(calls) == expect


def test_fixed_table_ranges_and_indivisible_work():
    k = solver.Kernel()
    profile = dict(launch_budget_seconds=10, launch_target_seconds=1, launch_cells_fixed=2, launch_cells_initial=32)
    k.device = lambda: ('test', profile, '')
    assert list(k.table_ranges('table', 1, 6, np.arange(7))) == [(1, 3), (3, 5), (5, 6)]
    rate = 100
    assert solver._bounded_end(np.array([0, 40, 70, 150, 200]), 0, 4, 1, rate, profile) == 2
    assert solver._bounded_end(np.array([0, 600, 900]), 0, 2, 1, rate, profile) == 1
    with pytest.raises(RuntimeError, match='E_LAUNCH_BUDGET'):
        solver._bounded_end(np.array([0, 801]), 0, 1, 1, rate, profile)

    k.predict_seconds = lambda *args: .5
    work = np.array([0, 80, 160, 600, 900])
    assert solver._bounded_end(work, 0, 4, 1, rate, profile, kernel=k, kind='warm') == 3
    k.predict_seconds = lambda *args: 1.01
    assert solver._bounded_end(work, 0, 4, 1, rate, profile, kernel=k, kind='warm') == 1


def test_zero_source_reduction_is_split_and_stops_early():
    k = solver.Kernel()
    profile = dict(launch_budget_seconds=30, launch_target_seconds=2.5, launch_cells_fixed=2, launch_cells_initial=32)
    k.device = lambda: ('test', profile, '')
    k.xp = SimpleNamespace(zeros=np.zeros)
    ranges = []

    def scan(grid, block, args, shared):
        n, source, result = args
        assert grid[0] * block[0] >= n == source.size
        result[0] |= bool(np.any(source))
    k.vprod_plan_module = lambda: SimpleNamespace(get_function=lambda name: scan)

    def timed(name, func, grid, block, args, shared, units, work, start, end):
        ranges.append((start, end))
        func(grid, block, args, shared)

    k.timed_launch = timed
    source = np.zeros((5, 3, 1), 'f4')
    assert k.source_is_zero(source)
    assert ranges == [(0, 2), (2, 4), (4, 5)]
    ranges.clear()
    source[2, 1] = 1.e-30
    assert not k.source_is_zero(source)
    assert ranges == [(0, 2), (2, 4)]

@pytest.mark.parametrize(('physical', 'groups', 'width'), [(1, 1, 1), (1, 2, 15), (7, 1, 15), (14, 3, 15), (15, 1, 15)])
def test_partial_spectral_chunk_keeps_last_lane_and_uses_same_layout(physical, groups, width, monkeypatch, tmp_path):
    from contextlib import nullcontext
    from sosfuse2.jobs import run_state
    from sosfuse2.atmosphere import density

    waves = np.linspace(400, 700, physical, dtype='f4')
    job = SimpleNamespace(trace=None, chunk=waves, chunks=[None] * groups, name='chunk', state={}, index=groups-1,
                          pipeline_lock=False, prepare=tmp_path, cfg={}, table_density=None, ksplit_decision=None)
    args = SimpleNamespace(backend='numpy', block_size=128, geometry_cache=None)
    created = []
    def create(path, cfg, wavelengths, *args, **kw):
        created.append(wavelengths)
        return SimpleNamespace(kernel=SimpleNamespace(xp=np, profile_n=161)), None

    monkeypatch.setattr(run_state, 'create_configured', create)
    monkeypatch.setattr(run_state, 'attach_cache_event', lambda *args: nullcontext([]))
    monkeypatch.setattr(run_state, 'phase', lambda *args, **kw: None)
    monkeypatch.setattr(density, 'evidence', lambda *args: {})
    assert run_state._prepare_wavelength_chunk(job, args, np, lambda *args: None) == physical
    rows, = created
    assert len(rows) == width
    np.testing.assert_array_equal(rows[:physical], waves)
    np.testing.assert_equal(rows[physical:], np.repeat(waves[-1], width - physical))
    for order in (16, 24, 32):
        cfg = resolve_config((), W=len(rows))
        layout = FastTransport.select(order, cfg, False, 0)
        assert layout == FastTransport.select(order, resolve_config((), W=width), False, 0)
        field = np.broadcast_to(rows, (2, 10 + count(order), width))
        sidecar = solver.sh_aux_waves(field, layout)
        np.testing.assert_array_equal(sidecar[..., :width], field[:, 10:])
        assert not sidecar[..., width:].any()
