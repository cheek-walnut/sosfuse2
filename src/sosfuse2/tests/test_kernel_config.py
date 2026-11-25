"""CPU seams for role resolution and the retained production sidecar ABI."""

from sosfuse2.foundation import paths
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from sosfuse2.checks import testkit
from sosfuse2.solve.source_solver import Kernel
from sosfuse2.backends.flags import resolve_config


def test_section_clocks_force_unsplit_and_preserve_production(monkeypatch):
    monkeypatch.setenv('SOSFUSE2_KSPLIT', '3')
    monkeypatch.delenv('SOSFUSE2_KSPLIT_VERIFY', raising=False)
    monkeypatch.delenv('SOSFUSE2_GROUND_SH_L', raising=False)
    flags = json.loads(paths.CONFIGS.joinpath('C3.json').read_text())['kernel_flags']
    assert resolve_config(flags).flags == tuple(sorted(flags))
    k = Kernel(flags=flags + ['EXP_SECTION_CLOCKS', 'WARM_RAYSPLIT'], L=16)
    assert k.ksplit_mode == 0
    assert 'EXP_SECTION_CLOCKS' in k.flags
    assert not {'FL_RAYSPLIT', 'WARM_RAYSPLIT'} & set(k.flags)
    assert resolve_config(flags, role='lut', W=1).flags == ('LUT_ONLY', 'STRATO')


@pytest.mark.parametrize('instrumented', [False, True])
def test_paired_sidecar_occupies_exactly_one_argument(tmp_path, monkeypatch, instrumented):
    from sosfuse2.tests.test_section_clocks_adversarial import A, launch_fixture, run_launch

    monkeypatch.setenv('SOSFUSE2_KSPLIT', '0')
    monkeypatch.delenv('SOSFUSE2_KSPLIT_VERIFY', raising=False)
    monkeypatch.delenv('SOSFUSE2_GROUND_SH_L', raising=False)
    monkeypatch.setenv('SOSFUSE2_DEVICE_PROFILE', 'mac')
    fixture = launch_fixture(flags=A + ('SH_AUX_PAIR',) + (('EXP_SECTION_CLOCKS',) if instrumented else ()))
    run_launch(fixture, tmp_path)
    args = fixture[2].calls[0][2]
    assert len(args) == 52 + (1 if instrumented else 3)  # clock buffer or split reduction
    assert int(args[48]) == 0 and int(args[49]) == 0  # absent density: NULL and analytic mask
    assert float(args[50]) == 1.0  # analytic air surface normalization
    if instrumented:
        assert args[47].shape == (5, 8, 576, 2)
        np.testing.assert_array_equal(args[47][:, :, :561, 0], fixture[-1][:, 10:, ::2].transpose(0, 2, 1))
    else:
        assert args[47].shape == (5, 561, 16)
        np.testing.assert_array_equal(args[47][:, :, :15], fixture[-1][:, 10:])
        assert not args[47][:, :, -1].any()


def test_instrumented_precompile_does_not_schedule_split_kernels(tmp_path, monkeypatch):
    from sosfuse2.jobs import precompile

    flags = json.loads(paths.CONFIGS.joinpath('C3.json').read_text())['kernel_flags']
    monkeypatch.setenv('SOSFUSE2_KERNEL_FLAGS', ','.join(flags + ['EXP_SECTION_CLOCKS']))
    monkeypatch.setenv('SOSFUSE2_KSPLIT', '3')
    monkeypatch.setattr('sys.argv', ['precompile', '--cache', str(tmp_path), '--ray-limit', 'on'])
    jobs = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, scheduled):
            jobs.extend(scheduled)
            return []

    monkeypatch.setattr(precompile, 'ProcessPoolExecutor', Pool)
    precompile.main()
    assert len(jobs) == len(precompile.SPECS) + 1  # unsplit specializations and LUT only
    assert jobs[-1][0] == 'strato_lut_build'
    assert all(not {'FL_RAYSPLIT', 'WARM_RAYSPLIT'} & set(job[1]) for job in jobs)


def test_retired_sigma_invocation_and_fw1_preprocess():
    import subprocess
    import pytest
    from sosfuse2.backends.flags import RETIRED_FLAGS

    flags = json.loads(paths.CONFIGS.joinpath('C3.json').read_text())['kernel_flags']
    sigma = [f for f in flags if f not in ('REG_LEAN', 'TRANSPORT_RS2', 'SH_COMBINE', 'CELL_SPLIT2')]
    assert resolve_config(sigma + ['STRATO'], W=1) == resolve_config(['STRATO'], W=1)
    legacy = tuple(sorted(RETIRED_FLAGS - {'REG_LEAN', 'TRANSPORT_RS2', 'SH_COMBINE', 'CELL_SPLIT2'}))
    with pytest.warns(UserWarning, match='W_KERNEL_FLAG_RETIRED'):
        old_call = resolve_config(legacy + ('STRATO',), W=1)
    empty_call = resolve_config(('STRATO',), W=1)
    assert old_call == empty_call and not empty_call.c3_fw15
    source = (paths.PKG / 'backends/kernel.cu').read_text()
    testkit.require_tool('clang')
    for role in ('transport', 'lut'):
        cfg = resolve_config(('STRATO',), role=role, W=1)
        cmd = ['clang', '-E', '-P', '-x', 'c++', '-DFW=1', '-DK_FIRST=0', '-DK_PHASE=0', '-DK_BOUNDARY=0']
        out = subprocess.run(
            cmd + ['-D' + f for f in cfg.defines] + ['-'], input=source, capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
    with pytest.raises(ValueError, match='E_KERNEL_FLAG'):
        resolve_config(('REG_LEAN_TYPO',))


def test_sinv_reserved_argument_removed():
    k = object.__new__(Kernel)
    assert len(k.ksplit_pointer_args({})) == 3


def test_physical_scope_and_default_ksplit(monkeypatch):
    from sosfuse2.solve.source_solver import ksplit_setting

    monkeypatch.delenv('SOSFUSE2_KSPLIT', raising=False)
    assert ksplit_setting('cupy', ('STRATO',), 16) == '3'
    assert ksplit_setting('cupy', (), 16) == '0'
    assert ksplit_setting('cupy', ('STRATO',), 24) == '0'
    assert not resolve_config(('STRATO',), W=1).ksplit_supported
    assert not resolve_config(()).ground_sh_supported
    assert not resolve_config(('STRATO', 'TROPO_MIE')).ground_sh_supported
    assert not resolve_config(('STRATO', 'EXP_SECTION_CLOCKS')).ground_sh_supported
    assert resolve_config(('STRATO',)).ground_sh_supported


@pytest.mark.parametrize('waves, planes, parts, mode', [(1, 2, 1, 0), (15, 1, 2, 3)])
def test_installed_transport_layout(tmp_path, monkeypatch, waves, planes, parts, mode):
    from sosfuse2.asset_tools.strato_states import build
    from sosfuse2.atmosphere.density import LEGACY
    from sosfuse2.jobs import pipeline_execution as pe, run_state
    from sosfuse2.solve.ray_geometry import Bundle, TransientGeometry
    from sosfuse2.tests.test_solver_promotion import prepare

    for key in ('KSPLIT', 'KSPLIT_VERIFY', 'GROUND_SH_L', 'DEVICE_PROFILE', 'KERNEL_FLAGS'):
        monkeypatch.delenv('SOSFUSE2_' + key, raising=False)
    monkeypatch.setenv('SOSFUSE2_LAUNCH_CELLS', '2')
    state = build()['states'][1]
    state['density'] = dict(LEGACY)
    cfg = run_state.settings_for(SimpleNamespace(reduced=True, settings_json=None))
    cfg.update(L=16, wavelengths=waves, wavelengths_nm=np.linspace(450, 650, waves).tolist())
    s, scope = prepare(tmp_path, monkeypatch, state, cfg)
    geometry = TransientGeometry(s, scope)
    bundle = Bundle([geometry.cell(i) for i in range(3)])
    pe.prepare_bundle(s.kernel, s, bundle)
    payload = pe.device_payload(s, SimpleNamespace(bundle=bundle, ground_bundle=bundle))

    # Select the service's CuPy defaults while allocating and inspecting its launch arguments on the CPU.
    monkeypatch.setitem(sys.modules, 'cupy', np)
    monkeypatch.setattr(np, 'cuda', SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1)), raising=False)
    restored, geometry = pe.restore_device_payload(payload, ({}, {}))
    k = restored.kernel
    raw = Mock()
    k.raw_kernel = Mock(return_value=raw)
    k.strato_lut_device = lambda coeff, W: np.zeros((33 + 16384 + 1) * W, 'f4')
    field = np.zeros((s.geo.ns, 10 + 2 * k.K, waves), 'f4')
    k.launch(geometry.bundle, restored, field, phase=2)

    assert raw.call_count == 2
    for call, cells in zip(raw.call_args_list, (2, 1)):
        grid, _, args = call.args
        # L16 has 153 coefficients padded to 160, with two floats per wavelength pair.
        expected_bytes = planes * s.geo.ns * ((waves + 1) // 2) * 160 * 8
        assert (k.config().W, k.ksplit_mode, args[47].nbytes, grid[0]) == (waves, mode, expected_bytes, cells * parts)
        order = args[-1] if waves == 1 else args[-4]
        np.testing.assert_array_equal(np.sort(order), np.arange(cells * parts))
    assert k.raw_kernel.call_args.args[0] == k._W == waves
    assert k.ksplit == bool(mode)
    if waves == 1:
        assert k.ksplit_preflight(restored, []) == dict(requested=0, effective=0, reason='fw1', items=dict(peak={}, reserve={}))
        k.launch(geometry.bundle, restored, field, phase=1)


@pytest.mark.parametrize(
    ('backend', 'flags', 'L', 'W', 'env', 'expected'),
    [
        ('cupy', ('STRATO',), 16, 15, '', ('3', None)),
        ('cupy', ('STRATO',), 24, 15, '', ('0', 'sh_order_gt_16')),
        ('cupy', ('STRATO',), 16, 1, '', ('0', 'fw1')),
        ('cupy', ('STRATO', 'EXP_SECTION_CLOCKS'), 16, 15, '3', ('0', 'instrumented')),
        ('cupy', (), 16, 15, '', ('0', 'no_strato')),
        ('cupy', ('STRATO', 'WARM_RAYSPLIT'), 16, 15, '', ('0', 'warm_raysplit')),
        ('cupy', ('STRATO',), 16, 15, '0', ('0', 'explicit_zero')),
        ('cupy', ('STRATO',), 24, 15, '1', ('1', None)),
        ('cupy', ('STRATO',), 16, 1, '3', ('3', None)),
        ('numpy', ('STRATO',), 16, 1, '', ('0', 'numpy')),
    ],
)
def test_ksplit_request_reason(monkeypatch, backend, flags, L, W, env, expected):
    from sosfuse2.solve.source_solver import _ksplit_request

    monkeypatch.setenv('SOSFUSE2_KSPLIT', env)
    assert _ksplit_request(backend, resolve_config(flags, W=W), L) == expected


def test_ksplit_zero_records_reason_without_fallback(monkeypatch, capsys, recwarn):
    from sosfuse2.jobs.pipeline_execution import install_constants
    from sosfuse2.solve.source_solver import Solver

    monkeypatch.delenv('SOSFUSE2_KSPLIT', raising=False)
    monkeypatch.delenv('SOSFUSE2_KSPLIT_VERIFY', raising=False)
    kernel = Kernel(backend='numpy', flags=(), L=16)
    kernel.ksplit_emit = lambda _: pytest.fail('unexpected fallback')
    state = SimpleNamespace(kernel=kernel, W=1, geo=SimpleNamespace(ns=1))
    install_constants(kernel, state, {1: dict(W=1)})

    def first_light(s, _field, **_):
        assert kernel.setup(s, 1) == dict(W=1)
        return np.empty(0), np.empty(0)

    solver = Solver(state, SimpleNamespace(evaluate=first_light), prepared=True)
    solver.initialize()
    decision = solver.timings['ksplit']
    assert decision == state.ksplit_decision == dict(requested=0, effective=0, reason='numpy', items=dict(peak={}, reserve={}))
    assert kernel.ksplit_preflight(state, []) == decision
    assert capsys.readouterr() == ('', '')
    assert not recwarn


def test_retired_names_do_not_change_cache_or_toolkey(monkeypatch):
    import pytest
    from sosfuse2.backends.compile_cache import cache_key
    from sosfuse2.checks.gate import toolkey

    monkeypatch.setenv('SOSFUSE2_KERNEL_FLAGS', '')
    plain = cache_key('synthetic', {'nvrtc': [12, 8]}, 256)
    monkeypatch.setenv('SOSFUSE2_KERNEL_FLAGS', 'REG_LEAN,SH_COMBINE,SPECIALIZE,PROFILE')
    with pytest.warns(UserWarning, match='W_KERNEL_FLAG_RETIRED'):
        assert cache_key('synthetic', {'nvrtc': [12, 8]}, 256) == plain
    meta = dict(device={}, config=dict(flags=['STRATO'], stage1=[], async_=False, L=16, block=256))
    plain = toolkey(meta)
    meta['config']['flags'] += ['REG_LEAN', 'SH_COMBINE', 'PROFILE']
    with pytest.warns(UserWarning, match='W_KERNEL_FLAG_RETIRED'):
        assert toolkey(meta) == plain
