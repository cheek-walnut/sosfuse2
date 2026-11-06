"""CPU seams for role resolution and the retained production sidecar ABI."""
from sosfuse2 import paths
import json
from pathlib import Path
import numpy as np
from sosfuse2 import testing
from sosfuse2.fused import Kernel
from sosfuse2.kernel_config import resolve_config


def test_section_clocks_force_unsplit_and_preserve_production(monkeypatch):
    monkeypatch.setenv('SOSFUSE2_KSPLIT', '3')
    monkeypatch.delenv('SOSFUSE2_KSPLIT_VERIFY', raising=False)
    monkeypatch.delenv('SOSFUSE2_GROUND_SH_L', raising=False)
    flags = json.loads(paths.CONFIGS.joinpath('C3.json').read_text())['kernel_flags']
    assert resolve_config(flags).flags == tuple(sorted(flags))
    k = Kernel(flags=flags+['EXP_SECTION_CLOCKS', 'WARM_RAYSPLIT'], L=16)
    assert k.ksplit_mode == 0
    assert 'EXP_SECTION_CLOCKS' in k.flags
    assert not {'FL_RAYSPLIT', 'WARM_RAYSPLIT'} & set(k.flags)
    assert resolve_config(flags, role='lut', W=1).flags == (
        'LUT_ONLY', 'STRATO')


def test_paired_sidecar_occupies_exactly_one_argument(tmp_path, monkeypatch):
    from sosfuse2.tests.test_section_clocks_adversarial import A, launch_fixture, run_launch
    monkeypatch.setenv('SOSFUSE2_KSPLIT', '0')
    monkeypatch.delenv('SOSFUSE2_KSPLIT_VERIFY', raising=False)
    monkeypatch.delenv('SOSFUSE2_GROUND_SH_L', raising=False)
    monkeypatch.setenv('SOSFUSE2_DEVICE_PROFILE', 'mac')
    fixture = launch_fixture(flags=A+('SH_AUX_PAIR',))
    run_launch(fixture, tmp_path)
    args = fixture[2].calls[0][2]
    assert len(args) == 52  # base + paired sidecar + profile pointer/mask/surface + cell order
    assert int(args[48]) == 0 and int(args[49]) == 0  # absent density: NULL and analytic mask
    assert float(args[50]) == 1.  # analytic air surface normalization
    assert args[47].shape == (5, 8, 576, 2)
    np.testing.assert_array_equal(args[47][:, :, :561, 0], fixture[-1][:, 10:, ::2].transpose(0, 2, 1))


def test_instrumented_precompile_does_not_schedule_split_kernels(tmp_path, monkeypatch):
    from sosfuse2 import precompile
    flags = json.loads(paths.CONFIGS.joinpath('C3.json').read_text())['kernel_flags']
    monkeypatch.setenv('SOSFUSE2_KERNEL_FLAGS', ','.join(flags+['EXP_SECTION_CLOCKS']))
    monkeypatch.setenv('SOSFUSE2_KSPLIT', '3')
    monkeypatch.setattr('sys.argv', ['precompile', '--cache', str(tmp_path), '--ray-limit', 'on'])
    jobs = []

    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, function, scheduled):
            jobs.extend(scheduled)
            return []

    monkeypatch.setattr(precompile, 'ProcessPoolExecutor', Pool)
    precompile.main()
    assert len(jobs) == len(precompile.SPECS)+1  # unsplit specializations and LUT only
    assert jobs[-1][0] == 'strato_lut_build'
    assert all(not {'FL_RAYSPLIT', 'WARM_RAYSPLIT'} & set(job[1]) for job in jobs)


def test_retired_sigma_invocation_and_fw1_preprocess():
    import subprocess
    import pytest
    from sosfuse2.kernel_config import RETIRED_FLAGS
    flags = json.loads(paths.CONFIGS.joinpath('C3.json').read_text())['kernel_flags']
    sigma = [f for f in flags if f not in ('REG_LEAN', 'TRANSPORT_RS2', 'SH_COMBINE', 'CELL_SPLIT2')]
    assert resolve_config(sigma+['STRATO'], W=1) == resolve_config(['STRATO'], W=1)
    legacy = tuple(sorted(RETIRED_FLAGS - {'REG_LEAN', 'TRANSPORT_RS2', 'SH_COMBINE', 'CELL_SPLIT2'}))
    with pytest.warns(UserWarning, match='W_KERNEL_FLAG_RETIRED'):
        old_call = resolve_config(legacy+('STRATO',), W=1)
    empty_call = resolve_config(('STRATO',), W=1)
    assert old_call == empty_call and not empty_call.c3_fw15
    source = (paths.PKG/'kernel.cu').read_text()
    testing.require_tool('clang')
    for role in ('transport', 'lut'):
        cfg = resolve_config(('STRATO',), role=role, W=1)
        cmd = ['clang', '-E', '-P', '-x', 'c++', '-DFW=1', '-DK_FIRST=0', '-DK_PHASE=0', '-DK_BOUNDARY=0']
        out = subprocess.run(cmd+['-D'+f for f in cfg.defines]+['-'], input=source, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
    with pytest.raises(ValueError, match='E_KERNEL_FLAG'):
        resolve_config(('REG_LEAN_TYPO',))


def test_sinv_reserved_argument_removed():
    k = object.__new__(Kernel)
    assert len(k.ksplit_pointer_args({})) == 3


def test_physical_scope_and_default_ksplit(monkeypatch):
    from sosfuse2.fused import ksplit_setting
    monkeypatch.delenv('SOSFUSE2_KSPLIT', raising=False)
    assert ksplit_setting('cupy', ('STRATO',), 16) == '3'
    assert ksplit_setting('cupy', (), 16) == '0'
    assert ksplit_setting('cupy', ('STRATO',), 24) == '0'
    assert not resolve_config(('STRATO',), W=1).ksplit_supported
    assert not resolve_config(()).ground_sh_supported
    assert not resolve_config(('STRATO', 'TROPO_MIE')).ground_sh_supported
    assert not resolve_config(('STRATO', 'EXP_SECTION_CLOCKS')).ground_sh_supported
    assert resolve_config(('STRATO',)).ground_sh_supported


def test_retired_names_do_not_change_cache_or_toolkey(monkeypatch):
    import pytest
    from sosfuse2.compile_cache import cache_key
    from sosfuse2.gate import toolkey
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
