"""CPU negative controls for the KSPLIT numerical contract."""
import numpy as np
import pytest
from .verify_tol import compare_tol, compare_chunks, tolerances


def test_roundoff_and_unused_sentinel():
    b = np.ones((2, 4, 3), 'f4'); a = b*(1+np.float32(1e-7))
    a[1].view('u4')[:] = 0xffffffff; b[1].view('u4')[:] = 0xffffffff
    out = compare_tol(a, b, np.array([True, False])[:, None, None], np.arange(3)[None, None, :])
    assert out['pass_'] and out['E_inf'] < 1 and out['violations'] == 0


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, np.uint32(0xffffffff).view('f4')])
@pytest.mark.parametrize('side', ['actual', 'reference', 'both'])
def test_valid_unwritten_and_nonfinite_fail(value, side):
    a = np.ones(4, 'f4'); b = a.copy()
    if side in ('actual', 'both'): a[2] = value
    if side in ('reference', 'both'): b[2] = value
    out = compare_tol(a, b, True, 0)
    assert not out['pass_'] and out['first'] == [2] and out['violations'] == 1 and out['E_inf'] == np.inf


def test_missing_slot_fails():
    assert not compare_tol(np.ones(3), np.ones(4), True, 0)['pass_']
    a = np.full(4, np.nan); a[:3] = 1.
    assert not compare_tol(a, np.ones(4), True, 0)['pass_']


def test_single_corruption_not_rescued_by_l2():
    b = np.ones(20000); a = b.copy(); a[123] += 1e-3
    out = compare_tol(a, b, True, 0)
    assert not out['pass_'] and out['violations'] == 1 and out['first'] == [123]
    assert out['groups']['0']['rel_l2'] < 1e-5 and out['E_inf'] > 1


@pytest.mark.parametrize('bad', [False, True])
def test_partition_invariant_scale_and_verdict(bad):
    b = np.array([1., 1e-8, 2., 1e-8, 3., 0.]); a = b.copy(); a[1] = 0.
    if bad: a[3] = 1e-3
    groups = np.array([0, 0, 1, 1, 0, 1]); full = compare_tol(a, b, True, groups)
    for cuts in ((0, 1, 6), (0, 2, 3, 5, 6), tuple(range(7))):
        parts = [(a[i:j], b[i:j], True, groups[i:j]) for i, j in zip(cuts, cuts[1:])]
        out = compare_chunks(parts)
        for key in ('pass_', 'E_inf', 'violations', 'first', 'max_abs', 'max_rel'):
            assert out[key] == full[key]
        for key in full['groups']:
            assert out['groups'][key]['scale'] == full['groups'][key]['scale']


def test_zero_scale_and_tolerance():
    assert compare_tol(np.zeros(3), np.zeros(3), True, 0, 0., 0.)['E_inf'] == 0.
    assert not compare_tol(np.array([1.]), np.zeros(1), True, 0, 0., 0.)['pass_']
    assert compare_tol(np.array([np.nan]), np.array([np.nan]), False, 0)['pass_']


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '-1', 'oops', ''])
@pytest.mark.parametrize('key', ['RTOL', 'ATOL'])
def test_env_invalid(value, key):
    with pytest.raises(ValueError, match='E_KSPLIT_VERIFY'):
        tolerances({'SOSFUSE2_KSPLIT_VERIFY_'+key: value})


def test_env_defaults_and_override():
    assert tolerances({}) == (1e-5, 1e-7)
    assert tolerances({'SOSFUSE2_KSPLIT_VERIFY_RTOL': '0', 'SOSFUSE2_KSPLIT_VERIFY_ATOL': '.1'}) == (0., .1)


def test_projection_failure_propagates_and_global_scale(capsys):
    from sosfuse2.fused import Kernel
    k = object.__new__(Kernel); k._verify_tolerances = (1e-5, 1e-7)
    ref = np.array([[[1.]], [[1e-8]]]); actual = ref.copy(); actual[1] = 0.
    def chunks(a):
        return [(i, i+1, a[i:i+1], np.ones((1, 1)), ref[i:i+1], np.ones((1, 1))) for i in range(2)]
    k.slim2_verify_proj(chunks(actual), {})
    actual[1] = 1e-3
    with pytest.raises(RuntimeError, match='E_KSPLIT_VERIFY'):
        k.slim2_verify_proj(chunks(actual), {})


def test_ray_missing_slot_fails():
    from sosfuse2.fused import Kernel
    k = object.__new__(Kernel); k.host = np.asarray; k._verify_tolerances = (1e-5, 1e-7)
    a = np.ones(8, 'f4'); b = a.copy(); a[4] = np.nan
    with pytest.raises(RuntimeError, match='E_KSPLIT_VERIFY'):
        k.slim2_verify_rays(a, b, 8, 1, {'first_ray': 20})


def test_aero_groups_use_whole_target_cell_and_metadata(capsys):
    from sosfuse2.fused import Kernel
    k = object.__new__(Kernel); k.host = np.asarray; k._verify_tolerances = (1e-5, 1e-7)
    pc = np.array([0, 2, 3]); mask = np.array([True, False, True])
    b = np.ones((3, 4, 2), 'f4'); b[1] = np.nan; b[0, 0, 0] = 1e-8
    a = b.copy(); a[0, 0, 0] = 0.
    k.ksplit_aero = b.ravel(); k._vprod_snap = a.ravel()
    k.vprod_verify(0, 24, 2, {'window': [0, 2], 'base': 0}, mask, pc, aw=2)
    # The same target cell can be in a smaller window, without changing its scale.
    k.vprod_verify(0, 16, 2, {'window': [0, 1], 'base': 0}, mask, pc, aw=2)
    a[2, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match='E_KSPLIT_VERIFY'):
        k.vprod_verify(0, 24, 2, {'window': [0, 2], 'base': 0}, mask, pc, aw=2)


def test_aero_window_log_is_bounded_and_vectorized(capsys, monkeypatch):
    import json
    from . import fused
    k = object.__new__(fused.Kernel); k.host = np.asarray; k._verify_tolerances = (1e-5, 1e-7)
    n, W = 2000, 15
    ref = np.ones((n, 4, W), 'f4'); actual = ref.copy()
    k.ksplit_aero = ref.ravel(); k._vprod_snap = actual.ravel()
    calls = []
    original = fused.compare_tol
    def compare(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(fused, 'compare_tol', compare)
    for fail in (False, True):
        if fail: actual[17, 2, 3] += 1e-3
        try:
            k.vprod_verify(0, ref.size, W, dict(window=[0, n], base=0), np.ones(n, bool), np.arange(n+1), aw=W)
        except RuntimeError:
            assert fail
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1 and len(lines[0].encode()) <= 4096
        row = json.loads(lines[0].removeprefix('KSPLIT '))
        assert row['cells'] == n and row['violations'] == int(fail)
        assert row['scale_min'] == row['scale_max'] == 1.
        assert 'groups' not in row
        assert ('failure_group' in row) == fail
        if fail: assert (row['max_cell'], row['max_wavelength']) == (17, 3)
    assert len(calls) == 2


@pytest.mark.parametrize('bad', [False, True])
def test_arithmetic_stage_invariance(bad):
    # Groups cross both chunk and arithmetic-piece boundaries; the largest
    # reference arrives last, so piece-local scales would reject slot 0.
    b = np.tile(np.array([1e-8, 0., 1., 2., 9.], 'f4'), 61).reshape(5, 61)
    a = b.copy(); a[0, 0] = 0.
    groups = np.arange(61)[None, :] % 7
    valid = np.ones(b.shape, bool); valid[1, 2] = False
    a[1, 2] = b[1, 2] = np.nan
    if bad:
        a[2, 11] += 1e-3; a[4, 17] = np.nan
    full = compare_tol(a, b, valid, groups)
    for size in (1, 7, 29, 64, 1000):
        out = compare_tol(a, b, valid, groups, stage_size=size)
        for key in ('pass_', 'E_inf', 'violations', 'first', 'peak', 'max_abs', 'max_rel', 'scale_min', 'scale_max'):
            assert out[key] == full[key]
        for key in full['groups']:
            assert out['groups'][key] == full['groups'][key]
        parts = [(a[i:i+1], b[i:i+1], valid[i:i+1], groups) for i in range(5)]
        split = compare_chunks(parts, stage_size=size)
        for key in ('pass_', 'E_inf', 'violations', 'max_abs', 'max_rel', 'groups'):
            assert split[key] == full[key]


def test_strided_broadcast_metadata_and_string_groups():
    a = np.arange(48., dtype='f4').reshape(6, 8).T
    b = a.copy(); a[3, 2] += .5
    valid = np.array([True, False, True, True, False, True])[None, :]
    keys = np.array(['x', 'y', 'x', 'z', 'z', 'y'])[None, :]
    out = compare_tol(a, b, valid, keys, stage_size=3)
    assert out['violations'] == 1 and out['first'] == [3, 2]
    assert out['failure_group']['key'] == 'x'
    assert out['groups']['x']['scale'] == 23.


@pytest.mark.parametrize('points', [16384, 32768])
def test_aero_host_peak_below_three_windows(points, monkeypatch, capsys):
    import tracemalloc
    from . import fused
    # CPU arrays stand in for device allocations, outside host accounting.
    W = 15
    device = np.ones((points, 4, W), 'f4')
    pc = np.arange(0, points+1, 256)
    valid = np.ones(points, bool); valid[::13] = False
    device[~valid] = np.nan
    monkeypatch.setattr(fused, 'VPROD_VERIFY_WINDOW', device.nbytes)
    monkeypatch.setattr(fused, 'VPROD_VERIFY_STAGE', 4093)
    k = object.__new__(fused.Kernel)
    k.host = lambda x: x.copy()  # include real D2H allocation, not np.asarray
    k.ksplit_aero = device.ravel(); k._verify_tolerances = (1e-5, 1e-7)
    tracemalloc.start()
    try:
        k._vprod_snap = device.ravel().copy()  # include the retained pinned snapshot
        k.vprod_verify(0, device.size, W, dict(window=[0, len(pc)-1], base=0), valid, pc, aw=W)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= 3*fused.VPROD_VERIFY_WINDOW, (peak, device.nbytes)
    capsys.readouterr()
    print(f'aero window={device.nbytes} stage=4093 peak={peak} ratio={peak/device.nbytes:.3f}')


def test_projection_snapshots_no_concatenation_peak(capsys, monkeypatch):
    import tracemalloc
    from .fused import Kernel
    device = np.ones((8192, 4, 15), 'f8')
    flux = np.ones((8192, 15), 'f8'); side_bytes = device.nbytes+flux.nbytes
    k = object.__new__(Kernel); k._verify_tolerances = (1e-5, 1e-7)
    # Account for all retained host chunks, as well as the comparison scratch.
    from . import fused
    monkeypatch.setattr(fused, 'VPROD_VERIFY_STAGE', 4093)
    tracemalloc.start()
    try:
        chunks = []
        for i in range(0, len(device), 1024):
            a = device[i:i+1024].copy(); b = a.copy()
            chunks.append((i, i+1024, a, flux[i:i+1024].copy(), b, flux[i:i+1024].copy()))
        k.slim2_verify_proj(chunks, {})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= 3*side_bytes, (peak, side_bytes)
    capsys.readouterr()
    print(f'projection side={side_bytes} stage=4093 peak={peak} ratio={peak/side_bytes:.3f}')


def test_aero_stage_boundaries_with_offset_and_empty_cells(monkeypatch):
    from . import fused
    W = 3
    pc = np.array([0, 3, 3, 8, 19, 25])
    wa, wb, base = 2, 4, 3
    ref = np.ones((22, 4, W), 'f4'); ref[5, 0, 1] = 1e-8
    actual = ref[:16].copy(); actual[5, 0, 1] = 0.; actual[11, 2, 0] += 1e-3
    valid = np.ones(22, bool); valid[8] = False
    k = object.__new__(fused.Kernel); k.host = np.asarray
    k._verify_tolerances = (1e-5, 1e-7); k.ksplit_aero = ref.ravel(); k._vprod_snap = actual.ravel()
    results = []
    k.verify_result = lambda event, result, info: results.append((result, info))
    cells = np.repeat(np.arange(wa, wb), np.diff(pc[wa:wb+1]))
    expected = compare_tol(actual, ref[:16], valid[:16, None, None], cells[:, None, None]*W+np.arange(W), details=False)
    for size in (1, 17, 61, 10000):
        monkeypatch.setattr(fused, 'VPROD_VERIFY_STAGE', size)
        k.vprod_verify(0, actual.size, W, dict(window=[wa, wb], base=base), valid, pc, aw=W)
        out, info = results[-1]
        assert out == expected
        assert (info['cell'], info['point'], info['corner'], info['wavelength']) == (3, 14, 2, 0)
        assert (info['max_cell'], info['max_wavelength']) == (3, 0)


def test_overflow_diagnostic_is_stage_invariant():
    a = np.array([0., 1e308, 0.]); b = np.array([0., -1e308, 0.])
    for size in (1, 2, 3):
        out = compare_tol(a, b, True, 0, r=10., stage_size=size)
        assert np.isnan(out['E_inf']) and out['peak'] == [1]
        assert out['pass_'] and out['violations'] == 0


def test_aero_pad_lanes_are_ignored():
    """Record rows are aero_w(W) floats wide (kernel.cu AERO_W); lanes >= W hold VPROD zeros or PROD garbage and must not be compared."""
    from . import fused
    W = 3; AW = fused.aero_w(W); assert AW == 4
    k = object.__new__(fused.Kernel); k.host = np.asarray; k._verify_tolerances = (1e-5, 1e-7)
    pc = np.array([0, 2, 5]); valid = np.ones(5, bool)
    ref = np.ones((5, 4, AW), 'f4'); ref[:, :, W:] = np.nan          # PROD never writes the pad lane
    actual = ref.copy(); actual[:, :, W:] = 0.                          # VPROD stores 0 there
    k.ksplit_aero = ref.ravel(); k._vprod_snap = actual.ravel()
    k.vprod_verify(0, actual.size, W, dict(window=[0, 2], base=0), valid, pc)
    actual[3, 1, 2] += 1e-3
    with pytest.raises(RuntimeError, match='E_KSPLIT_VERIFY'):
        k.vprod_verify(0, actual.size, W, dict(window=[0, 2], base=0), valid, pc)
