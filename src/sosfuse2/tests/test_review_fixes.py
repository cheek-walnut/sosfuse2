"""Checks that used to pass without comparing, and settings that two code paths read differently.

Each test names the condition that went unnoticed: a comparison with no sample, a control that is not a number, an
option pair that overwrote a result, a cache key that left out a file, a kernel looked up under another pass's key.
"""

import inspect
import os
import types
from pathlib import Path

import numpy as np
import pytest

from sosfuse2.foundation import content_identity, paths


@pytest.mark.parametrize('values', [{}, {'value': np.empty((2, 0))}], ids=['no-keys', 'no-samples'])
def test_empty_baseline_pair_is_not_a_pass(tmp_path, values):
    from sosfuse2.checks import stratosphere_checks

    current, baseline = tmp_path / 'current', tmp_path / 'baseline'
    for folder in (current, baseline):
        folder.mkdir()
        for name in ('selftest_values.npz', 'full_35_values.npz'):
            np.savez(folder / name, **values)
    report = stratosphere_checks.zero_bytes(current, baseline)
    assert report['pass_'] is False and [row['pass_'] for row in report['files'].values()] == [False, False]
    for folder in (current, baseline):
        for name in ('selftest_values.npz', 'full_35_values.npz'):
            np.savez(folder / name, value=np.arange(3.0))
    assert stratosphere_checks.zero_bytes(current, baseline)['pass_'] is True


def mie_rows(control_error, control='hg_strato_on_C3'):
    row = dict(first=1, moment_l2_relative=0.0, ground_flux_max_relative=0.0)
    return [dict(row, case='mie_strato_off_C2', ray_max_relative=1.0), dict(row, case=control, ray_max_relative=control_error)]


def test_mie_judgement_needs_a_finite_control_and_names_it():
    from sosfuse2.tests import test_tropo_mie as mie

    assert mie.judge(mie_rows(float('inf')))['pass_'] is False
    assert mie.judge(mie_rows(float('nan')))['pass_'] is False
    passed = mie.judge(mie_rows(1.0))
    assert passed['pass_'] is True and passed['checks'][0]['control'] == 'hg_strato_on_C3'
    twin = mie.judge(mie_rows(1.0, control='hg_strato_off_C2'))
    assert twin['checks'][0]['control'] == 'hg_strato_off_C2'


def test_mie_gpu_and_recheck_are_exclusive(tmp_path, monkeypatch, capsys):
    from sosfuse2.tests import test_tropo_mie as mie

    monkeypatch.setattr(mie, 'gpu', lambda *a: pytest.fail('the GPU cases must not run'))
    monkeypatch.setattr('sys.argv', ['tropo_mie', '--out', str(tmp_path / 'out'), '--gpu', '--recheck', str(tmp_path / 'old.json')])
    with pytest.raises(SystemExit) as stop:
        mie.main()
    assert stop.value.code == 2 and '--gpu and --recheck' in capsys.readouterr().err


@pytest.mark.parametrize(
    'message, skipped',
    [
        ('E_IPRT_OFFLINE: afglus.dat missing from /work/iprt', True),
        ('child failed:\nTraceback (most recent call last):\n  File "x.py", line 1\nRuntimeError: E_IPRT_OFFLINE: afglus.dat missing', True),
        ('the table differs; note that E_IPRT_OFFLINE: is the code of the offline stop', False),
        ('summary:\nRuntimeError: E_IPRT_OFFLINE: handled earlier\nE_OTHER: the failure', False),
        ('child failed:\nTraceback (most recent call last):\n  File "x.py", line 1\nValueError: E_OTHER: not the inputs', False),
        ('E_OTHER: after a handled stop', False),
    ],
)
def test_offline_stop_is_the_error_itself_or_the_error_line_of_a_traceback(message, skipped):
    from sosfuse2.tests import conftest

    assert conftest.offline_stop(message) is skipped


def test_a6_verdict_reads_the_reference_comparison_and_the_sample_count():
    from sosfuse2.tests import test_a6_flags

    def report(**ground):
        lane = dict(lane_vs_original_within_tol=True, original_vs_reference_within_tol=True, samples_checked=6)
        return dict(SH_AUX=dict(layout=dict(a=True), transport_within_tol=dict(b=True)), LANE_GROUND=dict(lane, **ground),
                    LM_LUT=dict(exact=True))

    assert test_a6_flags.verdict(report()) is True
    assert test_a6_flags.verdict(report(original_vs_reference_within_tol=False)) is False
    assert test_a6_flags.verdict(report(samples_checked=0)) is False
    assert test_a6_flags.verdict(report(lane_vs_original_within_tol=False)) is False


def test_a2_command_exit_code_follows_the_verdict():
    from sosfuse2.checks import testkit

    text = (paths.PKG / 'tests/test_a2_csr.py').read_text()
    assert "raise SystemExit(0 if main(sys.argv[1])['pass'] else 1)" in text, testkit.ENV


def test_attach_geometry_key_covers_the_files_of_the_ground_preparation(monkeypatch):
    """Every source file that defines a function of precompute/ground_parallel._cell changes the key."""
    from sosfuse2.atmosphere import stratosphere
    from sosfuse2.rays import incoming_quadrature, shell_segments
    from sosfuse2.tests.test_identity_caches import solver

    s, profile = solver(), {'center': 21.0, 'sigma': 3.0, 'norm': 1.0}
    real = content_identity.file_sha
    base = stratosphere._geometry_key(s, None, profile)
    used = (incoming_quadrature.incoming, shell_segments.frame, shell_segments.boundaries, shell_segments.SunGeometry.lookup)
    names = {str(Path(inspect.getsourcefile(f)).resolve().relative_to(paths.PKG)) for f in used}
    assert names == {'rays/incoming_quadrature.py', 'rays/shell_segments.py'}
    for name in sorted(names | {'precompute/ground_parallel.py'}):

        def changed(path, *a, _name=name, **k):
            value = real(path, *a, **k)
            return 'f' * 64 if str(Path(path).resolve().relative_to(paths.PKG)) == _name else value

        monkeypatch.setattr(stratosphere.source_identity, 'file_sha', changed)
        assert stratosphere._geometry_key(s, None, profile) != base, name
        monkeypatch.setattr(stratosphere.source_identity, 'file_sha', real)


class FakeRaw:
    attributes = {}

    def __init__(self, code, name, options=(), backend=None):
        self.name, self.options, self.max_dynamic_shared_size_bytes = name, options, 0

    def compile(self):
        pass


def test_warm_ray_range_projection_uses_the_warm_kernel(monkeypatch):
    """WARM_RAYSPLIT with a projected single launch: fl_project comes from the kernel compiled for the warm pass."""
    from sosfuse2.solve.source_solver import Kernel

    monkeypatch.setenv('SOSFUSE2_KERNEL_FLAGS', 'STRATO,WARM_RAYSPLIT')
    k = Kernel(backend='numpy')
    runtime = types.SimpleNamespace(getDevice=lambda: 0, getDeviceProperties=lambda d: {'sharedMemPerBlock': 1 << 30})
    k.xp = types.SimpleNamespace(RawKernel=FakeRaw, cuda=types.SimpleNamespace(runtime=runtime))
    assert k.config().split_warm
    for first in (0, 1):
        for boundary in (0, 1):
            k.raw.clear()
            k.__dict__.pop('raw_options', None)
            k.raw_kernel(15, first, 0, boundary, ray_limit=True)
            project = k.fl_project_kernel(15, boundary, first=first)
            assert project.name == 'fl_project'
            assert f'-DK_FIRST={first}' in project.options and f'-DK_BOUNDARY={boundary}' in project.options
    # a first-light kernel compiled earlier is not borrowed for the warm pass
    k.raw.clear()
    k.__dict__.pop('raw_options', None)
    k.raw_kernel(15, 1, 0, 1, ray_limit=True)
    with pytest.raises(KeyError):
        k.fl_project_kernel(15, 1, first=False)


@pytest.mark.parametrize('method', ['gl24', 'table-v1', 'exact'])
@pytest.mark.parametrize('density', [True, False])
def test_ground_boundary_columns_follow_the_column_method(monkeypatch, method, density):
    """NumPy ground boundary: the walk for gl24, the selected column method otherwise, as in the CUDA host preparation."""
    from sosfuse2.atmosphere import density as density_module, stratosphere
    from sosfuse2.backends import numpy_reference
    from sosfuse2.rays import column_table

    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', method)
    calls = []
    edges = np.ones((3, 2))

    def integrate(p, d, old, *, legacy=False, optics=None, profile=None, edges=None):
        assert p.shape == d.shape == (3, 3) and edges is g['edges'] and not legacy
        calls.append(('method', 'density' if optics is not None else 'strato', old, optics if optics is not None else profile))
        return None, None if optics is None else np.full((3, 4), 7.0), None if profile is None else np.full(3, 7.0)

    def correct_columns(p, d, base, old, optics, n=24, *, end=None):
        calls.append(('walk', 'density', n, old, optics))
        return np.full(np.shape(base), 5.0)

    def columns(p, d, old, profile, n=24):
        calls.append(('walk', 'strato', n, old, profile))
        return np.full(len(d), 5.0)

    monkeypatch.setattr(column_table, 'integrate', integrate)
    monkeypatch.setattr(density_module, 'correct_columns', correct_columns)
    monkeypatch.setattr(stratosphere, 'columns', columns)
    base = np.ones((3, 5))
    g = dict(p=np.array([0.0, 0.0, 6371.1]), d=np.eye(3), end_columns=base, edges=edges)
    c = dict(strato_old='old-s', strato_profile='profile')
    if density:
        c['density_optics'] = dict(old='old-d', mask=1)  # air selected, ozone not
    endcol, stcol = numpy_reference.boundary_end_columns(g, c, True)
    if method == 'gl24':
        expected = ([('walk', 'density', 24, 'old-d', c['density_optics'])] if density else []) + [('walk', 'strato', 24, 'old-s', 'profile')]
        assert calls == expected and (stcol == 5).all() and stcol.dtype == np.float32
    else:
        expected = ([('method', 'density', 'old-d', c['density_optics'])] if density else []) + [('method', 'strato', 'old-s', 'profile')]
        assert calls == expected and (stcol == 7).all() and stcol.dtype == np.float32
        if density:  # air (mask bit 0), NO2 and O4 replaced; aerosol and the unselected ozone kept
            assert endcol.dtype == np.float32 and endcol[0].tolist() == [7.0, 1.0, 1.0, 7.0, 7.0]
    if not density:
        assert endcol is base  # no density optics: the cached end columns, untouched
    assert (base == 1).all()
    assert numpy_reference.boundary_end_columns(g, dict(c), False)[1] is None


def test_ground_boundary_columns_of_real_rays_agree_between_methods(tmp_path, monkeypatch):
    """Real rays to the ground through the stratospheric layer, with the prepared-segments stand-in of the queue
    service as `old`: the table method and the exact integral agree with each other to 1e-4 of the column."""
    from sosfuse2.atmosphere import stratosphere
    from sosfuse2.backends import numpy_reference
    from sosfuse2.jobs.pipeline_execution import PreparedSegments
    from sosfuse2.rays import column_table, shell_segments

    monkeypatch.setenv('SOSFUSE2_COLUMN_TABLE_CACHE', str(tmp_path))
    old = types.SimpleNamespace(
        probe=types.SimpleNamespace(boundaries=shell_segments.boundaries), geometry_segments=None, domain=None
    )
    del old.domain
    radius = 6371.0
    p = np.array([[0.0, 0.0, radius + h] for h in (5.0, 20.0, 40.0)])
    d = np.array([[np.sin(a), 0.0, np.cos(a)] for a in np.deg2rad([180.0, 150.0, 120.0])])
    edges = shell_segments.boundaries(p, d)
    assert np.allclose(np.linalg.norm(p + edges[:, -1:] * d, axis=1), radius, atol=1e-6)
    profile = stratosphere.profile_config(dict(stratosphere={}, parameters=dict(stratosphere_tau550=0.003)))
    g = dict(p=p, d=d, end_columns=np.zeros((3, 5)), edges=edges)
    reference = None
    for method in ('exact', 'table-v1'):
        monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', method)
        column_table._MEMO.clear()
        prepared = PreparedSegments.__new__(PreparedSegments)  # as in the queue service: no probe, only the domain
        prepared.domain, prepared.rows = column_table.Domain(), {}
        _, stcol = numpy_reference.boundary_end_columns(g, dict(strato_old=prepared, strato_profile=profile), True)
        assert stcol.dtype == np.float32 and (stcol > 0).all()
        reference = stcol if reference is None else reference
        np.testing.assert_allclose(stcol, reference, rtol=1e-4)
    column_table._MEMO.clear()


@pytest.mark.parametrize('refine', ['none', 'b2a6'])
def test_stratospheric_source_density_under_an_extended_top(refine):
    """What an extended top changes below 80 km: nothing for the bundled layer, the top node for a layer whose mass
    reaches the last cell below 80 km. The cell sum stays 1 in both cases (docs/MODEL.md states the limit)."""
    from sosfuse2.atmosphere import stratosphere as S
    from sosfuse2.foundation import top
    from sosfuse2.jobs.run_state import refined_heights
    from sosfuse2.rays.shell_segments import grids

    def profile(center, sigma, T):
        state = dict(stratosphere=dict(center_km=center, sigma_km=sigma), parameters=dict(stratosphere_tau550=0.003))
        p = S.profile_config(state)
        if T > 80:
            p.update(end_km=top.resolve(dict(top_km=T, upper_air='vacuum'), 0.5).lower_end)
        return p

    h80 = np.asarray(refined_heights(grids(1)[0], refine), float)
    h120 = np.r_[h80, 90.0, 100.0, 110.0, 120.0]
    n = len(h80)
    for center, sigma, same in ((21.0, 3.0, True), (77.0, 1.0, False)):
        low, _ = S.cell_density(h80, profile(center, sigma, 80))
        high, edges = S.cell_density(h120, profile(center, sigma, 120))
        assert abs(float(high @ np.diff(edges)) - 1) <= 1e-6
        assert (high[n:] == 0).all()
        np.testing.assert_allclose(high[: n - 1], low[: n - 1], rtol=1e-12, atol=0)
        if same:
            assert high[n - 1] == low[n - 1] == 0
        else:
            assert 0 < high[n - 1] < low[n - 1]


def test_query_restores_the_active_lanes_after_an_error():
    from sosfuse2.solve import engine_core

    class PackFailed(Exception):
        pass

    def pack(*_):
        raise PackFailed

    up, east = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    probe = types.SimpleNamespace(frame=lambda *_: (up, up, east))
    shell = types.SimpleNamespace(old=types.SimpleNamespace(probe=probe), OBS=0.0)
    solver = types.SimpleNamespace(
        ctx=types.SimpleNamespace(b=types.SimpleNamespace(sh=shell, R=1.0)),
        W=1, ms=None, active=np.array([1, 0]), geo=types.SimpleNamespace(pack=pack), kernel=None,
    )
    with pytest.raises(PackFailed):
        engine_core.Solver.query(solver, 0.0, 10.0, 0.0)
    assert solver.active.tolist() == [1, 0]


def test_column_workers_compare_every_package_module_their_task_imports():
    import ast

    from sosfuse2.precompute import column_parallel

    watched = set(column_parallel.source_hashes())
    task = ast.parse(inspect.getsource(column_parallel._run))
    imported = {
        'atmosphere/' + node.module.rsplit('.', 1)[-1] + '.py'
        for node in ast.walk(task)
        if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module.startswith('atmosphere.')
    }
    assert imported == {'atmosphere/density.py', 'atmosphere/stratosphere.py'}
    assert imported | {'precompute/column_parallel.py'} <= watched


def test_two_threads_fetching_one_input_do_not_share_a_temporary_file(tmp_path, monkeypatch):
    import hashlib
    import threading

    from sosfuse2.asset_tools import iprt

    data = b'profile'
    both = threading.Barrier(2)

    class Reply:
        def __enter__(self):
            both.wait(timeout=30)  # both callers have missed the cache before either writes
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return data

    monkeypatch.setattr(iprt, 'FILES', {'input.dat': hashlib.sha256(data).hexdigest()})
    monkeypatch.setattr(iprt, 'cache_dir', lambda: tmp_path)
    monkeypatch.setattr(iprt.urllib.request, 'urlopen', lambda *_, **__: Reply())
    made = []
    replace = os.replace

    def record(source, target):
        made.append(Path(source).name)
        replace(source, target)

    monkeypatch.setattr(iprt.os, 'replace', record)
    errors = []

    def fetch():
        try:
            iprt.fetch('input.dat', offline=False)
        except Exception as exc:  # noqa: BLE001 - the test reports whatever a caller raised
            errors.append(exc)

    threads = [threading.Thread(target=fetch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and len(set(made)) == 2 and (tmp_path / 'input.dat').read_bytes() == data
    assert sorted(path.name for path in tmp_path.iterdir()) == ['input.dat']


def test_record_cell_range_stops_the_sort_before_the_scatter_kernel():
    from sosfuse2.solve import ksplit

    launched = []

    def get_function(name):
        def kernel(_grid, _block, arguments, **_):
            launched.append(name)
            if name == 'vp_count':
                arguments[-1][0] = 1  # the kernel reports a record whose cell is outside the grid

        return kernel

    solver = types.SimpleNamespace(xp=np, vprod_plan_module=lambda: types.SimpleNamespace(get_function=get_function))
    kb = dict(chunks=[(0, 1)], pcum=np.array([0, 2]), co=np.array([0, 1]), plans={})
    with pytest.raises(ValueError, match='E_VPROD_CELL_RANGE'):
        ksplit.KsplitMixin.vprod_build_plan(solver, kb, 0, None, np.zeros(8, 'i4'), 2, 2)
    assert launched == ['vp_count']


def test_pool_source_hashes_cover_all_shared_worker_entry_points():
    from sosfuse2.precompute import column_parallel

    watched = column_parallel.source_hashes()
    assert {
        'rays/ray_columns.py', 'precompute/sun_parallel.py', 'precompute/ground_parallel.py',
        'solve/ray_geometry_parallel.py',
    } <= watched.keys()


@pytest.mark.parametrize('offset', [0, 1])
def test_large_record_sort_checks_last_non_sentinel_cell(offset):
    from sosfuse2.solve import ksplit

    class SplitReached(Exception):
        pass

    nh, nc = 129, 96
    ncell = nh * nc
    c0cp = np.array([-1, 0, ncell - nc - 2 + offset, 0], 'i4')

    def get_function(name):
        if name == 'vp_split':
            raise SplitReached
        assert name == 'vp_keys'

        def keys(_grid, _block, args):
            n, base, stride, out = args
            for i in range(n):
                cell = int(base[2 * (i // 4)])
                cell = 0xffffffff if cell < 0 else cell + (0, 1, stride, stride + 1)[i % 4]
                out[i] = (int(cell) << 32) | i

        return keys

    solver = types.SimpleNamespace(xp=np, vprod_plan_module=lambda: types.SimpleNamespace(get_function=get_function))
    kb = {'chunks': [(0, 1)], 'pcum': [0, 2], 'co': [0, 1]}
    expected = pytest.raises(ValueError, match='E_VPROD_CELL_RANGE') if offset else pytest.raises(SplitReached)
    with expected:
        ksplit.KsplitMixin.vprod_build_plan(solver, kb, 0, None, c0cp, nh, nc)
