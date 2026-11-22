"""Selection, rejection and transport for explicit bake options.

On a CUDA machine, the NumPy and CUDA tables of a small 200 km fixture are compared. It shares the example's upper air,
stratospheric layer, L, tolerance, history and ground order; its grid and ground albedo differ.
SOSFUSE2_OPTION_SMOKE_SLOW=1 python -B -m pytest -q -rs --pyargs sosfuse2.tests.test_option_combinations
also runs the 200 km settings example through run_state on CUDA and checks its records.
"""

from itertools import combinations, product
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2.atmosphere import density
from sosfuse2.jobs import run_state as rs
from sosfuse2.outputs.radiance_table import make_table
from sosfuse2.solve.ray_geometry import TransientGeometry, normal_state
from sosfuse2.solve.source_solver import Kernel, Solver, ksplit_setting
from sosfuse2.tests.test_extended_top import config, serial, small_solver  # noqa: F401


EXTENDED = dict(top_km=200, h_refine='b2a6', chi_refine_deg=[[88, 104.5], [104, 104.25]])
FACTORS = ((None, 200), (16, 24, 32), (None, 2e-4), ('device', 'host'), (None, 24))
CASES = (
    (None, 16, 2e-4, 'device', None),
    (None, 24, 2e-4, 'host', 24),
    (None, 32, None, 'device', 24),
    (200, 16, None, 'host', None),
    (200, 16, 2e-4, 'device', 24),
    (200, 24, None, 'device', None),
    (200, 32, 2e-4, 'host', None),
)


class BeforeTransport(Exception):
    """Stop after option validation, before allocating or iterating a field."""


def stop_before_transport(*args, **kwargs):
    raise BeforeTransport


@pytest.fixture(autouse=True)
def default_options(monkeypatch):
    for key in ('KSPLIT', 'KSPLIT_VERIFY', 'KERNEL_FLAGS', 'BLOCK_SIZE', 'GROUND_SH_L', 'ANDERSON_HISTORY'):
        monkeypatch.delenv('SOSFUSE2_' + key, raising=False)


def history_probe():
    kernel = SimpleNamespace(xp=SimpleNamespace(zeros=stop_before_transport))
    return Solver(SimpleNamespace(kernel=kernel, geo=SimpleNamespace(ns=1), W=1), None, prepared=True)


def test_pairwise_coverage():
    assert len(FACTORS) == 5 and all(len(set(values)) == len(values) > 1 for values in FACTORS)
    assert len(set(CASES)) == len(CASES) and all(len(row) == len(FACTORS) for row in CASES)
    for i, j in combinations(range(len(FACTORS)), 2):
        assert {(row[i], row[j]) for row in CASES} == set(product(FACTORS[i], FACTORS[j])), (i, j)


@pytest.mark.parametrize('top_km,L,tol,history,ground', CASES)
def test_explicit_options(tmp_path, monkeypatch, top_km, L, tol, history, ground):
    raw = dict(EXTENDED if top_km else {}, L=L)
    if tol is not None:
        raw['anderson_tol'] = tol
    monkeypatch.setenv('SOSFUSE2_ANDERSON_HISTORY', history)
    if ground is not None:
        monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', str(ground))
    state = dict(density=density.DEFAULTS)
    cfg = rs.resolve_settings(config(tmp_path, raw), state)
    domain = rs.execution_domain(cfg, state)
    assert (domain.top_km, cfg['nr'], cfg['nchi']) == ((200, 71, 109) if top_km else (80, 44, 75))
    if top_km:
        assert domain.upper_air == 'us76_neff'
    assert cfg['L'] == L
    assert rs.effective_tol(cfg) == (1e-3 if tol is None else tol)
    assert cfg.get('kernel', {}).get('ground_sh', {}).get('order', 0) == (ground or 0)
    solver = history_probe()
    with pytest.raises(BeforeTransport):
        solver.solve()
    assert solver.timings['anderson_history'] == history


@pytest.mark.parametrize('state', [{}, dict(density=density.LEGACY),
                                  density.with_profile({}, z_km=density.grid(161), table=np.ones((161, 2)), mask=1)],
                         ids=['missing-density', 'other-density', 'external-profile'])
def test_extended_top_requires_usstd(tmp_path, state):
    cfg = config(tmp_path, EXTENDED)
    with pytest.raises(ValueError, match=r'^E_SETTINGS_TOP:'):
        rs.resolve_settings(cfg, state)


@pytest.mark.parametrize('changes,reduced,code', [
    ({}, True, 'E_SETTINGS_TOP'),
    ({'nr': 70}, False, 'E_SETTINGS_GRID'),
    ({'nchi': 108}, False, 'E_SETTINGS_GRID'),
    ({'L': 20}, False, 'E_SETTINGS_FIXED'),
], ids=['reduced-top', 'height-count', 'chi-count', 'degree'])
def test_reject_settings(tmp_path, changes, reduced, code):
    with pytest.raises(ValueError, match=f'^{code}:'):
        config(tmp_path, EXTENDED | changes, reduced=reduced)


@pytest.mark.parametrize('mode', ['2', '3'])
def test_reject_explicit_split(monkeypatch, mode):
    monkeypatch.setenv('SOSFUSE2_KSPLIT', mode)
    with pytest.raises(ValueError, match=r'^E_KSPLIT_FLAGS:'):
        Kernel(backend='cupy', flags=('STRATO',), L=24)


@pytest.mark.parametrize('phase,strato,error', [
    ('enabled', {}, 'E_GROUND_SH_PHASE'),
    ('disabled', None, 'E_GROUND_SH_PHASE'),
    ('disabled', {}, None),
], ids=['mie', 'no-layer', 'disabled-phase'])
def test_ground_sh_phase_guard(monkeypatch, phase, strato, error):
    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '24')
    monkeypatch.setattr(density, 'attach', lambda *_: None)
    monkeypatch.setattr(np.polynomial.legendre, 'leggauss', stop_before_transport)
    state = dict(troposphere_phase=dict(status=phase, model='mie', optics_table='default'))
    solver = SimpleNamespace(ctx=None, fuse_state=state, wl=[550.], W=1, strato=strato)
    kernel = Kernel(L=24)
    expected = pytest.raises(ValueError, match=f'^{error}:') if error else pytest.raises(BeforeTransport)
    with expected:
        kernel.setup(solver, 12)


def test_reject_history(monkeypatch):
    monkeypatch.setenv('SOSFUSE2_ANDERSON_HISTORY', 'disk')
    with pytest.raises(ValueError, match=r'^E_ANDERSON_HISTORY:'):
        history_probe().solve()


def test_reject_ground_sh(tmp_path, monkeypatch):
    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '20')
    with pytest.raises(ValueError, match=r'^E_GROUND_SH_L:'):
        config(tmp_path)


def test_unsplit_above_degree_16(tmp_path):
    cfg = config(tmp_path, EXTENDED | {'L': 24})
    assert cfg['L'] == 24
    assert ksplit_setting('cupy', ('STRATO',), L=cfg['L'], W=15) == '0'
    kernel = Kernel(L=cfg['L'])
    assert (kernel.L, kernel.ksplit) == (24, False)
    assert kernel.ksplit_preflight(SimpleNamespace(), [])['effective'] == 0


def option_table(path, backend):
    """Converged table of the small 200 km fixture with the stratospheric layer, ground albedo 0.3, L24, tol 2e-4 and
    the history and ground order set by the caller. CUDA applies the ground order; NumPy convolves the ground exactly."""
    s, scope = small_solver(path, T=200, upper_air='us76_neff', backend=backend, wavelengths=np.array([550.]), L=24,
                            directions=(2, 4))
    solver = Solver(s, TransientGeometry(s, scope), prepared=True)
    field, stats = rs.solve_chunk(solver, 64, 2e-4)
    table = make_table(s, s.kernel.host(field), [-3., -6.], [67.5, 22.5], [45., 135.], log=lambda _: None)[0]
    assert s.strato is not None and s.fuse_state['parameters']['albedo'] > 0
    assert solver.timings['solve_worker']['L'] == 24 and solver.timings['solve_worker']['top_km'] == 200
    assert solver.timings['anderson_history'] == 'host' and stats['tolerance'] == 2e-4 and all(stats['converged'])
    return table, solver


def test_cuda_option_parity(tmp_path, monkeypatch):
    cupy = pytest.importorskip('cupy', reason='CuPy is unavailable')
    if not cupy.is_available():
        pytest.skip('CUDA device is unavailable')
    monkeypatch.setenv('SOSFUSE2_ANDERSON_HISTORY', 'host')
    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '24')
    reference, reference_solver = option_table(tmp_path / 'numpy', 'numpy')
    actual, solver = option_table(tmp_path / 'cuda', 'cupy')
    assert reference_solver.timings['ground_sh_order'] == 0
    assert solver.timings['ground_sh_order'] == 24
    assert (solver.kernel.L, solver.kernel.ksplit, solver.timings['ksplit']['effective']) == (24, False, 0)
    assert actual.shape == reference.shape == (2, 2, 2, 1)
    assert np.all(np.isfinite(reference)) and np.all(reference > 0)
    np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=0)


def settings_example_table(work):
    """Run the 200 km settings example of docs/MODEL.md through run_state on CUDA and check its records."""
    work.mkdir()
    state = normal_state()
    state.update(density=density.DEFAULTS, stratosphere={'status': 'enabled'})
    state['parameters'].update(albedo=.05, stratosphere_tau550=.01)
    raw = dict(EXTENDED, L=24, anderson_tol=2e-4, spectral_mode='legacy', wavelengths=1,
               wavelengths_nm=[550.], solar_elevation_deg=[-3., -6.], view_shape=[2, 2])
    rs.resolve_settings(config(work, raw), state)  # config writes settings.json; a wrong example fails here, not in run_state
    states = work / 'states.json'
    states.write_text(json.dumps(dict(states=[state])))
    args = ['--id', state['id'], '--states', str(states), '--settings-json', str(work / 'settings.json'),
            '--work-root', str(work), '--backend', 'cupy', '--threads', '1']
    command = [sys.executable, '-B', '-m', 'sosfuse2.jobs.run_state']
    log_path = work / 'run.log'
    with log_path.open('w') as log:
        result = subprocess.run([*command, *args], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(log_path.read_text()[-4000:])
    out = work / 'out' / state['id']
    meta = json.loads((out / 'meta.json').read_text())
    assert (meta['settings']['nr'], meta['settings']['nchi'], meta['solve_grid']['L']) == (71, 109, 24)
    assert (meta['settings']['top_km'], meta['settings']['upper_air'], meta['anderson_tol']) == (200, 'us76_neff', 2e-4)
    assert meta['timings'][0]['anderson_history'] == 'host'
    assert {timing['ground_sh_order'] for timing in meta['timings']} == {24}
    assert meta['settings']['kernel']['ground_sh']['order'] == 24
    assert meta['ksplit_effective'] == meta['ksplit']['effective'] == 0
    assert meta['parameters']['albedo'] == .05
    launch = meta['kernel']['launch']
    assert launch['launches'] > 0
    # Per-kind records exist only under a launch budget (Kernel.timed_launch); ksplit_effective above is the record.
    if launch['by_kind']:
        assert not any('_split' in kind or '_record' in kind for kind in launch['by_kind'])
    with np.load(out / 'table.npz') as result:
        return result['radiance']


@pytest.mark.skipif(
    os.environ.get('SOSFUSE2_OPTION_SMOKE_SLOW') != '1',
    reason='set SOSFUSE2_OPTION_SMOKE_SLOW=1: run_state solves the full 200 km grid of the settings example',
)
def test_cuda_settings_example(tmp_path, monkeypatch):
    cupy = pytest.importorskip('cupy', reason='CuPy is unavailable')
    if not cupy.is_available():
        pytest.skip('CUDA device is unavailable')
    monkeypatch.setenv('SOSFUSE2_ANDERSON_HISTORY', 'host')
    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '24')
    table = settings_example_table(tmp_path / 'cuda')
    assert table.shape == (2, 2, 2, 1)
    assert np.all(np.isfinite(table)) and np.all(table > 0)
