"""CPU replay regression: one FW=15 chunk of the gate fixtures on the NumPy backend, without a GPU.

Two applies of the red chunk (bands 30-44) of FX-C3, FX-N and FX-AP are replayed and compared with the recorded NumPy
goldens; with a prepared O4 asset in the work root of the caller the other two chunks of FX-AP are compared as well.
The helper and its limits are in cpu_replay.py.

Run:
  python -B -m pytest -q -rs src/sosfuse2/tests/test_cpu_replay.py                   # all (about two minutes)
  python -B -m pytest -q src/sosfuse2/tests/test_cpu_replay.py -k "red and FX-AP"    # edit loop, one fixture
  SOSFUSE2_CPU_REPLAY_RTOL=1e-5 python -B -m pytest -q src/sosfuse2/tests/test_cpu_replay.py
      # tightened tolerance: a refactor check on a machine that reproduces the goldens to the bit
  SOSFUSE2_CPU_REPLAY_SLOW=1 python -B -m pytest -q src/sosfuse2/tests/test_cpu_replay.py -k albedo
      # FX-A (albedo 0.3) against its canonical CUDA record, about one more minute

Not covered:
  - a change made only in kernel.cu (the NumPy path does not execute it);
  - the full grid and the converged solve (reduced grid, two fixed applies);
  - 366-665 nm in a session with the synthetic O4 asset (the two O4 cases are skipped there).
"""

import json
import os
import shutil
import warnings

import numpy as np
import pytest

from sosfuse2.checks import gate
from sosfuse2.atmosphere import minor_absorbers
from sosfuse2.foundation import paths
from sosfuse2.tests.cpu_replay import canonical_slice, compare, golden_slice, replay

# Work root at import (collection) time: the session fixture of conftest.py has not yet pointed SOSFUSE2_WORK_ROOT at
# its synthetic root, so this is the root of the caller, where a prepared O4 asset may live.
ORIGINAL_ROOT = paths.work_root()
APPLIES = 2
RED = 2  # chunk of bands 30-44
DRIFT = 1e-5  # a passing comparison above this is reported as a warning


def rtol():
    """gate.RTOL, or the tighter SOSFUSE2_CPU_REPLAY_RTOL (loosening is refused)."""
    text = os.environ.get('SOSFUSE2_CPU_REPLAY_RTOL', '').strip()
    if not text:
        return gate.RTOL
    try:
        value = float(text)
    except ValueError:
        value = float('nan')
    if not 0 < value <= gate.RTOL:
        pytest.fail(f'E_TEST_CONFIG: SOSFUSE2_CPU_REPLAY_RTOL={text!r} must be a number in (0, {gate.RTOL:g}]')
    return value


def describe(row):
    return f"max_rel {row['max_rel']} worst {row['worst']} s_max {row['s_max']} band_max_rel {row['band_max_rel']}"


def judge(actual, ref):
    row = compare(actual, ref, rtol())
    assert row['pass_'], f"E_CPU_REPLAY_TOL: rtol {rtol():g} {row['reason'] or ''} {describe(row)}"
    if row['max_rel'] > DRIFT:
        warnings.warn(f"W_CPU_REPLAY_DRIFT: within rtol {rtol():g}, max_rel {row['max_rel']:.3e} worst {row['worst']}")


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """get(fx, chunk=RED): the replayed radiance, computed once per module; one host geometry cache for all replays."""
    base = tmp_path_factory.mktemp('cpu-replay')
    done = {}

    def get(fx, chunk=RED):
        if (fx, chunk) not in done:
            done[fx, chunk] = replay(fx, chunk, APPLIES, base / f'{fx}-{chunk}', base / 'geometry_cache')
        return done[fx, chunk]

    get.base = base
    return get


@pytest.fixture(scope='module')
def real_o4_root(tmp_path_factory):
    """A private work root holding the prepared (not synthetic) O4 asset of the caller's work root, or a skip."""
    source = ORIGINAL_ROOT / 'assets' / minor_absorbers.O4_NAME
    need = f'a prepared O4 asset in the work root of the caller: {minor_absorbers.PREPARE}'
    if not source.is_file():
        pytest.skip(f'needs {need}')
    note = (json.loads(source.read_text()).get('source') or {}).get('note') or ''
    if 'SYNTHETIC' in note:
        pytest.skip(f'the O4 asset of the work root is a synthetic fixture; needs {need}')
    root = tmp_path_factory.mktemp('real-o4')
    (root / 'assets').mkdir()
    shutil.copy2(source, root / 'assets' / source.name)
    held = paths.work_root() / 'iprt'
    if held.is_dir():
        shutil.copytree(held, root / 'iprt')
    return root


@pytest.mark.parametrize('fx', ['FX-C3', 'FX-N', 'FX-AP'])
def test_red_chunk_matches_golden(fx, runs):
    # Premise: O4 is zero from the lower edge of band 31 on, so bands 31-44 do not depend on which O4 asset the session
    # holds (band 30, 653.75-665.5 nm, does and is left out).
    o4 = minor_absorbers.load('o4')
    edge = gate.load_fixtures()['settings']['bin_edges_nm'][31]
    assert edge == 665.5 and not o4['values'][o4['wavelength_nm'] >= edge].any()
    judge(runs(fx)[..., 1:, :], golden_slice(fx, RED, APPLIES)[..., 1:, :])


def test_orders_finite_positive_increasing(runs):
    for fx in ('FX-C3', 'FX-N', 'FX-AP'):
        r = runs(fx)
        assert r.shape == (APPLIES, 4, 15, 15, 1), (fx, r.shape)
        assert np.isfinite(r).all() and (r > 0).all(), fx
        assert (r[1] >= r[0]).all(), fx  # a further scattering order only adds radiance


def test_comparison_rejects_other_fixture(runs):
    """Negative control: the comparison must tell the stratosphere fixture from the golden of the one without."""
    row = compare(runs('FX-C3')[..., 1:, :], golden_slice('FX-N', RED, APPLIES)[..., 1:, :], gate.RTOL)
    assert row['pass_'] is False and row['max_rel'] > 0.1, describe(row)


def test_thin_stratosphere_limit_and_column(tmp_path):
    from sosfuse2.checks import stratosphere_checks

    report, _, scope, _ = stratosphere_checks.thin_case(tmp_path)
    assert report['pass_'], report  # single scattering of a thin layer against an independent quadrature, 1 %
    grid = stratosphere_checks.profile_grid(scope)
    assert grid['pass_'], grid  # column of the cell-averaged profile, 1e-6


def test_stratosphere_source_normalised_no_absorption():
    from sosfuse2.checks import stratosphere_checks

    report = stratosphere_checks.energy()
    assert report['source_integral_pass_'], report


@pytest.mark.parametrize('chunk', [0, 1])
def test_real_o4_chunk_matches_golden(chunk, runs, real_o4_root):
    # The patched work root covers the replay only: the fixtures and the tests after this one stay on the session root.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('SOSFUSE2_WORK_ROOT', str(real_o4_root))
        actual = replay('FX-AP', chunk, APPLIES, runs.base / f'real-o4-{chunk}', runs.base / 'geometry_cache')
    judge(actual, golden_slice('FX-AP', chunk, APPLIES))


def test_albedo_boundary_matches_cuda_canonical(runs):
    if os.environ.get('SOSFUSE2_CPU_REPLAY_SLOW') != '1':
        pytest.skip('set SOSFUSE2_CPU_REPLAY_SLOW=1: the albedo 0.3 fixture FX-A takes about one more minute')
    actual = runs('FX-A')
    # Another backend, so the gate tolerance itself (never the tightened one).
    row = compare(actual[..., 1:, :], canonical_slice('FX-A', RED, APPLIES)[..., 1:, :], gate.RTOL)
    assert row['pass_'], f'E_CPU_REPLAY_TOL: rtol {gate.RTOL:g} {describe(row)}'
    assert (actual >= runs('FX-C3')).all()  # a reflecting ground only adds radiance
