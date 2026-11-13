"""E6/G-H record path: pass is judged against fixtures/selftest_ref_simpson.npz (1e-5), the old
references only appear as historical_delta, and the old CPU-engine fixture stays as it was.
Run: python -m pytest -q src/sosfuse2/tests/test_selftest_record.py"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
from sosfuse2 import gate, testing, validate as v  # noqa: E402

try:
    from sosfuse2.numerical_checks import within_tol
except ImportError:  # standalone research harness
    from numerical_checks import within_tol

# Only cases that read the old CPU-engine fixture need research data; the pinned-record check below always runs.
needs_legacy = pytest.mark.skipif(
    not v.legacy_available(),
    reason='legacy rig/selftest_ref.npz: sky-sos checkout (SOSFUSE2_SKY_SOS_ROOT) with the deploy-copied fixture',
)


def synthetic_o4():
    """True when this session's O4 asset is the synthetic test fixture (testing.synthetic_o4, written by conftest)."""
    from sosfuse2 import minor_absorbers

    try:
        note = json.loads(minor_absorbers.o4_path().read_text()).get('source', {}).get('note', '')
    except (OSError, ValueError):
        return False
    return 'SYNTHETIC' in note


@pytest.fixture(scope='module')
def computed(tmp_path_factory):
    """One NumPy reduced selftest (about 7 s); each test re-judges a copy of it. The record holds values computed with the
    real O4 asset (HITRAN 2024 CIA, prepare_minor_absorbers); under the conftest's synthetic O4 the recomputed values
    differ from it (about 1.3e-2), so the value tests skip instead of judging another physics."""
    if synthetic_o4():
        pytest.skip(
            'record made with the real O4 asset (HITRAN 2024 CIA); this session uses the synthetic O4 test fixture'
        )
    return v.reduced_values(tmp_path_factory.mktemp('reduced'), 'numpy', 128)


def judge(tmp_path, monkeypatch, computed):
    monkeypatch.setattr(v, 'reduced_values', lambda *a: copy.deepcopy(computed))
    report, _ = v.reduced(tmp_path, 'numpy', 128)
    return report


def test_record_is_pinned_with_decision_and_provenance():
    digest = hashlib.sha256(v.RECORD.read_bytes()).hexdigest()
    assert digest == v.RECORD_SHA256
    meta = json.loads(v.RECORD_META.read_text())
    rec = v.load_record()
    assert meta['decision'] == rec['contract']['decision'] == v.DECISION == 'b2-s2-gatev2-20261105'
    assert meta['record']['sha256'] == digest and meta['numerical_files_equal_commit'] is True
    assert meta['numpy'] and meta['generated_commit'] and meta['command']['argv']
    for key in ('selftest_radiance', 'full_35_single', 'full_35_multiple', 'spectral_15_first', 'spectral_15_multiple'):
        assert np.isfinite(rec['arrays'][key]).all() and meta['record']['arrays'][key]['sha256'] == gate.sha(
            rec['arrays'][key]
        )


@needs_legacy
def test_pass_uses_record_and_old_fixture_delta_is_informational(tmp_path, monkeypatch, computed):
    report = judge(tmp_path, monkeypatch, computed)
    assert report['pass_'] and report['identity_match'] and report['record']['mismatch'] == []
    assert report['max_relative'] <= 1e-5
    h = report['historical_delta']
    assert h['blocking'] is False and h['max_relative'] > 1e-3  # Simpson prefix vs the old CPU engine (5.39e-2 at HEAD)
    with np.load(tmp_path / 'selftest_values.npz') as z:  # saved: record (pass) and old fixture (history) side by side
        assert np.array_equal(z['record'], v.load_record()['arrays']['selftest_radiance'])
        assert within_tol(z['reference'], v.legacy_selftest()['radiance'])


@needs_legacy
def test_record_sha_mismatch_fails(tmp_path, monkeypatch, computed):
    monkeypatch.setattr(v, 'RECORD_SHA256', '0' * 64)
    report = judge(tmp_path, monkeypatch, computed)
    assert not report['pass_'] and not report['identity_match'] and report['record']['mismatch'] == ['sha256']
    assert report['max_relative'] <= 1e-5  # values agree; the identity alone fails it


@needs_legacy
def test_record_values_decide_and_old_fixture_does_not(tmp_path, monkeypatch, computed):
    rec = v.load_record()
    arrays = dict(rec['arrays'], contract=np.asarray(json.dumps(rec['contract'], sort_keys=True)))
    # (a) record moved by 2e-5 (sha re-pinned to the altered file): fails on the value
    bad = tmp_path / 'record_moved.npz'
    np.savez(bad, **dict(arrays, selftest_radiance=arrays['selftest_radiance'] * (1 + 2e-5)))
    monkeypatch.setattr(v, 'RECORD', bad)
    monkeypatch.setattr(v, 'RECORD_SHA256', hashlib.sha256(bad.read_bytes()).hexdigest())
    (tmp_path / 'a').mkdir()
    report = judge(tmp_path / 'a', monkeypatch, computed)
    assert report['identity_match'] and not report['pass_'] and report['max_relative'] > 1e-5
    monkeypatch.undo()
    # (b) old fixture values replaced (its identity and delta change): the verdict does not
    fake = copy.deepcopy(computed)
    fake['legacy']['radiance'] = fake['legacy']['radiance'] * 3.0
    report = judge(tmp_path, monkeypatch, fake)
    assert report['pass_'] and report['historical_delta']['max_relative'] > 0.5


@needs_legacy
def test_gate_historical_reports_delta_and_old_engine_fixture_unchanged(tmp_path, monkeypatch):
    def child(cmd, log, env, cwd=None, timeout=None):
        d = Path(cmd[cmd.index('--out') + 1])
        d.mkdir(parents=True)
        (d / 'historical.json').write_text(
            json.dumps(
                dict(
                    pass_=True,
                    max_relative=1.4e-6,
                    identity_match=True,
                    record=dict(mismatch=[]),
                    historical_delta=dict(blocking=False, max_relative=5.39e-2),
                )
            )
        )
        return 0, [], None

    monkeypatch.setattr(gate, 'child', child)
    g = gate.historical(tmp_path)
    assert g['pass_'] and g['code'] is None and g['historical_delta'] == dict(blocking=False, max_relative=5.39e-2)
    # The historical selftest keeps the old fixture.
    assert (
        hashlib.sha256(v.LEGACY.read_bytes()).hexdigest()
        == v.LEGACY_SHA256
        == 'a48688cdc86bf6cb88950944840b88bc846a813864a54ad4672c9ee034ef03be'
    )
    boot = testing.sky_sos('rig/bootstrap_v3.sh').read_text().splitlines()
    assert (
        boot[84]
        == 'run_step selftest_reference "$PYTHON" -B rig/run_state.py --id normal --selftest --work-root "$work_root/reference_check" --threads 4'
    )
    v3_text = testing.sky_sos('rig/run_state_v3.py').read_text()
    assert "np.load(HERE/'selftest_ref.npz')" in v3_text
    assert 'selftest_ref_simpson' not in testing.sky_sos('rig/run_state.py').read_text() + v3_text
