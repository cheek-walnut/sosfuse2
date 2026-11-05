"""Default repeat tolerance and opt-in deterministic compile identity."""
import numpy as np
import pytest
from . import gate, kernel_config, compile_cache


def test_mode_and_cache_key(monkeypatch):
    monkeypatch.delenv('SOSFUSE2_DETERMINISTIC', raising=False)
    assert kernel_config.deterministic() is False
    off = compile_cache.cache_key('test', {})
    monkeypatch.setenv('SOSFUSE2_DETERMINISTIC', '1')
    assert kernel_config.deterministic() is True
    assert compile_cache.cache_key('test', {}) != off
    monkeypatch.setenv('SOSFUSE2_DETERMINISTIC', 'true')
    with pytest.raises(ValueError, match='E_DETERMINISTIC'):
        compile_cache.cache_key('test', {})


def test_repeat_modes(monkeypatch):
    a = dict(radiance=np.ones((2, 3)), source=np.ones((2, 3)))
    b = {k: v*(1+5e-6) for k, v in a.items()}
    monkeypatch.setenv('SOSFUSE2_DETERMINISTIC', '0')
    r = gate.repeat_run(a, b)
    assert r['pass_'] and 0 < r['repeat_max_rel'] < 1e-5
    monkeypatch.setenv('SOSFUSE2_DETERMINISTIC', '1')
    assert not gate.repeat_run(a, b)['pass_']
    assert gate.repeat_run(a, a)['pass_']


@pytest.mark.parametrize('value', [1e-4, np.nan, np.inf])
def test_repeat_rejects_bad_values(value):
    assert not gate.repeat_arrays(np.array([value]), np.zeros(1))['pass_']


def test_repeat_rejects_layout():
    assert not gate.repeat_arrays(np.ones(2), np.ones((1, 2)))['pass_']
    assert not gate.repeat_arrays(np.ones(2, 'f4'), np.ones(2, 'f8'))['pass_']


def test_solve_history_is_diagnostic_by_default(monkeypatch):
    a = dict(radiance=np.ones((2, 3)), source=np.ones((2, 3)), table=np.ones(3),
             solve_source=np.ones(3), solve=dict(evaluations=4, source_solve_sha256='cold'))
    b = dict(a, solve=dict(evaluations=5, source_solve_sha256='warm'))
    monkeypatch.setenv('SOSFUSE2_DETERMINISTIC', '0')
    r = gate.repeat_run(a, b)
    assert r['pass_'] and r['evaluations_warning'] == 'W_REPEAT_EVALUATIONS_DIFFER'
    assert r['source_solve'] == ['cold', 'warm']
    monkeypatch.setenv('SOSFUSE2_DETERMINISTIC', '1')
    assert not gate.repeat_run(a, b)['pass_']
