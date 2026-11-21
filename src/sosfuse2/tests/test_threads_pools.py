"""run_state --threads: default 2; an explicit value also sizes the unset or blank preparation pools in every mode,
for the call only."""

import os
import sys
from unittest import mock

import pytest

from sosfuse2.jobs import pipeline, run_state

MODES = {
    'solve': ([], 'run'),
    'geometry-only': (['--geometry-only'], 'geometry_only'),
    'extra-elevations': (['--extra-elevations=-3.5', '--source-out-name', 'src'], 'run_extra'),
}


def test_pool_names_match_pipeline():
    assert run_state.POOL_WORKER_ENV == pipeline.PREP_WORKER_ENV


def test_explicit_threads_fill_only_unset_pools():
    environ = {'SOSFUSE2_COLUMN_WORKERS': '6', 'SOSFUSE2_SUN_WORKERS': '  '}
    run_state._bound_pool_workers(3, environ)
    assert environ == {
        'SOSFUSE2_COLUMN_WORKERS': '6',
        'SOSFUSE2_SUN_WORKERS': '3',
        'SOSFUSE2_GROUND_WORKERS': '3',
        'SOSFUSE2_GEOMETRY_WORKERS': '3',
    }


def call_main(monkeypatch, tmp_path, extra, entry, preset=None, fail=False, seen=None):
    """Run main() up to the mode call; return (a.threads, pool variables seen by the mode, pool variables after).

    `seen['after']` holds the pool variables right after main() returns or raises, before the test's own restore."""
    seen = {} if seen is None else seen

    def record(a):
        seen['threads'] = a.threads
        seen['pools'] = {name: os.environ.get(name) for name in run_state.POOL_WORKER_ENV}
        if fail:
            raise RuntimeError('E_TEST_MODE_FAILED')

    for name in ('run', 'geometry_only', 'run_extra'):
        monkeypatch.setattr(run_state, name, lambda a, n=name: pytest.fail(f'unexpected mode {n}'))
    monkeypatch.setattr(run_state, entry, record)
    argv = ['run_state', '--id', 'clear_34', '--work-root', str(tmp_path), '--backend', 'numpy', '--reduced', *extra]
    monkeypatch.setattr(sys, 'argv', argv)
    with mock.patch.dict(os.environ):
        for name in run_state.POOL_WORKER_ENV:
            os.environ.pop(name, None)
        os.environ.update(preset or {})
        try:
            run_state.main()
        finally:
            after = {name: os.environ.get(name) for name in run_state.POOL_WORKER_ENV}
            seen['after'] = after
    return seen['threads'], seen['pools'], after


@pytest.mark.parametrize('mode', MODES)
def test_omitted_threads_keep_pools_and_blas_two(monkeypatch, tmp_path, mode):
    extra, entry = MODES[mode]
    threads, pools, _ = call_main(monkeypatch, tmp_path, extra, entry)
    assert threads == 2
    assert pools == dict.fromkeys(run_state.POOL_WORKER_ENV)


@pytest.mark.parametrize('mode', MODES)
def test_explicit_threads_bound_unset_pools_in_every_mode(monkeypatch, tmp_path, mode):
    extra, entry = MODES[mode]
    preset = {'SOSFUSE2_COLUMN_WORKERS': '5', 'SOSFUSE2_SUN_WORKERS': ''}
    threads, pools, after = call_main(monkeypatch, tmp_path, [*extra, '--threads', '1'], entry, preset)
    assert threads == 1
    assert pools == {
        'SOSFUSE2_COLUMN_WORKERS': '5',
        'SOSFUSE2_SUN_WORKERS': '1',
        'SOSFUSE2_GROUND_WORKERS': '1',
        'SOSFUSE2_GEOMETRY_WORKERS': '1',
    }
    # The limit is for this call only: the previous values (set, blank, unset) come back.
    assert after == {'SOSFUSE2_COLUMN_WORKERS': '5', 'SOSFUSE2_SUN_WORKERS': '',
                     'SOSFUSE2_GROUND_WORKERS': None, 'SOSFUSE2_GEOMETRY_WORKERS': None}


@pytest.mark.parametrize('mode', MODES)
def test_pool_variables_restored_after_a_failing_mode(monkeypatch, tmp_path, mode):
    extra, entry = MODES[mode]
    seen = {}
    with pytest.raises(RuntimeError, match='E_TEST_MODE_FAILED'):
        call_main(monkeypatch, tmp_path, [*extra, '--threads', '1'], entry, {'SOSFUSE2_SUN_WORKERS': '7'}, fail=True, seen=seen)
    assert seen['pools'] == {'SOSFUSE2_COLUMN_WORKERS': '1', 'SOSFUSE2_SUN_WORKERS': '7',
                             'SOSFUSE2_GROUND_WORKERS': '1', 'SOSFUSE2_GEOMETRY_WORKERS': '1'}
    assert seen['after'] == {'SOSFUSE2_COLUMN_WORKERS': None, 'SOSFUSE2_SUN_WORKERS': '7',
                             'SOSFUSE2_GROUND_WORKERS': None, 'SOSFUSE2_GEOMETRY_WORKERS': None}


def test_pool_limit_restores_on_error():
    environ = {'SOSFUSE2_COLUMN_WORKERS': '6', 'SOSFUSE2_SUN_WORKERS': ' '}
    with pytest.raises(ValueError):
        with run_state._pool_limit(2, environ):
            assert environ['SOSFUSE2_GROUND_WORKERS'] == '2' and environ['SOSFUSE2_SUN_WORKERS'] == '2'
            raise ValueError
    assert environ == {'SOSFUSE2_COLUMN_WORKERS': '6', 'SOSFUSE2_SUN_WORKERS': ' '}


def test_direct_run_failing_early_leaves_pool_variables(monkeypatch):
    # run(a) itself never touches the pool variables, also when it stops on an option check.
    import argparse

    a = argparse.Namespace(id='clear_34', out_name=None, backend='numpy', polarization_model='preserving', threads=1)
    with mock.patch.dict(os.environ):
        for name in run_state.POOL_WORKER_ENV:
            os.environ.pop(name, None)
        os.environ['SOSFUSE2_SUN_WORKERS'] = ''
        before = {name: os.environ.get(name) for name in run_state.POOL_WORKER_ENV}
        with pytest.raises(RuntimeError, match='E_UNSUPPORTED_POL_MODEL'):
            run_state.run(a)
        assert {name: os.environ.get(name) for name in run_state.POOL_WORKER_ENV} == before


@pytest.mark.parametrize('value', ['0', '-2'])
def test_non_positive_threads_rejected(monkeypatch, tmp_path, value):
    with pytest.raises(SystemExit) as e:
        call_main(monkeypatch, tmp_path, ['--threads', value], 'run')
    assert e.value.code == 2
