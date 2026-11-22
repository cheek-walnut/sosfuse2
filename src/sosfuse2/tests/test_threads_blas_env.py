"""run_state run directly: the --threads value reaches the BLAS thread variables before NumPy is imported."""

import os
import subprocess
import sys
import textwrap

import pytest

from sosfuse2.jobs import run_state

THREAD_ENV = run_state._THREAD_ENV


@pytest.mark.parametrize(
    'argv, expected',
    [
        ([], 2),
        (['--id', 'x', '--threads', '3'], 3),
        (['--threads=4', '--id', 'x'], 4),
        (['--thr', '5'], 5),
        (['--thread=6'], 6),
        (['--threads', '3', '--threads', '7'], 7),
        (['--', '--threads', '9'], 2),
    ],
)
def test_value_written_to_every_thread_variable(argv, expected):
    environ = {'OMP_NUM_THREADS': '12'}
    assert run_state._thread_env_before_numpy(argv, environ) == expected
    assert environ == {k: str(expected) for k in THREAD_ENV}


@pytest.mark.parametrize('argv', [['--threads', '0'], ['--threads', 'x'], ['--threads=-1'], ['--threads', '2.5']])
def test_invalid_value_writes_nothing(argv):
    environ = {'OMP_NUM_THREADS': '12'}
    assert run_state._thread_env_before_numpy(argv, environ) is None
    assert environ == {'OMP_NUM_THREADS': '12'}


def test_other_options_are_not_taken_for_threads():
    environ = {}
    assert run_state._thread_env_before_numpy(['--t', '3', '--id', '--threads'], environ) == 2


# Runs the module as a script and records OPENBLAS_NUM_THREADS at the first import of numpy, which is when OpenBLAS
# reads it. --help stops main() after the module-level imports.
PROBE = textwrap.dedent(
    """
    import builtins, os, runpy, sys
    seen = []
    real = builtins.__import__
    def hook(name, *args, **kwargs):
        if name.split('.')[0] == 'numpy' and 'numpy' not in sys.modules and not seen:
            seen.append(os.environ.get('OPENBLAS_NUM_THREADS'))
        return real(name, *args, **kwargs)
    builtins.__import__ = hook
    sys.argv = ['run_state'] + sys.argv[1:] + ['--help']
    try:
        runpy.run_module('sosfuse2.jobs.run_state', run_name='__main__')
    except SystemExit:
        pass
    print('AT_NUMPY_IMPORT', seen[0] if seen else 'not-imported')
    """
)


@pytest.mark.parametrize('args, expected', [([], '2'), (['--threads', '3'], '3')])
def test_direct_run_sets_blas_before_numpy(args, expected):
    env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV}
    out = subprocess.run(
        [sys.executable, '-c', PROBE, *args], env=env, capture_output=True, text=True, check=True
    ).stdout
    assert f'AT_NUMPY_IMPORT {expected}' in out.splitlines()
