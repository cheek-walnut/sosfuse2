"""run_state run directly: the --threads value reaches the BLAS thread variables before NumPy is imported."""

import os
import subprocess
import sys
import textwrap

import pytest

from sosfuse2.jobs import pipeline, run_state

# Written out here so that a name dropped from the implementation fails the tests.
THREAD_ENV = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS',
)


def test_thread_variables_match_pipeline():
    assert run_state._THREAD_ENV == THREAD_ENV == pipeline.THREAD_ENV


@pytest.mark.parametrize(
    'argv, expected',
    [
        ([], 2),
        (['--id', 'x', '--threads', '3'], 3),
        (['--threads=4', '--id', 'x'], 4),
        (['--t=1'], 1),
        (['--t', '5'], 5),
        (['--thread=6'], 6),
        (['--threads', '3', '--threads', '7'], 7),
        (['--', '--threads', '9'], 2),
        (['--states=--threads=9'], 2),
        (['-t', '3'], 2),
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


def test_no_other_option_is_a_prefix_of_threads():
    # The early reader takes every prefix of --threads from --t as --threads. That matches argparse only while no
    # other option is named exactly such a prefix (an exact name wins over a prefix in argparse).
    names = {s for action in run_state.build_parser()._actions for s in action.option_strings}
    assert '--threads' in names
    assert not {n for n in names if n != '--threads' and len(n) >= 3 and '--threads'.startswith(n)}


BASE = ['--id', 'x', '--work-root', 'w']


@pytest.mark.parametrize(
    'extra',
    [
        [],
        ['--threads', '3'],
        ['--t=1'],
        ['--th', '4'],
        ['--thread=5'],
        ['--threads', '3', '--threads', '7'],
        ['--states=--threads=9'],
        ['--out-name', 'n', '--thr=6'],
    ],
)
def test_early_value_matches_the_parser(extra):
    argv = BASE + extra
    parsed = run_state.build_parser().parse_args(argv).threads
    assert run_state._thread_env_before_numpy(argv, {}) == (2 if parsed is None else parsed)


# Runs the module as a script and records the thread variables at the first import of numpy, which is when BLAS
# libraries read them. --help stops main() after the module-level imports.
PROBE = textwrap.dedent(
    """
    import builtins, json, os, runpy, sys
    names = json.loads(sys.argv[1])
    seen = []
    real = builtins.__import__
    def hook(name, *args, **kwargs):
        if name.split('.')[0] == 'numpy' and 'numpy' not in sys.modules and not seen:
            seen.append({k: os.environ.get(k) for k in names})
        return real(name, *args, **kwargs)
    builtins.__import__ = hook
    sys.argv = ['run_state'] + sys.argv[2:] + ['--help']
    try:
        runpy.run_module('sosfuse2.jobs.run_state', run_name='__main__')
    except SystemExit:
        pass
    print('AT_NUMPY_IMPORT', json.dumps(seen[0] if seen else None, sort_keys=True))
    """
)


@pytest.mark.parametrize('args, expected', [([], '2'), (['--threads', '3'], '3'), (['--t=1'], '1')])
def test_direct_run_sets_blas_before_numpy(args, expected):
    import json

    env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV}
    out = subprocess.run(
        [sys.executable, '-c', PROBE, json.dumps(THREAD_ENV), *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    line = next(l for l in out.splitlines() if l.startswith('AT_NUMPY_IMPORT '))
    assert json.loads(line.split(' ', 1)[1]) == {k: expected for k in THREAD_ENV}
