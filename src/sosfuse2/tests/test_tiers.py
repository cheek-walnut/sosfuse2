"""Check the heavy file list and the failure policy for required offline CPU tests."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from sosfuse2.foundation import paths
from sosfuse2.tests import conftest


def test_heavy_files_exist():
    here = Path(__file__).resolve().parent
    missing = sorted(name for name in conftest.HEAVY_FILES if not (here / name).is_file())
    assert not missing, f'remove from HEAVY_FILES: {missing}'


def test_this_file_is_not_heavy(request):
    assert Path(__file__).name not in conftest.HEAVY_FILES
    assert request.node.get_closest_marker('heavy') is None


def run_required_cpu(tmp_path, *options):
    env = paths.child_env(dict(os.environ, SOSFUSE2_OFFLINE='1', SOSFUSE2_WORK_ROOT=str(tmp_path / 'work')))
    return subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *options,
         '--pyargs', 'sosfuse2.tests.test_offline_cpu'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )


def test_required_cpu_skip_fails(tmp_path):
    # Exercise a real member of the required suite, including pytest's setup-time skip reporting.
    (tmp_path / 'skip_one.py').write_text(
        'import pytest\n'
        'def pytest_collection_modifyitems(items):\n'
        '    for item in items:\n'
        '        if item.name == "test_reduced_numpy_table":\n'
        '            item.add_marker(pytest.mark.skip(reason="injected offline CPU skip"))\n'
    )
    run = run_required_cpu(tmp_path, '-p', 'skip_one')
    assert run.returncode == pytest.ExitCode.TESTS_FAILED, run.stdout + run.stderr
    assert 'required offline CPU check skipped' in run.stdout
    assert 'injected offline CPU skip' in run.stdout


@pytest.mark.parametrize(('hook', 'statement', 'reason'), [
    ('pytest_runtest_call(item)', 'pytest.xfail("injected xfail")', 'injected xfail'),
    ('pytest_collection_modifyitems(items)', 'items[0].add_marker(pytest.mark.xfail(run=False, reason="no run"))',
     'no run'),
    ('pytest_collection_modifyitems(items)', 'items[0].add_marker(pytest.mark.xfail(strict=False, reason="xpass"))',
     'XPASS during call: xpass'),
    ('pytest_runtest_setup(item)', 'raise RuntimeError("injected error")', 'RuntimeError: injected error'),
], ids=['xfail', 'xfail-no-run', 'xpass', 'error'])
def test_required_cpu_outcome_fails(tmp_path, hook, statement, reason):
    # Select one real member before marking it so an XPASS exercises its normal test body.
    (tmp_path / 'inject_outcome.py').write_text(
        f'import pytest\n@pytest.hookimpl(trylast=True)\ndef {hook}:\n    {statement}\n'
    )
    run = run_required_cpu(tmp_path, '-p', 'inject_outcome', '-k', 'test_default_density_needs_iprt')
    assert run.returncode == pytest.ExitCode.TESTS_FAILED, run.stdout + run.stderr
    assert reason in run.stdout


def test_required_cpu_empty_selection_fails(tmp_path):
    run = run_required_cpu(tmp_path, '-k', 'no_such_required_check')
    assert run.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, run.stdout + run.stderr
