"""Public CI runs with a synthetic O4 fixture in a private temporary work root.

The two IPRT inputs are not redistributed. The session fetches them into the work root once, unless SOSFUSE2_OFFLINE is
set, and then runs offline. When they cannot be had, a test that stops with E_IPRT_OFFLINE is reported as skipped with
that reason, like a test whose compiler is missing; a test that fails for any other reason stays a failure, also when it
handled an E_IPRT_OFFLINE on the way (only the raised error and its explicit `raise ... from` causes are read).

The tests of the files in HEAVY_FILES carry the marker `heavy`: they solve reduced states, start child processes or
build caches and take most of the run time. `-m "not heavy"` runs the remaining files, which finish in a few minutes.
"""

import os
import re
import shutil

import pytest

IPRT_MISSING = []  # names of the IPRT inputs the session work root lacks; filled once by synthetic_minor_work
FETCH_FAILURE = []  # the E_IPRT_DOWNLOAD or E_IPRT_SHA256 message of the session's own fetch, if it failed

# Test files that took 20 s or longer in a serial run on one machine; tests/test_tiers.py checks that each name exists.
HEAVY_FILES = frozenset({
    'test_b2_s2.py',
    'test_bake_tolerance.py',
    'test_column_cache.py',
    'test_cpu_replay.py',
    'test_density_order.py',
    'test_extended_top.py',
    'test_gate.py',
    'test_host_cache.py',
    'test_ozone_temperature.py',
    'test_pipeline_persistent.py',
    'test_profile_step.py',
    'test_r17_port.py',
    'test_regressions_110.py',
    'test_sh_l.py',
    'test_strato_delta_adversarial.py',
    'test_sun_method.py',
    'test_tables_from_sources.py',
    'test_upper_air.py',
})


def pytest_configure(config):
    config.addinivalue_line('markers', 'heavy: test of a long-running file (see HEAVY_FILES in tests/conftest.py)')


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name in HEAVY_FILES:
            item.add_marker(pytest.mark.heavy)


def offline_stop(text):
    """True when `text` is the offline stop itself: the message starts with the code, or the message quotes a child's
    traceback whose error line carries it. A message that mentions the code elsewhere is another failure."""
    if text.startswith('E_IPRT_OFFLINE: '):
        return True
    _, found, after = text.rpartition('Traceback (most recent call last):')
    if not found:
        return False
    error = next((line for line in after.splitlines()[1:] if line and not line[0].isspace()), '')
    return re.match(r'[A-Za-z_][\w.]*: E_IPRT_OFFLINE: ', error) is not None


def _chain(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__


@pytest.fixture(scope='session', autouse=True)
def synthetic_minor_work(tmp_path_factory):
    from sosfuse2.asset_tools import iprt
    from sosfuse2.foundation import paths
    from sosfuse2.checks import testkit

    original = paths.work_root()
    if os.environ.get('SOSFUSE2_OFFLINE', '') in ('', '0'):
        try:
            iprt.inputs()
        except RuntimeError as e:  # the tests that need the inputs skip below and carry this reason
            if not str(e).startswith(('E_IPRT_DOWNLOAD:', 'E_IPRT_SHA256:')):
                raise
            FETCH_FAILURE[:] = [str(e)]
    root = tmp_path_factory.mktemp('synthetic-optics')
    if (original / 'iprt').is_dir():
        shutil.copytree(original / 'iprt', root / 'iprt')
    testkit.synthetic_o4(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('SOSFUSE2_WORK_ROOT', str(root))
        patch.setenv('SOSFUSE2_OFFLINE', '1')
        held = root / 'iprt'
        IPRT_MISSING[:] = [
            n for n, want in iprt.FILES.items() if not ((held / n).is_file() and iprt.sha256(held / n) == want)
        ]
        yield root


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not (report.failed and IPRT_MISSING and call.excinfo is not None):
        return
    if not any(offline_stop(str(e)) for e in _chain(call.excinfo.value)):
        return
    reason = (
        f'E_IPRT_OFFLINE: {", ".join(IPRT_MISSING)} not in the work root; fetch with '
        '`python -B -c "from sosfuse2.asset_tools import iprt; iprt.inputs()"`'
    )
    if FETCH_FAILURE:
        reason += '; the fetch of this session failed: ' + FETCH_FAILURE[0][:300]
    report.outcome = 'skipped'
    report.longrepr = (str(item.path), (item.location[1] or 0) + 1, 'Skipped: ' + reason)
