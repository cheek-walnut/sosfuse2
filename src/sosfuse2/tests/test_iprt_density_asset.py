"""IPRT download failures and the packaged-asset digest check (no GPU, no network, no sky-sos root).

iprt.fetch retries a failed download 3 times and then raises E_IPRT_DOWNLOAD with the file, URL and cause;
density.load_asset lets that reason through instead of replacing it with a bare E_DENSITY_ASSET;
density.asset_path refuses a packaged usstd76_profile.json whose digest is not the pinned one (E_DENSITY_ASSET_SHA256);
a test session without the inputs reports a test that stops with E_IPRT_OFFLINE as skipped and every other failure as failed.
Run: python -m pytest -q src/sosfuse2/tests/test_iprt_density_asset.py
"""

import hashlib
import os
import subprocess
import sys
import urllib.error

import pytest

from sosfuse2 import density, iprt

NAME = 'tau_absorption_320nm_usstd.dat'


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path / 'work'))
    monkeypatch.delenv('SOSFUSE2_OFFLINE', raising=False)
    monkeypatch.setattr(iprt.time, 'sleep', lambda s: None)
    return tmp_path


def test_download_failure_is_retried_then_reported_with_its_cause(work, monkeypatch):
    calls = []

    def refuse(url, timeout=None):
        calls.append(url)
        raise urllib.error.URLError('connection refused')

    monkeypatch.setattr(iprt.urllib.request, 'urlopen', refuse)
    with pytest.raises(RuntimeError) as exc:
        iprt.fetch(NAME)
    assert len(calls) == iprt.DOWNLOAD_ATTEMPTS == 3
    message = str(exc.value)
    assert message.startswith('E_IPRT_DOWNLOAD:') and NAME in message and iprt.BASE in message and 'URLError' in message
    assert not list(iprt.cache_dir().glob('*.tmp'))


def test_download_succeeds_on_a_later_attempt(work, monkeypatch):
    payload = b'sha256 is checked, so the wrong bytes must not be accepted'
    state = dict(n=0)

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return payload

    def flaky(url, timeout=None):
        state['n'] += 1
        if state['n'] < 3:
            raise TimeoutError('timed out')
        return Reply()

    monkeypatch.setattr(iprt.urllib.request, 'urlopen', flaky)
    with pytest.raises(RuntimeError, match='E_IPRT_SHA256'):  # third attempt answered, digest differs
        iprt.fetch(NAME)
    assert state['n'] == 3 and not (iprt.cache_dir() / NAME).exists()


def test_load_asset_keeps_the_iprt_reason(work, monkeypatch):
    monkeypatch.setattr(density, '_PACKAGED', work / 'absent.json')
    monkeypatch.setenv('SOSFUSE2_OFFLINE', '1')
    with pytest.raises(RuntimeError, match='E_IPRT_OFFLINE'):
        density.load_asset()


def test_packaged_asset_with_another_digest_is_refused(work, monkeypatch):
    fake = work / 'usstd76_profile.json'
    fake.write_text('{"not": "the pinned asset"}\n')
    assert hashlib.sha256(fake.read_bytes()).hexdigest() != density.ASSET_SHA256
    monkeypatch.setattr(density, '_PACKAGED', fake)
    with pytest.raises(RuntimeError, match='E_DENSITY_ASSET_SHA256'):
        density.asset_path()


def test_session_without_inputs_skips_only_the_offline_stop(tmp_path):
    case = tmp_path / 'test_case.py'
    case.write_text(
        'import pytest\n'
        'def test_needs_inputs():\n'
        '    from sosfuse2 import iprt\n'
        '    iprt.inputs()\n'
        'def test_wrapped():\n'
        '    from sosfuse2 import iprt\n'
        '    try:\n'
        '        iprt.inputs()\n'
        '    except RuntimeError as e:\n'
        "        raise ValueError('wrapped') from e\n"
        'def test_handled_then_fails():\n'
        '    from sosfuse2 import iprt\n'
        '    try:\n'
        '        iprt.inputs()\n'
        '    except RuntimeError:\n'
        "        raise AssertionError('E_OTHER: after a handled stop')\n"
        'def test_other_failure():\n'
        "    raise RuntimeError('E_OTHER: not the inputs')\n"
        'def test_expects_the_stop():\n'
        '    from sosfuse2 import iprt\n'
        "    with pytest.raises(RuntimeError, match='E_IPRT_OFFLINE'):\n"
        '        iprt.inputs()\n'
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith('SOSFUSE2_')}
    env.update(SOSFUSE2_WORK_ROOT=str(tmp_path / 'work'), SOSFUSE2_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    run = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider', '-p', 'sosfuse2.tests.conftest']
        + ['--rootdir', str(tmp_path), '--basetemp', str(tmp_path / 'bt'), str(case)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=300,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    assert '2 failed, 1 passed, 2 skipped' in run.stdout, run.stdout
    skipped = [line for line in run.stdout.splitlines() if line.startswith('SKIPPED')]
    assert len(skipped) == 2, run.stdout
    assert all('E_IPRT_OFFLINE: ' in line and 'not in the work root' in line for line in skipped), run.stdout
    assert 'test_other_failure' in run.stdout and 'test_handled_then_fails' in run.stdout and 'E_OTHER' in run.stdout
