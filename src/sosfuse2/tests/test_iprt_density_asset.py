"""IPRT download failures and the packaged-asset digest check (no GPU, no network, no sky-sos root).

iprt.fetch retries a failed download 3 times and then raises E_IPRT_DOWNLOAD with the file, URL and cause;
density.load_asset lets that reason through instead of replacing it with a bare E_DENSITY_ASSET;
density.asset_path refuses a packaged usstd76_profile.json whose digest is not the pinned one (E_DENSITY_ASSET_SHA256).
Run: python -m pytest -q src/sosfuse2/tests/test_iprt_density_asset.py
"""

import hashlib
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
