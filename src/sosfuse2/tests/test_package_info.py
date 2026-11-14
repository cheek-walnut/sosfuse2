"""run_gpu_state_v3.package_info(): where the recorded package version and commit come from (no GPU, no network, no sky-sos root).

Priority: SOSFUSE2_PACKAGE_COMMIT, then the installer's direct_url.json (only when the distribution's sosfuse2/__init__.py is
THIS package), then the git checkout whose src/sosfuse2 is this package; otherwise None, never an error.
Run: python -m pytest -q src/sosfuse2/tests/test_package_info.py
"""

import importlib.metadata
import json
import re

from sosfuse2.foundation import paths
from sosfuse2.jobs import run_state as v3

COMMIT = 'a' * 40


class FakeDist:
    def __init__(self, located, direct_url):
        self.version, self.located, self.direct_url = '9.8.7', located, direct_url

    def locate_file(self, name):
        return self.located / name

    def read_text(self, name):
        return self.direct_url if name == 'direct_url.json' else None


def install(monkeypatch, dist):
    monkeypatch.setattr(importlib.metadata, 'distribution', lambda name: dist)


def vcs(commit=COMMIT):
    return json.dumps(dict(url='https://example.invalid/sosfuse2', vcs_info=dict(vcs='git', commit_id=commit)))


def test_env_wins_over_direct_url(monkeypatch):
    install(monkeypatch, FakeDist(paths.PKG.parent, vcs()))
    monkeypatch.setenv('SOSFUSE2_PACKAGE_COMMIT', 'b' * 40)
    assert v3.package_info() == dict(package_version='9.8.7', package_commit='b' * 40)


def test_direct_url_commit_is_used_only_for_this_copy_of_the_package(monkeypatch, tmp_path):
    monkeypatch.delenv('SOSFUSE2_PACKAGE_COMMIT', raising=False)
    install(monkeypatch, FakeDist(paths.PKG.parent, vcs()))  # metadata describes this very package
    assert v3.package_info()['package_commit'] == COMMIT
    install(monkeypatch, FakeDist(tmp_path, vcs()))  # an installed copy elsewhere: not this package
    commit = v3.package_info()['package_commit']
    assert commit != COMMIT and (commit is None or re.fullmatch('[0-9a-f]{40}', commit))


def test_without_env_or_distribution_the_commit_is_a_git_hash_or_none(monkeypatch):
    monkeypatch.delenv('SOSFUSE2_PACKAGE_COMMIT', raising=False)

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'distribution', missing)
    info = v3.package_info()
    assert info['package_version'] is None
    assert info['package_commit'] is None or re.fullmatch('[0-9a-f]{40}', info['package_commit'])
