"""rss_cap_gb eda detection (host_limits.py). CPU only, stdlib + pytest.

HOME points at a temporary folder; sys.platform and physical_ram_gb are faked so the Linux host rule is reached.
The protected profile is recognised by either supported marker.

Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider src/sosfuse2/tests/test_host_limits.py
"""

import sys
from pathlib import Path

import pytest

from sosfuse2 import host_limits as hl  # noqa: E402

RIG_RAM_GB = 64.0


@pytest.fixture
def linux_rig(tmp_path, monkeypatch):
    """A Linux host with 64 GB RAM and an empty HOME; returns that HOME."""
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(hl.sys, 'platform', 'linux')
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: RIG_RAM_GB)
    assert Path.home() == tmp_path
    return tmp_path


def test_tools_gpu_guard_marks_eda(linux_rig):
    (linux_rig / 'tools' / 'gpu_guard').mkdir(parents=True)
    assert hl.rss_cap_gb() == hl.PROTECTED_RSS_GB == 6.0


def test_old_gpu_guard_marks_eda(linux_rig):
    (linux_rig / 'gpu_guard').mkdir()
    assert hl.rss_cap_gb() == hl.PROTECTED_RSS_GB == 6.0


def test_no_marker_uses_ram_rule(linux_rig):
    assert hl.rss_cap_gb() == min(hl.RIG_RSS_GB, RIG_RAM_GB) == 64.0


def test_mid_size_host_gets_ram_share(linux_rig, monkeypatch):
    """A mid-size host uses a share of physical memory."""
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: 32.0)
    assert hl.rss_cap_gb() == 24.0
    assert hl.default_rss_gb(1) == 24.0
    assert hl.default_rss_gb(2) == pytest.approx(12.8)


def test_small_host_stays_protected(linux_rig, monkeypatch):
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: 16.0)
    assert hl.rss_cap_gb() == hl.default_rss_gb(1) == 6.0


def test_default_on_eda_is_protected(linux_rig):
    (linux_rig / 'tools' / 'gpu_guard').mkdir(parents=True)
    assert hl.default_rss_gb(1) == hl.default_rss_gb(4) == 6.0


def test_default_splits_ram_over_workers(linux_rig):
    """The limit is per process: the defaults of all workers together stay within 0.8 x RAM, never below 6."""
    assert hl.default_rss_gb(1) == pytest.approx(51.2)
    assert hl.default_rss_gb(4) == pytest.approx(12.8)
    assert hl.default_rss_gb(16) == 6.0
