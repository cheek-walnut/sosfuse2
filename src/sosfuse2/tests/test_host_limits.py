"""rss_cap_gb host rules (host_limits.py). CPU only, stdlib + pytest.

HOME points at a temporary folder; sys.platform and physical_ram_gb are faked so the Linux host rule is reached.

Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider src/sosfuse2/tests/test_host_limits.py
"""

from pathlib import Path

import pytest

from sosfuse2.foundation import host_limits as hl  # noqa: E402
from sosfuse2.jobs import pipeline

RIG_RAM_GB = 64.0


@pytest.fixture
def linux_rig(tmp_path, monkeypatch):
    """A Linux host with 64 GB RAM and an empty HOME; returns that HOME."""
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.delenv(hl.RSS_ENV, raising=False)
    monkeypatch.setattr(hl.sys, 'platform', 'linux')
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: RIG_RAM_GB)
    monkeypatch.setattr(hl, '_PROC_CGROUP', tmp_path / 'cgroup')
    monkeypatch.setattr(hl, '_PROC_MOUNTINFO', tmp_path / 'mountinfo')
    assert Path.home() == tmp_path
    return tmp_path


@pytest.fixture
def cgroup(linux_rig):
    mount = linux_rig / 'cgroup mount'
    mount.mkdir()
    escaped_mount = str(mount).replace(' ', r'\040')

    def setup(group='0::/parent/leaf', root='/', filesystem='cgroup2', options='rw'):
        hl._PROC_CGROUP.write_text(group + '\n')
        hl._PROC_MOUNTINFO.write_text(f'1 0 0:1 {root} {escaped_mount} rw - {filesystem} cgroup {options}\n')
        return mount

    return setup


def test_home_folders_do_not_change_the_cap(linux_rig):
    """Folders in HOME (the former protection markers) no longer lower the cap."""
    (linux_rig / 'tools' / 'gpu_guard').mkdir(parents=True)
    (linux_rig / 'gpu_guard').mkdir()
    assert hl.rss_cap_gb() == 64.0


def test_non_linux_keeps_six(linux_rig, monkeypatch):
    monkeypatch.setattr(hl.sys, 'platform', 'darwin')
    assert hl.rss_cap_gb() == hl.default_rss_gb(1) == hl.PROTECTED_RSS_GB == 6.0


def test_rig_uses_ram_rule(linux_rig):
    assert hl.rss_cap_gb() == min(hl.RIG_RSS_GB, RIG_RAM_GB) == 64.0


def test_mid_size_host_gets_ram_share(linux_rig, monkeypatch):
    """A mid-size host uses a share of physical memory."""
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: 32.0)
    assert hl.rss_cap_gb() == 24.0
    assert hl.default_rss_gb(1) == 24.0
    assert hl.default_rss_gb(2) == pytest.approx(12.8)


def test_small_host_gets_ram_share(linux_rig, monkeypatch):
    """Below 24 GB the cap is 0.8 x RAM; the per-process default splits the same share and stays at 6 or more."""
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: 16.0)
    assert hl.rss_cap_gb() == hl.default_rss_gb(1) == pytest.approx(12.8)
    assert hl.default_rss_gb(2) == pytest.approx(6.4)
    assert hl.default_rss_gb(4) == 6.0


def test_tiny_host_gets_ram_share_and_unknown_keeps_six(linux_rig, monkeypatch):
    """No floor under the RAM share: a limit above the RAM would never act. Unknown size keeps 6."""
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: 4.0)
    assert hl.rss_cap_gb() == hl.default_rss_gb(1) == pytest.approx(3.2)
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: 0.0)
    assert hl.rss_cap_gb() == hl.default_rss_gb(1) == 6.0


def test_environment_replaces_the_host_rule(linux_rig, monkeypatch):
    """SOSFUSE2_RSS_GB is the ceiling and the per-process default on every host; worker rules do not follow it."""
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: 16.0)
    monkeypatch.setenv(hl.RSS_ENV, '12')
    assert hl.rss_cap_gb() == hl.default_rss_gb(1) == hl.default_rss_gb(8) == 12.0
    assert hl.small_host()
    monkeypatch.setattr(hl.sys, 'platform', 'darwin')
    monkeypatch.setenv(hl.RSS_ENV, '64')
    assert hl.rss_cap_gb() == hl.default_rss_gb(2) == 64.0
    assert hl.small_host()
    monkeypatch.setenv(hl.RSS_ENV, ' ')
    assert hl.rss_cap_gb() == 6.0
    for bad in ('0', '-1', 'nan', 'inf', 'six'):
        monkeypatch.setenv(hl.RSS_ENV, bad)
        with pytest.raises(ValueError, match='E_RSS_ENV'):
            hl.rss_cap_gb()


def test_over_budget_text_never_raises():
    assert hl.over_budget_text(3_000_000_000, 2_000_000_000) == 'peak 3.00 GB > budget 2.00 GB (+50.0 %)'
    assert hl.over_budget_text(3_000_000_000, 0) == 'peak 3.00 GB > budget 0.00 GB'
    assert hl.over_budget_text(None, 'x') == "peak None > budget 'x'"
    record = dict(requested=3, effective=0, peak=3_000_000_000, budget=2_000_000_000, reason='r')
    assert hl.ksplit_fallback_warning(record, 'a') == (
        'WARNING KSPLIT fallback in state a: requested 3 -> 0, peak 3.00 GB > budget 2.00 GB (+50.0 %), r'
    )
    assert hl.ksplit_fallback_warning({}).startswith('WARNING KSPLIT fallback: requested None -> None, peak None')


def test_small_host_follows_platform_and_ram(linux_rig, monkeypatch):
    assert not hl.small_host()
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: 47.0)
    assert hl.small_host()
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: RIG_RAM_GB)
    monkeypatch.setattr(hl.sys, 'platform', 'darwin')
    assert hl.small_host()


def test_default_splits_ram_over_workers(linux_rig):
    """The limit is per process: the defaults of all workers together stay within 0.8 x RAM, never below 6."""
    assert hl.default_rss_gb(1) == pytest.approx(51.2)
    assert hl.default_rss_gb(4) == pytest.approx(12.8)
    assert hl.default_rss_gb(16) == 6.0


def test_cgroup_v2_uses_lowest_visible_ancestor_limit(cgroup, monkeypatch):
    mount = cgroup()
    leaf = mount / 'parent' / 'leaf'
    leaf.mkdir(parents=True)
    (mount / 'memory.max').write_text('32000000000\n')
    (leaf.parent / 'memory.max').write_text('12000000000\n')
    (leaf / 'memory.max').write_text('16000000000\n')
    assert hl.usable_ram_gb() == 12.0
    assert hl.rss_cap_gb() == hl.default_rss_gb(1) == pytest.approx(9.6)
    assert hl.default_rss_gb(2) == 6.0
    assert hl.small_host()
    monkeypatch.setenv(hl.RSS_ENV, '20')
    assert hl.rss_cap_gb() == hl.default_rss_gb(2) == 20.0
    assert hl.small_host()


@pytest.mark.parametrize('value', [None, 'max\n', 'invalid', '96000000000'])
def test_cgroup_v2_falls_back_to_physical_ram(cgroup, value):
    mount = cgroup()
    (mount / 'parent' / 'leaf').mkdir(parents=True)
    if value is not None:
        (mount / 'memory.max').write_text(value)
    assert hl.usable_ram_gb() == RIG_RAM_GB
    assert hl.rss_cap_gb() == RIG_RAM_GB


@pytest.mark.parametrize(('group', 'root'), [('0::/', '/host/slice'), ('0::/../outside', '/')])
def test_cgroup_v2_namespace_root(cgroup, linux_rig, group, root):
    mount = cgroup(group=group, root=root)
    (mount / 'memory.max').write_text('4000000000\n')
    (linux_rig / 'memory.max').write_text('1000000000\n')
    assert hl.usable_ram_gb() == 4.0
    assert hl.rss_cap_gb() == hl.default_rss_gb(2) == pytest.approx(3.2)


def test_cgroup_v1_uses_memory_controller_and_ignores_unlimited(cgroup, monkeypatch):
    mount = cgroup(group='5:memory:/parent/leaf\n0::/parent/leaf', filesystem='cgroup', options='rw,memory')
    leaf = mount / 'parent' / 'leaf'
    leaf.mkdir(parents=True)
    (mount / 'memory.limit_in_bytes').write_text('9223372036854771712\n')
    limit = leaf.parent / 'memory.limit_in_bytes'
    limit.write_text('8000000000\n')
    assert hl.usable_ram_gb() == 8.0
    limit.unlink()
    monkeypatch.setattr(hl, 'physical_ram_gb', lambda: 0.0)
    assert hl.usable_ram_gb() is None
    assert hl.rss_cap_gb() == 6.0


@pytest.mark.parametrize(('ram', 'expected'), [(6, 1), (16, 2), (60, 16)])
def test_auto_workers_respects_cgroup_ram(cgroup, monkeypatch, ram, expected):
    mount = cgroup()
    (mount / 'memory.max').write_text(str(ram * 10**9))
    monkeypatch.setattr(pipeline, 'cpu_budget', lambda: 64)
    assert pipeline.auto_workers(64, 1) == expected
    monkeypatch.setenv(hl.RSS_ENV, '24')
    assert pipeline.auto_workers(64, 1) == expected
    if ram == 6:
        monkeypatch.setattr(pipeline, 'slot_workers', lambda states, persistent: 2)
        assert pipeline.auto_workers(64, 1, backend='cupy') == 1
