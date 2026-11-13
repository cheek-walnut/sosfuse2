"""Host-dependent resource ceilings (stdlib only).

The default RSS ceiling is 6 GB. Linux hosts without a protection marker use
RAM-dependent ceilings above 24 GB; hosts with at least 48 GB use at most 128 GB.
default_workers (usable CPUs: affinity and cgroup-v2 cpu.max) moved here from column_parallel.py so the stdlib-only
pipeline controller can size the per-state process pools with the same rule the children use.
"""

import os
import sys
from pathlib import Path

PROTECTED_RSS_GB = 6.0
RIG_RSS_GB = 128.0
RIG_MIN_RAM_GB = 48.0
MID_MIN_RAM_GB = 24.0
MID_RAM_SHARE = 0.75  # mid-size host: the OS and the controller keep a quarter
DEFAULT_RAM_SHARE = 0.8  # all per-process defaults together stay within this share of the RAM


def physical_ram_gb():
    try:
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


EDA_MARKERS = ('tools/gpu_guard', 'gpu_guard')  # current marker first, then the older home-relative marker


def rss_cap_gb():
    """6 on the Mac, on a host with a protection marker, and on any small/unknown host;
    min(128, physical RAM) on a Linux host with at least 48 GB RAM; 0.75 x RAM on one with 24-48 GB."""
    try:
        markers = [Path.home() / marker for marker in EDA_MARKERS]
    except RuntimeError:
        markers = []  # no home directory (container user without a passwd entry): no marker, never an error
    if sys.platform != 'linux' or any(marker.exists() for marker in markers):
        return PROTECTED_RSS_GB
    ram = physical_ram_gb()
    if ram >= RIG_MIN_RAM_GB:
        return min(RIG_RSS_GB, ram)
    return MID_RAM_SHARE * ram if ram >= MID_MIN_RAM_GB else PROTECTED_RSS_GB


def default_rss_gb(processes):
    """Per-process --rss-gb when none is given: 6 on a protected host; elsewhere [DEFAULT_RAM_SHARE] of the RAM split
    over the [processes] running at once (workers, plus the persistent service), at most the host cap. The limit is per
    process, so the cap itself would let them together exceed the RAM. The floor of 6 (the old default) can still
    exceed the share when many processes run; the CPU budget, not this, bounds their number."""
    cap = rss_cap_gb()
    if cap <= PROTECTED_RSS_GB:
        return cap
    return min(cap, max(PROTECTED_RSS_GB, DEFAULT_RAM_SHARE * physical_ram_gb() / max(1, processes)))


def default_workers():
    """Bound process parallelism by affinity and every visible cgroup-v2 ancestor quota."""
    try:
        count = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        count = max(1, os.cpu_count() or 1)
    try:
        groups = Path('/proc/self/cgroup').read_text()
        mounts = Path('/proc/self/mountinfo').read_text()
    except FileNotFoundError:  # macOS, or procfs unavailable
        return count
    return min([count, *_quota_limits(groups, mounts)])


def _quota_limits(groups, mounts):
    group = next((line[3:] for line in groups.splitlines() if line.startswith('0::')), None)
    if group is None:
        return []
    limits = []
    for line in mounts.splitlines():
        left, sep, right = line.partition(' - ')
        if not sep or right.split()[0] != 'cgroup2':
            continue
        fields = left.split()
        decode = lambda s: s.replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\')
        root, mount = Path(decode(fields[3])), Path(decode(fields[4]))
        try:
            relative = Path(group).relative_to(root)
        except ValueError:
            if group != '/':
                continue
            relative = Path('.')  # cgroup namespace exposes its root as "/"
        # Do not escape the mounted hierarchy for namespace-relative /../ paths.
        if '..' in relative.parts:
            relative = Path('.')
        current = mount / relative
        while True:
            try:
                quota, period = (current / 'cpu.max').read_text().split()
                if quota != 'max':
                    limits.append(max(1, int(quota) // int(period)))
            except FileNotFoundError:
                pass
            if current == mount:
                break
            current = current.parent
    return limits


def peak_rss_bytes():
    """Peak resident set size of this process, in bytes."""
    import resource
    import sys

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)
