"""Host-dependent resource ceilings (stdlib only).

The RSS ceiling is 6 GB on a non-Linux host and on a host of unknown size. A Linux host uses a RAM-dependent
ceiling: 0.8 x RAM below 24 GB, 0.75 x RAM from 24 GB, and at most 128 GB from 48 GB. SOSFUSE2_RSS_GB replaces the
host rule: it is both the ceiling and the per-process default.
default_workers (usable CPUs: affinity and cgroup-v2 cpu.max) moved here from column_parallel.py so the stdlib-only
pipeline controller can size the per-state process pools with the same rule the children use.
monitor (the RSS watchdog thread of the runners) moved here from checks/solver_diagnostics.py so the runners in jobs/ do
not import a diagnostics module.
"""

import math
import os
import sys
import time
from pathlib import Path

PROTECTED_RSS_GB = 6.0
RIG_RSS_GB = 128.0
RIG_MIN_RAM_GB = 48.0
MID_MIN_RAM_GB = 24.0
MID_RAM_SHARE = 0.75  # mid-size host: the OS and the controller keep a quarter
SMALL_RAM_SHARE = 0.8  # small host (under 24 GB): one state process at a time, the rest stays with the OS
DEFAULT_RAM_SHARE = 0.8  # all per-process defaults together stay within this share of the RAM
RSS_ENV = 'SOSFUSE2_RSS_GB'  # explicit per-process RSS limit in GB; replaces the host rule


def physical_ram_gb():
    try:
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def rss_override_gb():
    """The SOSFUSE2_RSS_GB value, or None when it is unset or empty."""
    text = os.environ.get(RSS_ENV, '').strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f'E_RSS_ENV: {RSS_ENV} must be a positive number of GB, got {text!r}')
    return value


def small_host():
    """A non-Linux host or one with less than 48 GB RAM. Worker counts use this, not the RSS ceiling, so
    SOSFUSE2_RSS_GB does not change them."""
    return sys.platform != 'linux' or physical_ram_gb() < RIG_MIN_RAM_GB


def rss_cap_gb():
    """SOSFUSE2_RSS_GB when set. Otherwise 6 on a non-Linux host (macOS) and on a host of unknown size; on Linux
    min(128, physical RAM) with at least 48 GB RAM, 0.75 x RAM with 24-48 GB, and 0.8 x RAM below that."""
    override = rss_override_gb()
    if override is not None:
        return override
    if sys.platform != 'linux':
        return PROTECTED_RSS_GB
    ram = physical_ram_gb()
    if ram >= RIG_MIN_RAM_GB:
        return min(RIG_RSS_GB, ram)
    if ram >= MID_MIN_RAM_GB:
        return MID_RAM_SHARE * ram
    return SMALL_RAM_SHARE * ram if ram > 0 else PROTECTED_RSS_GB


def default_rss_gb(processes):
    """Per-process --rss-gb when none is given: SOSFUSE2_RSS_GB when set (the caller then owns the sum over the
    processes); 6 on a non-Linux host; elsewhere [DEFAULT_RAM_SHARE] of the RAM split over the [processes] running at
    once (workers, plus the persistent service), at least 6 and at most the host cap. The limit is per process, so the
    cap itself would let them together exceed the RAM. The floor of 6 can still exceed the share when many processes
    run; the CPU budget, not this, bounds their number."""
    cap = rss_cap_gb()
    if rss_override_gb() is not None or sys.platform != 'linux':
        return cap
    return min(cap, max(PROTECTED_RSS_GB, DEFAULT_RAM_SHARE * physical_ram_gb() / max(1, processes)))


def over_budget_text(peak, budget):
    """'peak 7.06 GB > budget 6.88 GB (+2.6 %)' for a byte count that exceeds its byte budget."""
    text = f'peak {peak / 1e9:.2f} GB > budget {budget / 1e9:.2f} GB'
    return text + (f' (+{100.0 * (peak / budget - 1):.1f} %)' if budget > 0 else '')


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


def monitor(stop, limit):
    """RSS watchdog loop for a daemon thread: every 0.5 s until [stop] is set, exit the process with code 75 once the
    peak RSS reaches 95 % of [limit] bytes; one heartbeat line per minute."""
    last = time.monotonic()
    while not stop.wait(0.5):
        rss = peak_rss_bytes()
        if rss >= limit * 0.95:
            print(f'E_RSS_LIMIT peak={rss} cap={limit}', flush=True)
            os._exit(75)
        if time.monotonic() - last >= 60:
            print(f'SOSFUSE HEARTBEAT RSS_peak={rss / 1e9:.3f}GB', flush=True)
            last = time.monotonic()
