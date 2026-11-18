"""Host-dependent resource ceilings (stdlib only).

The RSS ceiling is 6 GB on a non-Linux host and on a host of unknown size. A Linux host uses a RAM-dependent
ceiling: 0.8 x usable RAM below 24 GB, 0.75 x usable RAM from 24 GB, and at most 128 GB from 48 GB. Usable RAM
is bounded by physical RAM and visible cgroup memory limits. SOSFUSE2_RSS_GB replaces the
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
_PROC_CGROUP = Path('/proc/self/cgroup')
_PROC_MOUNTINFO = Path('/proc/self/mountinfo')
_V1_UNLIMITED_MIN = 0x7fffffffffff0000  # LONG_MAX rounded down to a page, allowing pages up to 64 KiB


def physical_ram_gb():
    try:
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def usable_ram_gb():
    """RAM in GB bounded by physical size and visible cgroup limits, or None if unknown."""
    physical = physical_ram_gb()
    ram = [physical] if physical > 0 else []
    if sys.platform == 'linux':
        try:
            groups = _PROC_CGROUP.read_text()
            mounts = _PROC_MOUNTINFO.read_text()
        except (OSError, UnicodeError):
            pass
        else:
            ram.extend(limit / 1e9 for limit in _memory_limits(groups, mounts))
    return min(ram, default=None)


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
    """A non-Linux host or one with unknown or less than 48 GB of usable RAM."""
    return sys.platform != 'linux' or (usable_ram_gb() or 0.0) < RIG_MIN_RAM_GB


def rss_cap_gb():
    """SOSFUSE2_RSS_GB when set. Otherwise 6 on a non-Linux host (macOS) and on a host of unknown size; on Linux
    min(128, usable RAM) with at least 48 GB, 0.75 x usable RAM with 24-48 GB, and 0.8 x usable RAM below that."""
    override = rss_override_gb()
    if override is not None:
        return override
    if sys.platform != 'linux':
        return PROTECTED_RSS_GB
    return _rss_cap_for_ram(usable_ram_gb())


def _rss_cap_for_ram(ram):
    if ram is None:
        return PROTECTED_RSS_GB
    if ram >= RIG_MIN_RAM_GB:
        return min(RIG_RSS_GB, ram)
    if ram >= MID_MIN_RAM_GB:
        return MID_RAM_SHARE * ram
    return SMALL_RAM_SHARE * ram


def default_rss_gb(processes):
    """Per-process --rss-gb when none is given: SOSFUSE2_RSS_GB when set (the caller then owns the sum over the
    processes); 6 when RAM is unknown or on non-Linux; elsewhere [DEFAULT_RAM_SHARE] of usable RAM split over the
    concurrent [processes] (workers plus the persistent service). Apply the 6 GB floor first, then the host cap.
    The floor can make the sum of the per-process limits exceed the shared RAM budget."""
    override = rss_override_gb()
    if override is not None:
        return override
    if sys.platform != 'linux':
        return PROTECTED_RSS_GB
    ram = usable_ram_gb()
    if ram is None:
        return PROTECTED_RSS_GB
    share = DEFAULT_RAM_SHARE * ram / max(1, processes)
    return min(_rss_cap_for_ram(ram), max(PROTECTED_RSS_GB, share))


def over_budget_text(peak, budget):
    """'peak P GB > budget B GB (+X %)' for a byte count over its byte budget; the percentage is left out when the
    budget is not positive. Values that are not numbers are shown as they are, so a damaged record never raises."""
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (peak, budget)):
        return f'peak {peak!r} > budget {budget!r}'
    text = f'peak {peak / 1e9:.2f} GB > budget {budget / 1e9:.2f} GB'
    return text + (f' (+{100.0 * (peak / budget - 1):.1f} %)' if budget > 0 else '')


def ksplit_fallback_warning(decision, state=None):
    """One warning line for a KSPLIT fallback record (requested, effective, peak, budget, reason)."""
    warning = 'WARNING KSPLIT fallback%s: requested %s -> %s, %s, %s' % (
        '' if state is None else f' in state {state}',
        decision.get('requested'),
        decision.get('effective'),
        over_budget_text(decision.get('peak'), decision.get('budget')),
        decision.get('reason'),
    )
    items = decision.get('items')
    if not isinstance(items, dict):
        return warning
    costs = []
    for group in ('peak', 'reserve'):
        counts = items.get(group)
        if isinstance(counts, dict):
            costs.extend((size, f'{group}.{name}') for name, size in counts.items() if type(size) is int and size > 0)
    if costs:
        largest = sorted(costs, key=lambda pair: (-pair[0], pair[1]))[:2]
        warning += ', largest: ' + ', '.join(f'{label}={size:,} B' for size, label in largest)
    return warning


def default_workers():
    """Bound process parallelism by affinity and every visible cgroup-v2 ancestor quota."""
    try:
        count = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        count = max(1, os.cpu_count() or 1)
    try:
        groups = _PROC_CGROUP.read_text()
        mounts = _PROC_MOUNTINFO.read_text()
    except FileNotFoundError:  # macOS, or procfs unavailable
        return count
    return min([count, *_quota_limits(groups, mounts)])


def _quota_limits(groups, mounts):
    group = next((line[3:] for line in groups.splitlines() if line.startswith('0::')), None)
    if group is None:
        return []
    limits = []
    for current in _cgroup_ancestors(group, mounts, 'cgroup2'):
        try:
            quota, period = (current / 'cpu.max').read_text().split()
            if quota != 'max':
                limits.append(max(1, int(quota) // int(period)))
        except FileNotFoundError:
            pass
    return limits


def _memory_limits(groups, mounts):
    for line in groups.splitlines():
        fields = line.split(':', 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, group = fields
        if hierarchy == '0' and not controllers:
            filename = 'memory.max'
            ancestors = _cgroup_ancestors(group, mounts, 'cgroup2')
        elif 'memory' in controllers.split(','):
            filename = 'memory.limit_in_bytes'
            ancestors = _cgroup_ancestors(group, mounts, 'cgroup', controller='memory')
        else:
            continue
        for current in ancestors:
            try:
                value = (current / filename).read_text().strip()
                if value == 'max':
                    continue
                limit = int(value)
            except (OSError, ValueError):
                continue
            if limit >= 0 and (filename == 'memory.max' or limit < _V1_UNLIMITED_MIN):
                yield limit


def _decode_mount_path(value):
    return Path(value.replace('\\040', ' ').replace('\\011', '\t').replace('\\012', '\n').replace('\\134', '\\'))


def _cgroup_ancestors(group, mounts, filesystem, controller=None):
    group_path = Path(group)
    for line in mounts.splitlines():
        left, sep, right = line.partition(' - ')
        source = right.split()
        fields = left.split()
        if not sep or len(fields) < 5 or len(source) < 3 or source[0] != filesystem:
            continue
        if controller is not None and controller not in source[2].split(','):
            continue
        root, mount = _decode_mount_path(fields[3]), _decode_mount_path(fields[4])
        try:
            relative = group_path.relative_to(root)
        except ValueError:
            if group != '/':
                continue
            relative = Path('.')  # cgroup namespace exposes its root as "/"
        # Do not escape the mounted hierarchy for namespace-relative /../ paths.
        if '..' in relative.parts:
            relative = Path('.')
        current = mount / relative
        while True:
            yield current
            if current == mount:
                break
            current = current.parent


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
