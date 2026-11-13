"""Device profile: per-machine launch budget and launch splitting (configs/devices.json).

Each profile sets a kernel launch budget and optional launch splitting.
The budgets live in configs/devices.json.

Selection order: SOSFUSE2_DEVICE_PROFILE, else the CUDA device name against each profile's gpu_names,
else 'rig' for an unmatched CUDA device and 'mac' without CUDA. Single fields can be overridden by
SOSFUSE2_LAUNCH_BUDGET_SECONDS, SOSFUSE2_LAUNCH_TARGET_SECONDS ('0' or '' = off), SOSFUSE2_LAUNCH_CELLS
(fixed cells per launch, disables the adaptive split) and SOSFUSE2_LAUNCH_PAUSE_MS.
"""

import json
import math
import os
from pathlib import Path

CONFIG = Path(__file__).resolve().with_name('configs') / 'devices.json'


def _number(name, value, allow_none=True):
    if value is None or (isinstance(value, str) and value.strip().lower() in ('', 'none', 'off', '0')):
        if allow_none:
            return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'E_DEVICE_PROFILE: invalid {name}={value!r}') from exc
    if not math.isfinite(out) or out < 0:
        raise ValueError(f'E_DEVICE_PROFILE: invalid {name}={value!r}')
    return out


def profiles():
    data = json.loads(CONFIG.read_text())
    if data.get('schema') != 1 or not isinstance(data.get('profiles'), dict):
        raise ValueError('E_DEVICE_PROFILE: configs/devices.json schema 1 required')
    return data['profiles']


def select(device_name=None):
    """Return (name, profile dict with overrides applied, reason)."""
    table = profiles()
    forced = os.environ.get('SOSFUSE2_DEVICE_PROFILE', '').strip()
    if forced:
        if forced not in table:
            raise ValueError(f'E_DEVICE_PROFILE: unknown SOSFUSE2_DEVICE_PROFILE={forced!r}; choose {sorted(table)}')
        name, reason = forced, 'env SOSFUSE2_DEVICE_PROFILE'
    elif device_name is None:
        name, reason = 'mac', 'no CUDA device'
    else:
        name = next((k for k, v in table.items() if device_name in v.get('gpu_names', ())), 'rig')
        reason = f'CUDA device {device_name!r}' + ('' if name != 'rig' else ' (no profile lists it)')
    p = dict(table[name])
    budget = os.environ.get('SOSFUSE2_LAUNCH_BUDGET_SECONDS')
    if budget is not None:
        p['launch_budget_seconds'] = _number('SOSFUSE2_LAUNCH_BUDGET_SECONDS', budget)
    target = os.environ.get('SOSFUSE2_LAUNCH_TARGET_SECONDS')
    if target is not None:
        p['launch_target_seconds'] = _number('SOSFUSE2_LAUNCH_TARGET_SECONDS', target)
    pause = os.environ.get('SOSFUSE2_LAUNCH_PAUSE_MS')
    if pause is not None:
        p['launch_pause_ms'] = _number('SOSFUSE2_LAUNCH_PAUSE_MS', pause, allow_none=False)
    cells = os.environ.get('SOSFUSE2_LAUNCH_CELLS', '').strip()
    if cells:
        if not cells.isdigit():
            raise ValueError('E_LAUNCH_CELLS: SOSFUSE2_LAUNCH_CELLS must be a non-negative integer')
        p['launch_cells_fixed'] = int(cells)
    else:
        p['launch_cells_fixed'] = None
    for key in ('launch_budget_seconds', 'launch_target_seconds'):
        p[key] = _number(key, p.get(key))
    if (
        p['launch_target_seconds'] is not None
        and p['launch_budget_seconds'] is not None
        and p['launch_target_seconds'] > p['launch_budget_seconds']
    ):
        raise ValueError('E_DEVICE_PROFILE: launch_target_seconds exceeds launch_budget_seconds')
    p['launch_cells_initial'] = int(p.get('launch_cells_initial') or 0)
    p['launch_pause_ms'] = float(p.get('launch_pause_ms') or 0)
    return name, p, reason


def cuda_device_name(xp):
    runtime = getattr(getattr(xp, 'cuda', None), 'runtime', None)
    if runtime is None:  # host-only test doubles (FakeXP): no device, no profile limits
        return None
    props = runtime.getDeviceProperties(runtime.getDevice())
    name = props.get('name', b'')
    return name.decode() if isinstance(name, bytes) else str(name)


def pool_limit_bytes(free):
    """Cap for the CuPy default memory pool, in bytes: the free device memory minus 1 GiB, or SOSFUSE2_POOL_LIMIT_GIB
    (GiB, a positive number) when that is smaller. Up to 1.3.4 the cap was min(20 GiB, free - 1 GiB) at every site, which
    refused a state that needed more than 20 GiB on any card (E_PERSISTENT_VRAM_ADMISSION, a 24 GB state on a 48 GB card)."""
    room = int(free) - 1024**3
    value = os.environ.get('SOSFUSE2_POOL_LIMIT_GIB', '').strip()
    if not value:
        return room
    gib = _number('SOSFUSE2_POOL_LIMIT_GIB', value, allow_none=False)
    if gib * 1024**3 >= room:  # float compare first: a huge value would overflow int()
        return room
    limit = int(gib * 1024**3)
    if limit < 1:  # set_limit(size=0) removes the limit in CuPy, the opposite of what was asked
        raise ValueError(f'E_DEVICE_PROFILE: invalid SOSFUSE2_POOL_LIMIT_GIB={value!r}')
    return limit


def reserved_field_bytes(field_bytes):
    """Bytes the persistent service reserves for a state's source fields next to its payload arrays when it admits the
    state (pipeline_execution.handle): SOSFUSE2_FIELD_RESERVE fields (a positive number), default 30 (the 1.3.x value,
    computed in integers as before). A smaller value admits larger states or more states at once; the admission is a
    reservation, so too small a value turns a wait into an out-of-memory error during the solve."""
    value = os.environ.get('SOSFUSE2_FIELD_RESERVE', '').strip()
    if not value:
        return 30 * int(field_bytes)
    k = _number('SOSFUSE2_FIELD_RESERVE', value, allow_none=False)
    out = k * int(field_bytes)
    if k <= 0 or not math.isfinite(out):
        raise ValueError(f'E_DEVICE_PROFILE: invalid SOSFUSE2_FIELD_RESERVE={value!r}')
    return int(out)


def smi_device_name():
    """Visible GPU name from nvidia-smi without initializing CUDA (CPU-side workers); None if unavailable."""
    import subprocess

    dev = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0]
    try:
        out = (
            subprocess.run(
                ['nvidia-smi', '-i', dev, '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
        return out[0].strip() if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def first_light_split_error(flags, strato, profile_name, profile):
    """Reject first-light configurations unable to split below one cell on guarded devices."""
    if not profile.get('requires_split'):
        return None
    from .kernel_config import resolve_config

    if resolve_config(flags).split_first:
        return None
    return (
        f'E_LAUNCH_BUDGET_CONFIG: the {profile_name} profile needs first light split below one cell; '
        'the instrumentation role uses unsplit transport. Not launched.'
    )


def apply_pool_limit(xp):
    """Apply the memory pool ceiling and return (limit, free), in bytes."""
    free, _ = xp.cuda.runtime.memGetInfo()
    if free < 4 * 1024**3:
        raise RuntimeError('E_VRAM_FREE: need at least 4 GiB free')
    limit = pool_limit_bytes(free)
    xp.get_default_memory_pool().set_limit(size=limit)
    return limit, free
