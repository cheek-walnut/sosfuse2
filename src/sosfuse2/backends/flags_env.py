#!/usr/bin/env python3
"""Print shell exports for a named sosfuse2 kernel config (sosfuse2/backends/configs/<name>.json).

usage: eval "$(python -m sosfuse2.backends.flags_env C3)"
Checks: active flags are validated by flags, none is automatic or diagnostic; retired names warn and disappear,
block size 128/256, and the config's launch rule (single_launch) agrees with SOSFUSE2_LAUNCH_CELLS.
Physical dependency checks run when source_solver.Kernel attaches the state."""

import json, os, sys
from sosfuse2.foundation import paths

HERE = paths.CONFIGS
AUTOMATIC = {'STRATO', 'TROPO_MIE', 'AEROSOL_2L'}
DIAGNOSTIC = {'EXP_SECTION_CLOCKS'}  # wrong-physics / observer-only flags


def _kernel_config(path):
    """The parsed config, or None when the file is not readable JSON with kernel_flags and block_size."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and 'kernel_flags' in data and 'block_size' in data else None


def load(name):
    from sosfuse2.backends.flags import RETIRED_FLAGS, resolve_config

    path = HERE / f'{name}.json'
    if not path.is_file():  # the stderr code keeps `eval "$(...)"` from passing silently on an empty stdout
        raise SystemExit(f'E_CONFIG_UNKNOWN: {name!r}: no backends/configs/{name}.json')
    c = _kernel_config(path)
    if c is None:  # e.g. devices.json in the same folder
        kernels = sorted(p.stem for p in HERE.glob('*.json') if _kernel_config(p) is not None)
        raise SystemExit(f'E_CONFIG_UNKNOWN: {name!r}: backends/configs/{name}.json is not a kernel config '
                         f'(kernel configs: {", ".join(kernels)})')
    flags = c['kernel_flags']
    resolved = resolve_config(flags)
    bs = c['block_size']
    bad = [f for f in resolved.flags if f in AUTOMATIC or f in DIAGNOSTIC]
    active = [f for f in flags if f not in RETIRED_FLAGS]
    if bad or len(set(active)) != len(active):
        raise SystemExit(f'E_CONFIG_FLAGS: {name}: {bad or "duplicates"}')
    if bs not in (128, 256):
        raise SystemExit(f'E_CONFIG_BLOCK: {name}: {bs}')
    if c.get('single_launch') and os.environ.get('SOSFUSE2_LAUNCH_CELLS', '').strip() not in ('', '0'):
        raise SystemExit(f'E_CONFIG_LAUNCH_CELLS: {name} needs one launch per cell grid; unset SOSFUSE2_LAUNCH_CELLS')
    c['kernel_flags'] = list(resolved.flags)
    return c


if __name__ == '__main__':
    c = load(sys.argv[1] if len(sys.argv) > 1 else 'C3')
    print(f"export SOSFUSE2_BLOCK_SIZE={c['block_size']} SOSFUSE2_KERNEL_FLAGS={','.join(sorted(c['kernel_flags']))}")
