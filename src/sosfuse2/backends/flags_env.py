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


def load(name):
    from sosfuse2.backends.flags import RETIRED_FLAGS, resolve_config

    path = HERE / f'{name}.json'
    if not path.is_file():  # the stderr code keeps `eval "$(...)"` from passing silently on an empty stdout
        raise SystemExit(f'E_CONFIG_UNKNOWN: {name!r}: no backends/configs/{name}.json')
    c = json.loads(path.read_text())
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
