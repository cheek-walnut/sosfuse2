#!/usr/bin/env python3
"""Print shell exports for a named sosfuse2 kernel config (sosfuse2/backends/configs/<name>.json).

usage: flags=$(python -m sosfuse2.backends.flags_env C3) && eval "$flags"
Checks: active flags are validated by flags, none is automatic or diagnostic; retired names warn and disappear,
block size 128/256, and the config's launch rule (single_launch) agrees with SOSFUSE2_LAUNCH_CELLS.
Physical dependency checks run when source_solver.Kernel attaches the state.

The same settings can be typed as options of jobs/run_state.py and jobs/pipeline.py: add_run_options adds
--config, --ground-sh, --device-profile, --compile-cache and --attach-cache, and apply_run_options writes each given
option to the environment variable that the package reads, so an option takes precedence over the variable and the
child processes inherit it. run_options_environment does the same for the duration of one call and restores the
variables afterwards.

resolve_backend fills in what was not typed: --backend auto (the default) is cupy when nvidia-smi names a visible
device and numpy on a machine without CUDA (cuda_device_present), and numpy selects the reduced grid, the only one it solves."""

import contextlib
import importlib.util
import json, os, shutil, sys
from pathlib import Path

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

    if '/' in name or '\\' in name:
        raise SystemExit(f'E_CONFIG_UNKNOWN: {name!r}: expected a kernel config name')
    path = HERE / f'{name}.json'
    if not path.is_file():
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


def add_run_options(parser):
    """Options for the settings a full run usually changes; each one sets an environment variable (apply_run_options)."""
    parser.add_argument(
        '--config',
        metavar='NAME',
        help='kernel config backends/configs/NAME.json (for instance C3): sets SOSFUSE2_KERNEL_FLAGS and '
        'SOSFUSE2_BLOCK_SIZE',
    )
    parser.add_argument(
        '--ground-sh', type=int, choices=(0, 16, 24, 32), help='ground convolution order; sets SOSFUSE2_GROUND_SH_L'
    )
    parser.add_argument(
        '--device-profile', metavar='NAME', help='profile of backends/configs/devices.json; sets SOSFUSE2_DEVICE_PROFILE'
    )
    parser.add_argument(
        '--compile-cache', metavar='DIR', help='CUDA runtime compilation cache; sets SOSFUSE2_COMPILE_CACHE'
    )
    parser.add_argument(
        '--attach-cache', metavar='DIR|off', help='stratosphere attach cache folder, or off; sets STRATO_ATTACH_CACHE'
    )


def apply_run_options(args, environ):
    """Write the given options of add_run_options into `environ`; an option that was not given leaves its variable as
    it is. A blank --compile-cache or --attach-cache writes an empty value, which selects the default like an empty
    variable. Raises ValueError for an unknown config and for a --block-size that differs from the config's."""
    if args.config is not None:
        try:
            config = load(args.config)
        except SystemExit as e:
            raise ValueError(str(e)) from None
        block = getattr(args, 'block_size', None)
        if block is not None and block != config['block_size']:
            raise ValueError(
                f"E_KERNEL_BLOCK_SIZE: --block-size {block} differs from block_size {config['block_size']} "
                f'of --config {args.config}'
            )
        environ['SOSFUSE2_KERNEL_FLAGS'] = ','.join(sorted(config['kernel_flags']))
        environ['SOSFUSE2_BLOCK_SIZE'] = str(config['block_size'])
    if args.ground_sh is not None:
        environ['SOSFUSE2_GROUND_SH_L'] = str(args.ground_sh)
    if args.device_profile is not None:
        environ['SOSFUSE2_DEVICE_PROFILE'] = args.device_profile
    if args.compile_cache is not None:
        text = args.compile_cache.strip()
        environ['SOSFUSE2_COMPILE_CACHE'] = str(Path(text).resolve()) if text else ''
    if args.attach_cache is not None:
        text = args.attach_cache.strip()
        environ['STRATO_ATTACH_CACHE'] = 'off' if text.lower() == 'off' else str(Path(text).resolve()) if text else ''


def cuda_device_present():
    """Whether --backend auto has a CUDA device to use, without initializing CUDA. False on a Mac, when
    CUDA_VISIBLE_DEVICES is set and blank (every device hidden), and when the machine has neither nvidia-smi nor CuPy.
    True when nvidia-smi names the visible device and CuPy is installed. The cases in between are E_BACKEND_AUTO, so
    the choice between a full CUDA table and a reduced NumPy table is not made silently: nvidia-smi without an
    answering device, CuPy installed without nvidia-smi on PATH, and a device without CuPy."""
    if sys.platform == 'darwin':
        return False
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and not visible.strip():
        return False
    if shutil.which('nvidia-smi') is None:
        if importlib.util.find_spec('cupy') is None:
            return False
        raise ValueError(
            'E_BACKEND_AUTO: CuPy is installed but nvidia-smi is not on PATH; pass --backend cupy or --backend numpy'
        )
    from sosfuse2.backends.device_profile import smi_device_name

    if smi_device_name() is None:
        raise ValueError(
            'E_BACKEND_AUTO: nvidia-smi is installed but names no visible device; pass --backend cupy or --backend numpy'
        )
    if importlib.util.find_spec('cupy') is None:
        raise ValueError(
            'E_BACKEND_AUTO: nvidia-smi names a CUDA device but CuPy is not installed; install the gpu extra '
            '(pip install ".[gpu]") or pass --backend numpy'
        )
    return True


def resolve_backend(backend, reduced, *, full_numpy=False, cuda=None):
    """(backend, reduced, notes) for the typed --backend and --reduced. 'auto' becomes cupy when `cuda()` (default
    cuda_device_present) is true and numpy otherwise. NumPy solves only the reduced grid, so numpy turns `reduced` on unless `full_numpy` (a phase that
    runs on NumPy at full resolution, such as --geometry-only). Typed values are kept. `notes` are the lines to show
    the person for each value that was filled in."""
    notes = []
    if backend == 'auto':
        backend = 'cupy' if (cuda or cuda_device_present)() else 'numpy'
        notes.append(f'--backend auto: {backend} ' + ('(CUDA device found)' if backend == 'cupy' else '(no CUDA device)'))
    if backend == 'numpy' and not reduced and not full_numpy:
        reduced = True
        notes.append('--backend numpy solves the reduced grid: --reduced is on (full resolution needs CUDA)')
    return backend, reduced, notes


RUN_OPTION_VARIABLES = (
    'SOSFUSE2_KERNEL_FLAGS',
    'SOSFUSE2_BLOCK_SIZE',
    'SOSFUSE2_GROUND_SH_L',
    'SOSFUSE2_DEVICE_PROFILE',
    'SOSFUSE2_COMPILE_CACHE',
    'STRATO_ATTACH_CACHE',
)


@contextlib.contextmanager
def run_options_environment(args, environ):
    """apply_run_options for the duration of the block; the variables it can write are restored afterwards, also when
    the options or the block raise."""
    before = {name: environ.get(name) for name in RUN_OPTION_VARIABLES}
    try:
        apply_run_options(args, environ)
        yield
    finally:
        for name, value in before.items():
            if value is None:
                environ.pop(name, None)
            else:
                environ[name] = value


if __name__ == '__main__':
    c = load(sys.argv[1] if len(sys.argv) > 1 else 'C3')
    print(f"export SOSFUSE2_BLOCK_SIZE={c['block_size']} SOSFUSE2_KERNEL_FLAGS={','.join(sorted(c['kernel_flags']))}")
