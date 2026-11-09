"""Package-relative locations and the writable work root (public split; provenance keys: docs/PROVENANCE_V3.md).

Three kinds of paths, never mixed:
  read-only package resources  -> PKG, ASSETS, FIXTURES, ACCEPTED, NATIVE, CONFIGS, GATE
  writable caches and outputs  -> work_root()  (SOSFUSE2_WORK_ROOT, else $XDG_CACHE_HOME/sosfuse2, else ~/.cache/sosfuse2)
  sky-sos research extras      -> sky_sos_root()  (SOSFUSE2_SKY_SOS_ROOT; None when the package runs on its own)
Provenance keys are package-relative ('sosfuse2/fused.py'), see rel().

The package must be installed as regular files (no zip import): the native source is handed to a C++ compiler and the
solver definitions are read as text. resource_root() raises E_PACKAGE_NOT_ON_DISK otherwise.
"""

import os
from importlib import resources
from pathlib import Path


def resource_root():
    root = resources.files(__package__ or 'sosfuse2')
    try:
        p = Path(str(root))
    except TypeError:
        p = None
    if p is None or not p.is_dir():
        raise RuntimeError(
            'E_PACKAGE_NOT_ON_DISK: sosfuse2 must be installed as plain files (accepted sources and sh_cpu.cpp are read from disk)'
        )
    return p.resolve()


PKG = resource_root()
ASSETS = PKG / 'assets'
FIXTURES = PKG / 'fixtures'
ACCEPTED = PKG / 'accepted'  # loader location, retained for fused.load_engine
NATIVE = PKG / 'native'
CONFIGS = PKG / 'configs'
GATE = PKG / 'gate'
SRC = PKG.parent  # directory that makes `import sosfuse2` work (src/ in-tree, site-packages when installed)


def work_root():
    """Writable root for caches, builds and run outputs. Created on first use."""
    env = os.environ.get('SOSFUSE2_WORK_ROOT', '').strip()
    if env:
        root = Path(env).expanduser()
    else:
        xdg = os.environ.get('XDG_CACHE_HOME', '').strip()
        root = (Path(xdg) if xdg else Path.home() / '.cache') / 'sosfuse2'
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def sky_sos_root():
    """The sky-sos research checkout when the package is used from inside it; None for a stand-alone install (variable unset).
    Only optional extras live there (legacy selftest fixture, results/ records read by research tests).
    A value that is not a directory is an error (E_SKY_SOS_ROOT), never a silent "no extras"."""
    env = os.environ.get('SOSFUSE2_SKY_SOS_ROOT', '').strip()
    if not env:
        return None
    p = Path(env).expanduser().resolve()
    if not p.is_dir():
        raise RuntimeError(f'E_SKY_SOS_ROOT: {env!r} is not a directory (SOSFUSE2_SKY_SOS_ROOT)')
    return p


def child_env(env=None):
    """Environment for a child Python that must import this package: SRC is put first on PYTHONPATH, so an in-tree
    (not installed) checkout works the same as an installed one. Children are started as `python -m sosfuse2.<module>`
    (module_command), never by script path: a script's own directory would hide the package."""
    e = dict(os.environ if env is None else env)
    e['PYTHONPATH'] = os.pathsep.join(filter(None, [str(SRC), e.get('PYTHONPATH', '')]))
    return e


def module_command(name, *args):
    import sys

    return [sys.executable, '-B', '-m', 'sosfuse2.' + name, *map(str, args)]


# Production modules whose bytes enter the v4 provenance format (run_gpu_state_v3.package_hashes) and the gate bless identity
# (gate.code_identity): a fixed list, not a glob, so a file that goes missing is an error instead of a silently different
# signature. test_*.py and testing.py (test support) are excluded; numerical_checks stays (validate_strato
# imports it). test_ozone_temperature.test_03c keeps this list equal to the tree: adding a module means adding it here.
PRODUCTION_MODULES = (
    '__init__',
    'air_wavelength',
    'arithmetic',
    'cache_store',
    'column_cache',
    'column_parallel',
    'column_table',
    'compile_cache',
    'components',
    'density',
    'device_profile',
    'fused',
    'gate',
    'gate_flags',
    'generate_density_profile',
    'generate_minor_absorbers',
    'generate_strato_optics',
    'generate_tropo_optics',
    'geometry',
    'geometry_contract',
    'geometry_cpu',
    'geometry_parallel',
    'ground_parallel',
    'ground_sh',
    'host_cache',
    'host_limits',
    'iprt',
    'kernel_config',
    'ksplit',
    'mie',
    'minor_absorbers',
    'numerical_checks',
    'ozone_temperature',
    'paths',
    'prepare_minor_absorbers',
    'solver_columns',
    'solver_optics',
    'solver_sky',
    'solver_legacy',
    'solver_poly',
    'solver_core',
    'solver_render',
    'pipeline',
    'pipeline_execution',
    'plan_queue',
    'precompile',
    'provenance',
    'ray_columns',
    'rayleigh',
    'reference',
    'run_fuse',
    'run_gpu_state_v3',
    'sh_layout',
    'spectral',
    'strato_parallel',
    'strato_states',
    'stratosphere',
    'sun_parallel',
    'table_fast',
    'table_q',
    'tables_from_sources',
    'timers',
    'tropo_mie',
    'tropo_profile',
    'validate',
    'validate_strato',
    'verify_tol',
)


def production_files(exclude=()):
    """Paths of PRODUCTION_MODULES (minus `exclude` names); a missing file is E_PACKAGE_FILE_MISSING, never a gap."""
    files = [PKG / f'{name}.py' for name in PRODUCTION_MODULES if name not in set(exclude)]
    missing = [p.name for p in files if not p.is_file()]
    if missing:
        raise RuntimeError(f'E_PACKAGE_FILE_MISSING: production modules absent from the package: {missing}')
    return files


def rel(path):
    """Provenance key for a file inside the package: 'sosfuse2/<relative posix path>'."""
    p = Path(path).resolve()
    try:
        return 'sosfuse2/' + p.relative_to(PKG).as_posix()
    except ValueError:
        return p.as_posix()
