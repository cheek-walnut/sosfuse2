"""Package-relative locations and the writable work root (public split; provenance keys: docs/PROVENANCE_V3.md).

Three kinds of paths, never mixed:
  read-only package resources  -> PKG, ASSETS, FIXTURES, ACCEPTED, NATIVE, CONFIGS, GATE
  writable caches and outputs  -> work_root()  (SOSFUSE2_WORK_ROOT, else $XDG_CACHE_HOME/sosfuse2, else ~/.cache/sosfuse2)
  sky-sos research extras      -> sky_sos_root()  (SOSFUSE2_SKY_SOS_ROOT; None when the package runs on its own)
Provenance keys are package-relative ('sosfuse2/solve/source_solver.py'), see rel().

The package must be installed as regular files (no zip import): the native source is handed to a C++ compiler and the
solver definitions are read as text. resource_root() raises E_PACKAGE_NOT_ON_DISK otherwise.
"""

import os
from importlib import resources
from pathlib import Path


def resource_root():
    root = resources.files('sosfuse2')
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
FIXTURES = PKG / 'checks/fixtures'
ACCEPTED = PKG / 'solve'  # context factory location; source_solver.load_engine retains isolated engine module copies
NATIVE = PKG / 'backends/native'
CONFIGS = PKG / 'backends/configs'
GATE = PKG / 'checks/gate_data'
KERNEL_CU = PKG / 'backends/kernel.cu'
SRC = PKG.parent  # directory that makes `import sosfuse2` work (src/ in-tree, site-packages when installed)


def work_root():
    """Writable root for caches, builds and run outputs. Created on first use. A root that cannot be located or
    created is E_WORK_ROOT with the path and the underlying error."""
    env = os.environ.get('SOSFUSE2_WORK_ROOT', '').strip()
    xdg = os.environ.get('XDG_CACHE_HOME', '').strip()
    where = env or (xdg + '/sosfuse2' if xdg else '~/.cache/sosfuse2')
    try:
        if env:
            root = Path(env).expanduser()
        else:
            root = (Path(xdg) if xdg else Path.home() / '.cache') / 'sosfuse2'
        where = str(root)
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f'E_WORK_ROOT: {where}: {type(exc).__name__}: {exc}') from exc


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


# Production modules that enter the provenance format (content_identity.package_hashes) and the gate bless identity
# (gate.code_identity): a fixed list, not a glob, so a file that goes missing is an error instead of a silently different
# signature. Each module enters by its identity text (content_identity.python_text: the source without comments and
# docstrings), not by its bytes. test_*.py and checks/testkit.py (test support) are excluded; numerical_checks stays
# (stratosphere_checks imports it). test_ozone_temperature.test_03c keeps this list equal to the tree: adding a module
# means adding it here.
PRODUCTION_MODULES = (
    '__init__',
    '__main__',
    'foundation/__init__',
    'atmosphere/__init__',
    'rays/__init__',
    'precompute/__init__',
    'backends/__init__',
    'solve/__init__',
    'outputs/__init__',
    'jobs/__init__',
    'checks/__init__',
    'asset_tools/__init__',
    'foundation/available_cpus',
    'backends/flags_env',
    'solve/engine',
    'outputs/pack_state',
    'atmosphere/air_wavelength',
    'backends/arithmetic',
    'jobs/cache_store',
    'precompute/column_cache',
    'precompute/column_parallel',
    'rays/column_table',
    'backends/compile_cache',
    'atmosphere/components',
    'atmosphere/density',
    'backends/device_profile',
    'solve/source_solver',
    'checks/gate',
    'checks/kernel_diagnostics',
    'asset_tools/generate_density_profile',
    'asset_tools/generate_minor_absorbers',
    'asset_tools/generate_strato_optics',
    'asset_tools/generate_tropo_optics',
    'solve/ray_geometry',
    'precompute/endpoint_contract',
    'solve/ray_geometry_cpu',
    'solve/ray_geometry_parallel',
    'precompute/ground_parallel',
    'solve/ground_sh',
    'precompute/host_cache',
    'foundation/host_limits',
    'foundation/content_identity',
    'asset_tools/iprt',
    'backends/flags',
    'solve/ksplit',
    'asset_tools/mie',
    'atmosphere/minor_absorbers',
    'checks/numerical_checks',
    'atmosphere/ozone_temperature',
    'foundation/paths',
    'asset_tools/prepare_minor_absorbers',
    'rays/shell_segments',
    'atmosphere/extinction',
    'rays/incoming_quadrature',
    'rays/incoming_rings',
    'rays/zenith_interpolation',
    'solve/engine_core',
    'outputs/preview_functions',
    'jobs/pipeline',
    'jobs/pipeline_execution',
    'jobs/plan_queue',
    'jobs/precompile',
    'checks/validation_evidence',
    'rays/ray_columns',
    'atmosphere/rayleigh',
    'backends/numpy_reference',
    'checks/solver_diagnostics',
    'jobs/run_state',
    'jobs/walkthrough',
    'backends/sh_layout',
    'atmosphere/spectral',
    'solve/strato_sh_aux',
    'asset_tools/strato_states',
    'atmosphere/stratosphere',
    'precompute/sun_parallel',
    'outputs/radiance_table',
    'outputs/quantized_table',
    'jobs/tables_from_sources',
    'foundation/timers',
    'atmosphere/tropo_mie',
    'atmosphere/tropo_profile',
    'checks/validate',
    'checks/stratosphere_checks',
    'solve/ksplit_tolerance',
)


def production_files(exclude=()):
    """Paths of PRODUCTION_MODULES (minus `exclude` names); a missing file is E_PACKAGE_FILE_MISSING, never a gap."""
    files = [PKG / f'{name}.py' for name in PRODUCTION_MODULES if name not in set(exclude)]
    missing = [p.relative_to(PKG).as_posix() for p in files if not p.is_file()]
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
