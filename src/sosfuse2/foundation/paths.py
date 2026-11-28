"""Package-relative locations and the writable work root (public split; provenance keys: docs/PROVENANCE_V3.md).

Three kinds of paths, never mixed:
  read-only package resources  -> PKG, ASSETS, FIXTURES, ACCEPTED, NATIVE, CONFIGS, GATE
  writable caches and outputs  -> work_root()  (SOSFUSE2_WORK_ROOT, else $XDG_CACHE_HOME/sosfuse2, else ~/.cache/sosfuse2)
  upstream research extras      -> sky_sos_root()  (SOSFUSE2_SKY_SOS_ROOT; None when the package runs on its own)
Provenance keys are package-relative ('sosfuse2/solve/source_solver.py'), see rel().

The package must be installed as regular files (no zip import): the native source is handed to a C++ compiler and the
solver definitions are read as text. resource_root() raises E_PACKAGE_NOT_ON_DISK otherwise.
"""

import os
from importlib import resources
from pathlib import Path

from . import dependencies


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
KERNEL_CU = PKG / dependencies.CUDA_SOURCE
SRC = PKG.parent  # directory that makes `import sosfuse2` work (src/ in-tree, site-packages when installed)


def work_root(*, create=True):
    """Writable root for caches, builds and run outputs. Created on first use. A root that cannot be located or
    created is E_WORK_ROOT with the path and the underlying error. create=False only locates it, for lookups that
    read an optional file there and must not create the root as a side effect."""
    env = os.environ.get('SOSFUSE2_WORK_ROOT', '').strip()
    xdg = os.environ.get('XDG_CACHE_HOME', '').strip()
    where = env or (xdg + '/sosfuse2' if xdg else '~/.cache/sosfuse2')
    try:
        if env:
            root = Path(env).expanduser()
        else:
            root = (Path(xdg) if xdg else Path.home() / '.cache') / 'sosfuse2'
        where = str(root)
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root.resolve()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f'E_WORK_ROOT: {where}: {type(exc).__name__}: {exc}') from exc


def sky_sos_root():
    """The upstream research checkout when the package is used from inside it; None for a stand-alone install (variable unset).
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


# Keep the historical suffix-free module names for callers of production_files().
PRODUCTION_MODULES = tuple(name.removesuffix('.py') for name in dependencies.FILES if name.endswith('.py'))


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
