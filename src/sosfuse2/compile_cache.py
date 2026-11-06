"""Shared NVRTC disk cache; no solver state or numerical data are shared."""

import hashlib
import json
import os
from pathlib import Path
from .kernel_config import resolve_config, deterministic


def cache_key(gpu_name, compiler, block_size=None):
    here = Path(__file__).resolve().parent
    # Kernel sources only (2026-10-07): CuPy keys every entry by the exact source text, NVRTC options, architecture and
    # compiler, so host-side Python edits need not orphan a deployment's compiled kernels.
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob('*.cu'))}
    identity = dict(
        deterministic=deterministic(),
        sources=sources,
        gpu_name=gpu_name,
        compiler=compiler,
        flags=list(
            resolve_config(
                f for f in os.environ.get('SOSFUSE2_KERNEL_FLAGS', '').replace(' ', '').split(',') if f
            ).flags
        ),
        block_size=int(block_size or os.environ.get('SOSFUSE2_BLOCK_SIZE', '128')),
        automatic_flags=['STRATO', 'TROPO_MIE', 'AEROSOL_2L'],
        note='CuPy also keys each entry by exact source/options/architecture/compiler',
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]


def configure(root, backend='cupy', block_size=None):
    """SOSFUSE2_COMPILE_CACHE=<dir> replaces the per-run root with a deployment-level one (2026-10-07): a new
    work root then reuses the NVRTC binaries of earlier runs with the same sources, flags and toolchain."""
    if backend != 'cupy':
        return None
    root = os.environ.get('SOSFUSE2_COMPILE_CACHE', '').strip() or root
    import cupy

    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    name = props['name']
    name = name.decode() if isinstance(name, bytes) else str(name)
    compiler = dict(
        cupy=cupy.__version__,
        nvrtc=cupy.cuda.nvrtc.getVersion(),
        runtime=cupy.cuda.runtime.runtimeGetVersion(),
        driver=cupy.cuda.runtime.driverGetVersion(),
    )
    path = Path(root) / cache_key(name, compiler, block_size)
    path.mkdir(parents=True, exist_ok=True)
    os.environ['CUPY_CACHE_DIR'] = str(path.resolve())
    print('COMPILE_CACHE ' + json.dumps(dict(path=str(path.resolve()), gpu=name, compiler=compiler)), flush=True)
    return path
