"""Build the exact endpoint cache on the CPU without importing CuPy.

layers=1: ResidentGeometry (cache identity v2, packed layout v3; signature includes the scale height).
layers=2: ContractTwoLayerGeometry = TwoLayerGeometry whose COMMON-layer
          signature hashes precompute/endpoint_contract.py instead of solve/ray_geometry.py.
          ensure_column() writes this state's aerosol column before the worker
          (or the CPU prefetch helper) hands over, so prepare() only loads it.
"""

import copy
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if __name__ == '__main__':
    sys.dont_write_bytecode = True


def _contract_class():
    import numpy as np
    from . import ray_geometry as gm
    from ..precompute import endpoint_contract as gc

    class ContractTwoLayerGeometry(gm.TwoLayerGeometry):
        """TwoLayerGeometry with the common-layer signature."""

        _version = gc.SIGNATURE_VERSION

        def __init__(self, s, scope, cache):
            gc.check(gm)
            gm.TransientGeometry.__init__(self, s, scope)
            if s.kernel.xp is np and not self._host_test:
                raise ValueError('E_RESIDENT_BACKEND: production residency requires CUDA')
            content, signature = gc.identity(s)
            _, legacy = gm.identity(s, 1)
            self.root = Path(cache)
            self.cache = self.root / ('endpoints_' + signature)
            self.cache.mkdir(parents=True, exist_ok=True)
            self.legacy = self.find_legacy(legacy)
            tmp = self.cache / f'identity.json.{os.getpid()}.tmp'
            tmp.write_text(json.dumps(content, indent=2) + '\n')
            os.replace(tmp, self.cache / 'identity.json')
            self.stats.update(
                scope='endpoint-resident option, no samples',
                signature=signature,
                legacy_signature=legacy,
                build_seconds=0.0,
                upload_seconds=0.0,
                load_seconds=0.0,
                legacy_cells_read=0,
                layers=2,
                geometry_contract=gc.VERSION,
            )
            self._host_offsets = None

    return ContractTwoLayerGeometry


def host_geometry(s, scope, cache, layers=1):
    import numpy as np
    from .ray_geometry import ResidentGeometry
    from ..atmosphere.stratosphere import attach

    attach(s, scope)
    cls = _contract_class() if int(layers) == 2 else ResidentGeometry
    # Pass the residency guard without CUDA, then restore NumPy before cell work.
    host = copy.copy(s)
    host.kernel = copy.copy(s.kernel)
    host.kernel.xp = object()
    geometry = cls(host, scope, cache)
    host.kernel.xp = np
    return geometry


def cache_complete(geometry):
    return geometry.packed().complete() and (
        not hasattr(geometry, 'column_path') or (geometry.cache / 'source_height.json').is_file()
    )


def column_ready(geometry):
    """Whether this state's aerosol column is reusable (layers=1: always True)."""
    if not hasattr(geometry, 'column_path'):
        return True
    rays = _packed_ray_count(geometry)
    return rays is not None and read_column(geometry.column_path(geometry.s.kernel.height), rays) is not None


def _packed_ray_count(geometry):
    try:
        rays = json.loads((geometry.cache / 'packed/layout.json').read_text())['rays']
        return rays if type(rays) is int and rays >= 0 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_column(path, rays):
    """Read one checked snapshot of a raw aerosol column, or return None on a cache miss."""
    import numpy as np

    side = path.with_suffix('.f4.sha256')
    try:
        if path.is_symlink() or side.is_symlink() or not path.is_file() or not side.is_file():
            return None
        digest = side.read_bytes().strip()
        if path.stat().st_size != 4 * rays:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) != 4 * rays or hashlib.sha256(data).hexdigest().encode('ascii') != digest:
        return None
    column = np.frombuffer(data, 'f4')
    return column if np.isfinite(column).all() else None


def ensure_column(geometry):
    """Prepare the aerosol column on the CPU, and return the cache outcome (None for layers=1)."""
    return load_or_build_column(geometry)[1] if hasattr(geometry, 'column_path') else None


def load_or_build_column(geometry, host=None):
    """Load or repair the checksum pair under the shared per-height writer lock."""
    import numpy as np
    from ..atmosphere.tropo_profile import same_source
    from ..foundation.timers import stage

    height = geometry.s.kernel.height
    preparing_on_cpu = host is None
    path = geometry.column_path(height)
    rays = _packed_ray_count(geometry) if host is None else len(host['dirs'])
    if rays is None:
        host, _ = geometry.host_arrays()
        rays = len(host['dirs'])
    source = json.loads((geometry.cache / 'source_height.json').read_text())['height']
    with stage('aerosol_column_load'):
        column = read_column(path, rays)
    if column is not None:
        return column, 'cached', source
    path.parent.mkdir(exist_ok=True)
    with path.with_suffix('.f4.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        column = read_column(path, rays)
        if column is not None:
            return column, 'cached', source
        if host is None:
            host, _ = geometry.host_arrays()
        started = time.monotonic()
        if same_source(geometry.s, source, height):
            column = np.array(host['end_col'][:, 1])
            how = 'common layer (same height)'
        else:
            with stage('aerosol_column_compute'):
                column = geometry.recompute_column(host, height)
            how = 'computed'
        column = np.asarray(column, dtype='f4')
        if column.shape != (len(host['dirs']),):
            raise RuntimeError('E_AEROSOL_COLUMN: wrong length')
        if not np.isfinite(column).all():
            raise RuntimeError('E_AEROSOL_COLUMN: nonfinite values')
        data = column.tobytes()
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        side = path.with_suffix('.f4.sha256')
        side_tmp = tmp.with_suffix('.sha256.tmp')
        try:
            tmp.write_bytes(data)
            side_tmp.write_text(hashlib.sha256(data).hexdigest() + '\n', encoding='ascii')
            os.replace(tmp, path)
            os.replace(side_tmp, side)
        finally:
            tmp.unlink(missing_ok=True)
            side_tmp.unlink(missing_ok=True)
        if preparing_on_cpu:
            print(f'CPU_AEROSOL_COLUMN height={height} how={how} seconds={time.monotonic() - started:.2f}', flush=True)
        return column, how, source


def build_cells(geometry, count=None):
    """Build/resume the packed cache (count is accepted for interface parity; the
    packed cache is always built for the whole grid).  Returns array bytes."""
    started = time.monotonic()
    geometry.build_packed()
    layout = json.loads((geometry.packed().folder / 'layout.json').read_text())
    import numpy as np

    total = sum(int(np.prod(shape)) * np.dtype(layout['dtypes'][k]).itemsize for k, shape in layout['shapes'].items())
    print(
        f'CPU_ENDPOINTS packed cells={layout["cells"]} bytes={total} seconds={time.monotonic() - started:.1f}',
        flush=True,
    )
    return total
