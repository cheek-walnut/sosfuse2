"""sosfuse2 geometry replacement for the legacy helper, which imports the
original sosfuse.geometry. Same requirement: build the exact
endpoint cache on the CPU without importing CuPy, via sosfuse2.geometry.

layers=1: ResidentGeometry (cache identity v2, packed layout v3; signature includes the scale height).
layers=2: ContractTwoLayerGeometry = TwoLayerGeometry whose COMMON-layer
          signature hashes geometry_contract.py instead of geometry.py (M2).
          ensure_column() writes this state's aerosol column before the worker
          (or the CPU prefetch helper) hands over, so prepare() only loads it.
"""
import copy
import fcntl
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def _contract_class():
    import numpy as np
    from . import geometry as gm
    from . import geometry_contract as gc

    class ContractTwoLayerGeometry(gm.TwoLayerGeometry):
        """TwoLayerGeometry with the common-layer signature (geometry.py:345-354 otherwise identical)."""
        _version = gc.SIGNATURE_VERSION

        def __init__(self, s, scope, cache):
            gc.check(gm)
            gm.TransientGeometry.__init__(self, s, scope)
            if s.kernel.xp is np and not self._host_test:
                raise ValueError('E_RESIDENT_BACKEND: production residency requires CUDA')
            content, signature = gc.identity(s); _, legacy = gm.identity(s, 1)
            self.root = Path(cache); self.cache = self.root/('endpoints_'+signature); self.cache.mkdir(parents=True, exist_ok=True)
            self.legacy = self.find_legacy(legacy)
            tmp = self.cache/f'identity.json.{os.getpid()}.tmp'
            tmp.write_text(json.dumps(content, indent=2)+'\n'); os.replace(tmp, self.cache/'identity.json')
            self.stats.update(scope='endpoint-resident option, no samples', signature=signature, legacy_signature=legacy,
                              build_seconds=0., upload_seconds=0., load_seconds=0., legacy_cells_read=0,
                              layers=2, geometry_contract=gc.VERSION)
            self._host_offsets = None
    return ContractTwoLayerGeometry


def host_geometry(s, scope, cache, layers=1):
    import numpy as np
    from .geometry import ResidentGeometry
    from .stratosphere import attach
    attach(s,scope)
    cls = _contract_class() if int(layers) == 2 else ResidentGeometry
    # Same shallow-host trick as the rig helper: pass the residency guard
    # without CUDA, then restore NumPy before any cell work.
    host = copy.copy(s); host.kernel = copy.copy(s.kernel); host.kernel.xp = object()
    geometry = cls(host, scope, cache)
    host.kernel.xp = np
    return geometry


def cache_complete(geometry):
    return geometry.packed().complete()


def column_ready(geometry):
    """layers=2: True if this state's aerosol column file exists (layers=1: always True)."""
    return not hasattr(geometry, 'column_path') or geometry.column_path(geometry.s.kernel.height).is_file()


def ensure_column(geometry):
    """layers=2 only: write aerosol/H_<hex>.f4 with the same rule as
    TwoLayerGeometry.prepare() (geometry.py:478-498) without any upload.
    A per-height lock serialises writers; tmp names are per process.
    Returns 'cached' | 'common layer (same height)' | 'computed' | None (layers=1)."""
    if not hasattr(geometry, 'column_path'):
        return None
    import numpy as np
    from .geometry import aerosol_column
    from .tropo_profile import same_source, column_function
    height = geometry.s.kernel.height; path = geometry.column_path(height)
    path.parent.mkdir(exist_ok=True)
    with (path.parent/(path.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            return 'cached'
        host, _ = geometry.host_arrays()
        source = json.loads((geometry.cache/'source_height.json').read_text())['height']
        started = time.monotonic()
        if same_source(geometry.s, source, height):
            column = np.array(host['end_col'][:, 1]); how = 'common layer (same height)'
        else:
            column = aerosol_column(host, height, profile=column_function(geometry.s)); how = 'computed'
        if column.shape != (len(host['dirs']),):
            raise RuntimeError('E_AEROSOL_COLUMN: wrong length')
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp'); column.astype('f4').tofile(tmp); os.replace(tmp, path)
        print(f'CPU_AEROSOL_COLUMN height={height} how={how} seconds={time.monotonic()-started:.2f}', flush=True)
        return how


def build_cells(geometry, count=None):
    """Build/resume the packed cache (count is accepted for interface parity; the
    packed cache is always built for the whole grid).  Returns array bytes."""
    started = time.monotonic(); geometry.build_packed()
    layout = json.loads((geometry.packed().folder/'layout.json').read_text())
    import numpy as np
    total = sum(int(np.prod(shape))*np.dtype(layout['dtypes'][k]).itemsize for k, shape in layout['shapes'].items())
    print(f'CPU_ENDPOINTS packed cells={layout["cells"]} bytes={total} seconds={time.monotonic()-started:.1f}', flush=True)
    return total
