"""A4 checks packed cache arrays against the original v1 reader.

Oracle = the unmodified scripts/sosfuse/geometry.py ResidentGeometry.prepare
(read-only import), run on a NumPy shim instead of CuPy, writing its v1
per-cell cache under the scratch folder.  Then sosfuse2 ResidentGeometry
(a) converts that v1 cache, (b) builds fresh, (c) reloads the complete packed
cache. Bundle and ground arrays must match in dtype and shape and pass within_tol.
"""

import json
import shutil
import sys
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.checks import testkit

_SKY = testkit.sky_sos_root()
if _SKY is None:  # the v1 oracle lives in the upstream checkout (manifest: sibling-import test)
    import pytest

    pytest.skip(f'v1 oracle scripts/sosfuse needs {testkit.ENV}', allow_module_level=True)
sys.path.append(
    str(testkit.sky_sos('scripts', 'sosfuse').parent)
)  # appended: the installed sosfuse2 package stays first
import sosfuse.geometry as old_geometry  # read-only oracle
from sosfuse2.solve import ray_geometry as new_geometry
from sosfuse2.foundation.timers import TIMERS

from sosfuse2.checks.numerical_checks import within_tol


def numpy_shim():
    m = types.ModuleType('numpy_as_device')
    m.__dict__.update(np.__dict__)
    m.asnumpy = np.asarray
    return m


def files_under(path):
    return sum(1 for p in Path(path).rglob('*') if p.is_file())


def same_arrays(a, b):
    out = {}
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        out[k] = bool(x.dtype == y.dtype and x.shape == y.shape and (within_tol(y, x, rtol=1e-3)))
    return out


def run(grid, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    s, scope, _ = new_geometry.create(root / 'prepare', grid, [550.0], 'numpy', 128)
    cache = root / 'cache'
    # Oracle: original prepare() with xp shim; v1 folder uses the current identity(s, 1).
    _, legacy = new_geometry.identity(s, 1)
    oracle = old_geometry.ResidentGeometry.__new__(old_geometry.ResidentGeometry)
    old_geometry.TransientGeometry.__init__(oracle, s, scope)
    oracle.cache = cache / ('endpoints_' + legacy)
    oracle.cache.mkdir(parents=True)
    oracle.stats.update(build_seconds=0.0, upload_seconds=0.0)
    oracle._host_offsets = None
    kernel_xp = s.kernel.xp
    s.kernel.xp = numpy_shim()
    s.kernel.sync = lambda: None
    t = time.perf_counter()
    oracle.prepare()
    v1_build = time.perf_counter() - t
    oracle2 = old_geometry.ResidentGeometry.__new__(old_geometry.ResidentGeometry)
    old_geometry.TransientGeometry.__init__(oracle2, s, scope)
    oracle2.cache = oracle.cache
    oracle2.stats.update(build_seconds=0.0, upload_seconds=0.0)
    oracle2._host_offsets = None
    t = time.perf_counter()
    oracle2.prepare()
    v1_load = time.perf_counter() - t
    del oracle2
    s.kernel.xp = kernel_xp
    del s.kernel.sync
    v1_files = files_under(oracle.cache)
    # Check that the v1 reader wrote a cell completion marker under that identity.
    v1_cell_marker_found = (oracle.cache / 'cell_00000' / 'complete.json').exists()
    new_geometry.ResidentGeometry._host_test = True
    rows = {}
    for label in ('convert_v1', 'reload_packed', 'fresh_build'):
        if label == 'fresh_build':
            shutil.rmtree(cache / ('endpoints_' + new_geometry.identity(s, 2)[1]))
            shutil.move(str(oracle.cache), str(root / 'v1_moved_away'))
        TIMERS.reset()
        t = time.perf_counter()
        geo = new_geometry.ResidentGeometry(s, scope, cache)
        geo.prepare()
        seconds = time.perf_counter() - t
        cmp = same_arrays(oracle.bundle.arrays, geo.bundle.arrays)
        gcmp = same_arrays(oracle.ground_bundle.arrays, geo.ground_bundle.arrays)
        rows[label] = dict(
            seconds=seconds,
            bundle_within_tol=all(cmp.values()),
            ground_within_tol=all(gcmp.values()),
            arrays=cmp,
            legacy_cells_read=geo.stats['legacy_cells_read'],
            packed_files=files_under(geo.cache / 'packed'),
            timers=TIMERS.report()['wall_seconds'],
            meta_equal=bool(
                geo.bundle.rays == oracle.bundle.rays
                and geo.bundle.cells == oracle.bundle.cells
                and geo.stats['samples'] == oracle.stats['samples']
            ),
        )
        print(
            f"A4 {grid} {label}: within_tol={rows[label]['bundle_within_tol']}/{rows[label]['ground_within_tol']} s={seconds:.2f} "
            f"legacy_read={geo.stats['legacy_cells_read']} files={rows[label]['packed_files']}",
            flush=True,
        )
    # Interrupted build: stop after 3 cells, then resume.
    shutil.rmtree(cache / ('endpoints_' + new_geometry.identity(s, 2)[1]))
    geo = new_geometry.ResidentGeometry(s, scope, cache)
    packed = new_geometry.PackedCache(geo.cache / 'packed')
    calls = [0]

    def flaky(i):
        calls[0] += 1
        if calls[0] == 4:
            raise KeyboardInterrupt('simulated interrupt')
        return geo.cached_cell(i)

    try:
        packed.build(s.geo.ns, flaky)
    except KeyboardInterrupt:
        pass
    # Simulate a torn write: garbage tail beyond the last committed size.
    with open(geo.cache / 'packed' / 'edges.bin', 'ab') as f:
        f.write(b'\x01' * 13)
    geo.prepare()
    rcmp = same_arrays(oracle.bundle.arrays, geo.bundle.arrays)
    rows['resume_after_interrupt'] = dict(bundle_within_tol=all(rcmp.values()), arrays=rcmp)
    print(
        f"A4 {grid} resume_after_interrupt: within_tol={rows['resume_after_interrupt']['bundle_within_tol']}", flush=True
    )
    new_geometry.ResidentGeometry._host_test = False
    return dict(
        grid=grid,
        cells=s.geo.ns,
        rays=oracle.bundle.rays,
        v1_files=v1_files,
        v1_build_seconds=v1_build,
        v1_load_upload_seconds=v1_load,
        v1_cell_marker_found=v1_cell_marker_found,
        cases=rows,
        pass_=bool(
            v1_cell_marker_found
            and all(r['bundle_within_tol'] for r in rows.values())
            and all(r.get('ground_within_tol', True) and r.get('meta_equal', True) for r in rows.values())
            and rows['convert_v1']['legacy_cells_read'] == s.geo.ns
            and rows['fresh_build']['legacy_cells_read'] == 0
        ),
    )


if __name__ == '__main__':
    out = Path(sys.argv[1])
    grids = sys.argv[2].split(',')
    reports = [run(g, out.parent / f'a4_{g}') for g in grids]
    report = dict(grids=reports, pass_=all(r['pass_'] for r in reports))
    out.write_text(json.dumps(report, indent=1) + '\n')
    print('A4 PASS' if report['pass_'] else 'A4 FAIL', flush=True)
    raise SystemExit(0 if report['pass_'] else 1)
