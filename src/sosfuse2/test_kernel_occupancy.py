#!/usr/bin/env python3
"""Untiled occupancy checks and bounded selftest CUDA-event benchmark.

No full-grid mode. GPU exceptions propagate: do not retry them. Occupancy is
CUDA's resource-based resident-block limit, not measured achieved occupancy.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

# Keep the documented direct-script and `python -m` entry points working (the bare import below needs this folder on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent))
# New portable entry point; legacy GPU/reference entry points below are retained.
if '--host-only' in sys.argv:
    import argparse
    from test_strato_parallel_host import run as run_strato_host
    parser=argparse.ArgumentParser()
    parser.add_argument('--host-only',action='store_true')
    parser.add_argument('--out',type=Path,default=Path('strato_host_report.json'))
    option=parser.parse_args()
    raise SystemExit(0 if run_strato_host(option.out,occupancy_only=True)['pass_'] else 1)

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2 import paths
from sosfuse2.fused import Kernel
try:
    from .numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol

SELECTED = ('LANE_GROUND', 'LM_LUT', 'PHASE2_NOBARRIER', 'SH_AUX', 'SPECIALIZE')


def thermal(read=None, sleep=time.sleep):
    """Every launch requires <75 C; >=85 or missing GPU fails closed."""
    if read is None:
        def read():
            text = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm',
                                            '--format=csv,noheader'], text=True).strip()
            return text
    while True:
        row = read()
        temperature = int(row.split(',')[0])
        if temperature >= 85:
            raise RuntimeError('E_GPU_THERMAL_STOP: '+row)
        if temperature < 75:
            return row
        sleep(5)


def host_tests():
    readings = iter(['75, 0 W, 0 MHz', '80, 0 W, 0 MHz', '74, 0 W, 0 MHz'])
    waits = []
    assert thermal(lambda: next(readings), waits.append).startswith('74,')
    assert waits == [5, 5]
    try:
        thermal(lambda: '85, 0 W, 0 MHz', waits.append)
    except RuntimeError as e:
        assert 'E_GPU_THERMAL_STOP' in str(e)
    else:
        raise AssertionError('thermal stop ignored')
    print('HOST PASS', flush=True)


def benchmark(out, repeats=3, all_cells=False, seed=None):
    if repeats < 3:
        raise ValueError('E_REPEATS: at least three CUDA-event samples')
    thermal()
    import cupy as cp
    from sosfuse2.geometry import create, TransientGeometry, Bundle
    # Three real selftest destination cells. Baseline official parity already
    # exercises one of these; the fixture is intentionally short, never full.
    s, scope, _ = create(out/'geometry', 'selftest', (360+(np.arange(15)+.5)*470/15).tolist(), 'cupy', 256)
    geometry = TransientGeometry(s, scope)
    cell_ids = list(range(s.geo.ns)) if all_cells else [0, 12, 25]
    groups = [geometry.cell(c) for c in cell_ids]
    bundle = Bundle(groups, cp)
    seed_path = seed if seed is not None else out.parent/'before/selftest_values.npz'
    with np.load(seed_path) as z:
        field = np.tile(z['source'], (1, 1, 15))*np.linspace(.5, 1.5, 15, dtype='f4')
    field = cp.asarray(field)
    boundary = cp.asarray(np.linspace(.0001, .001, len(s.geo.chi)*15).reshape(-1, 15))
    # Small real table_fast capture: phase 2 exercises both first flags.
    from functools import partial
    from sosfuse2.table_fast import _capture, _bundle, _evaluate
    ee, aa = np.meshgrid([20., 60.], [0., 90.], indexing='ij')
    raws = _capture(s, -6., ee, aa, np.empty(0, 'f4'))
    query_bundle = _bundle(s, raws, 32)
    records = []
    baseline = {}
    for tile in (None,):
        s.kernel = Kernel(backend='cupy', block_size=256, flags=SELECTED+(() if tile is None else (tile,)))
        for name, first, phase, bd in (('first', True, 0, None), ('ground', False, 1, None), ('warm', False, 0, boundary),
                                       ('query_first', True, 2, None), ('query', False, 2, None)):
            active_bundle = query_bundle if phase == 2 else bundle
            telemetry = []
            times = []
            launch_seconds = []
            for i in range(repeats+1):
                telemetry.append(thermal())
                if phase == 2:
                    s.kernel.launch = partial(s.kernel.launch, timed=True)
                    try:
                        query_arrays = _evaluate(s, raws, field, first, 32, query_bundle)
                    finally:
                        del s.kernel.launch
                else:
                    outputs = s.kernel.launch(bundle, s, field, first=first, phase=phase, boundary=bd, timed=True)
                seconds = s.kernel.cuda_events[-1]
                launch_seconds.append(seconds)
                if seconds > 5:
                    raise RuntimeError(f'E_LAUNCH_BUDGET: {seconds}; stop further GPU work')
                # Post-launch observation catches a thermal stop before another launch.
                telemetry.append(thermal())
                if i: times.append(seconds)
            arrays = query_arrays if phase == 2 else [s.kernel.host(outputs[j]) for j in ((0, 1) if phase == 0 else (1,))]
            if tile is None:
                baseline[name] = arrays
            errors = [float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300)) for a, b in zip(arrays, baseline[name])]
            assert all(np.isfinite(a).all() for a in arrays)
            assert max(errors) <= 1e-6, errors
            raw = s.kernel.raw_kernel(15, first, phase, bd is not None)
            shared = s.kernel.shared_bytes(15, phase, first=first)
            active = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(raw.kernel.ptr, 256, shared)
            props = cp.cuda.runtime.getDeviceProperties(0)
            record = dict(tile=tile or 'off', kernel=name, seconds=times, median_seconds=statistics.median(times),
                          launch_seconds=launch_seconds, max_l2=max(errors), within_tol=all((within_tol(a, b)) for a,b in zip(arrays,baseline[name])),
                          dynamic_shared_bytes=shared, attributes=raw.attributes, resident_blocks_per_sm=active,
                          resource_occupancy=active*256/props['maxThreadsPerMultiProcessor'], telemetry=telemetry,
                          cells=active_bundle.cells, rays=active_bundle.rays, grid='selftest',
                          tile_width=s.kernel.tile_width(15, first, phase), first=first, phase=phase, scope='selftest cells / table_fast 2x2 angles at sun -6; device event excludes SH_AUX build, capture, bundle and D2H')
            records.append(record)
            (out/'benchmark.json').write_text(json.dumps(records, indent=2)+'\n')
            print(json.dumps(record), flush=True)
    # Ray counts and segment quadrature work per warp assignment are immutable
    # geometry proxies, not clock64 measurements of actual warp elapsed cycles.
    hist=[]
    for cell, group in zip(cell_ids,groups):
        valid=(np.diff(group['edges'],axis=1)>1e-8).sum(axis=1)*group['nq']
        work=[int(valid[w::8].sum()) for w in range(8)]
        hist.append(dict(cell=cell, rays=len(valid), ray_counts=[len(valid[w::8]) for w in range(8)],
                         path_samples=work, max_to_mean=max(work)/np.mean(work)))
    (out/'warp_work.json').write_text(json.dumps(hist,indent=2)+'\n')
    return records


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--all-cells', action='store_true', help='all cells of the selftest grid, never full grid')
    p.add_argument('--out', type=Path, default=paths.work_root()/'occupancy')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=Path, help='selftest_values.npz from the official selftest')
    args=p.parse_args()
    host_tests()
    if args.gpu:
        args.out.mkdir(parents=True,exist_ok=True)
        os.environ['CUPY_CACHE_DIR']=str(args.out/'nvrtc_cache')
        benchmark(args.out,args.repeats,args.all_cells,args.seed)
