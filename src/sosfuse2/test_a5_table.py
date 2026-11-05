"""A5 checks on NumPy.

1. table_fast.make_table with group 4/32/64: table bytes (NumPy has no block
   partition, so this only proves the refactor (shared Bundle, timers) kept the
   accepted values; CUDA group timing is a rig TODO).
2. Bundle partition invariance: for group 4/32/64 every per-ray kernel input
   (dirs, weights, edge_off, edges, lower, end_*, hits) is within numerical tolerance;
   only cell_off/starts/targets (block descriptors) differ.  With kernel.cu
   phase 2 having no cross-warp reduction this is the numerical-equivalence argument.
3. Solver.boxes_fixed vs Solver.boxes (selftest grid, reduced source; full grid,
   accepted converged b2 550 nm source): max relative difference (gate 1e-12).
"""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2 import validate
from sosfuse2 import table_fast as tf
from sosfuse2.geometry import create, TransientGeometry, Bundle, normal_state
from sosfuse2.fused import Solver
try:
    from .numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


def rel(a, b):
    return float(np.max(abs(a-b)/np.maximum(abs(b), 1e-30)))


def main(out, source_npz):
    out = Path(out); work = out.parent/'a5'; report = {}
    with np.load(source_npz) as z: src = z['source']
    # 1. table, reduced axes (550 nm, 2 solar elevations, 4x6 view grid)
    s, scope, _ = tf.build_solver(normal_state(), work/'table_prepare', True, 'numpy')
    wl, hs, el, az = tf.axes(True); tables = {}
    for group in (4, 32, 64):
        t = time.perf_counter(); tables[group], _ = tf.make_table(s, src, hs, el, az, group, 25, lambda m: None)
        print(f'A5 table group={group} seconds={time.perf_counter()-t:.2f} shape={tables[group].shape}', flush=True)
    report['table'] = dict(shape=list(tables[4].shape), within_tol_32=(within_tol(tables[4], tables[32])),
                           within_tol_64=(within_tol(tables[4], tables[64])),
                           note='NumPy path evaluates per captured pack; group only partitions CUDA blocks')
    # 2. partition invariance of per-ray kernel inputs (captured table packs)
    ee, aa = np.meshgrid(el, az, indexing='ij'); raws = tf._capture(s, hs[0], ee, aa, np.empty(0, 'f4'))
    bundles = {g: Bundle(tf._groups(raws, g)[0], np) for g in (4, 32, 64)}
    per_ray = ('dirs', 'weights', 'edge_off', 'edges', 'lower', 'end_j', 'end_f', 'end_col', 'hits')
    inv = {k: all((within_tol(bundles[4].arrays[k], bundles[g].arrays[k])) for g in (32, 64)) for k in per_ray}
    blocks = {g: int(bundles[g].cells) for g in bundles}
    report['partition'] = dict(per_ray_identical=inv, all_identical=all(inv.values()), blocks=blocks, rays=int(bundles[4].rays))
    # 3. boxes: fixed bundle path vs accepted path
    cases = []
    for grid, field in (('selftest', src), ('full', None)):
        s, scope, _ = create(work/f'boxes_{grid}', grid, [550.], 'numpy', 128)
        if field is None:
            if validate.FULL_SOURCE is None:   # the accepted full-grid source record lives in sky-sos results/
                raise RuntimeError('E_SKY_SOS_ROOT: the full-grid case reads the sky-sos sos_full/b2_run source record (set SOSFUSE2_SKY_SOS_ROOT)')
            with np.load(validate.FULL_SOURCE, allow_pickle=False) as z:
                full = z['source']
            field = np.ascontiguousarray(full[..., ::2], 'f4')
        solver = Solver(s, TransientGeometry(s, scope), fixed_boxes=True)
        t = time.perf_counter(); a = solver.boxes(field); ta = time.perf_counter()-t
        t = time.perf_counter(); b = solver.boxes_fixed(field); tb = time.perf_counter()-t
        t = time.perf_counter(); c = solver.boxes_fixed(field); tc = time.perf_counter()-t
        r = dict(grid=grid, shape=list(a.shape), max_relative=rel(b, a), within_tol=(within_tol(a, b)),
                 repeat_within_tol=(within_tol(b, c)), seconds_boxes=ta, seconds_fixed_first=tb, seconds_fixed_repeat=tc)
        braws = solver._box_rays[0]
        bb = {g: Bundle(tf._groups(braws, g)[0], np) for g in (4, 32, 64)}
        r['partition_per_ray_identical'] = all((within_tol(bb[4].arrays[k], bb[g].arrays[k])) for k in per_ray for g in (32, 64))
        r['partition_blocks'] = {g: int(bb[g].cells) for g in bb}; r['partition_rays'] = int(bb[4].rays)
        r['pass_'] = bool(r['max_relative'] <= 1e-12 and r['repeat_within_tol'] and r['partition_per_ray_identical']); cases.append(r)
        print(f"A5 boxes {grid}: rel={r['max_relative']:.3g} bit={r['within_tol']} t={ta:.2f}/{tb:.2f}/{tc:.2f}s", flush=True)
    report['boxes'] = cases
    report['pass_'] = bool(report['table']['within_tol_32'] and report['table']['within_tol_64'] and
                           report['partition']['all_identical'] and all(c['pass_'] for c in cases))
    out.write_text(json.dumps(report, indent=1)+'\n'); print('A5 PASS' if report['pass_'] else 'A5 FAIL', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
