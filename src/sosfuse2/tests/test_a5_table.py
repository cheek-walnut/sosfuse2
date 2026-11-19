"""A5 checks on NumPy.

1. Compare NumPy tables for groups 4, 32 and 64 at rtol 1e-3.
2. Bundle partition invariance: for group 4/32/64 every per-ray kernel input
   (dirs, weights, edge_off, edges, lower, end_*, hits) is within numerical tolerance;
   only cell_off/starts/targets (block descriptors) differ.  With kernel.cu
   phase 2 having no cross-warp reduction this is the numerical-equivalence argument.
3. Solver.boxes_fixed vs Solver.boxes (selftest grid, reduced source; full grid,
   converged 550 nm source): compare paths at rtol 1e-3 and repeated calls at rtol 1e-5.
   Supply the full source with --full-source or SOSFUSE2_FULL_SOURCE.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.checks import validate
from sosfuse2.outputs import radiance_table as tf
from sosfuse2.solve.ray_geometry import create, TransientGeometry, Bundle, normal_state
from sosfuse2.solve.source_solver import Solver

from sosfuse2.checks.numerical_checks import within_tol


def rel(a, b):
    return float(np.max(abs(a - b) / np.maximum(abs(b), 1e-30)))


def main(out, source_npz, full_source=None):
    out = Path(out)
    work = out.parent / 'a5'
    report = {}
    with np.load(source_npz) as z:
        src = z['source']
    # 1. table, reduced axes (550 nm, 2 solar elevations, 4x6 view grid)
    s, scope, _ = tf.build_solver(normal_state(), work / 'table_prepare', True, 'numpy')
    wl, hs, el, az = tf.axes(True)
    tables = {}
    for group in (4, 32, 64):
        t = time.perf_counter()
        tables[group], _ = tf.make_table(s, src, hs, el, az, group, 25, lambda m: None)
        print(f'A5 table group={group} seconds={time.perf_counter() - t:.2f} shape={tables[group].shape}', flush=True)
    report['table'] = dict(
        shape=list(tables[4].shape),
        within_tol_32=(within_tol(tables[32], tables[4], rtol=1e-3)),
        within_tol_64=(within_tol(tables[64], tables[4], rtol=1e-3)),
        note='NumPy path evaluates per captured pack; group only partitions CUDA blocks',
    )
    # 2. partition invariance of per-ray kernel inputs (captured table packs)
    ee, aa = np.meshgrid(el, az, indexing='ij')
    raws = tf._capture(s, hs[0], ee, aa, np.empty(0, 'f4'))
    bundles = {g: Bundle(tf._groups(raws, g)[0], np) for g in (4, 32, 64)}
    per_ray = ('dirs', 'weights', 'edge_off', 'edges', 'lower', 'end_j', 'end_f', 'end_col', 'hits')
    inv = {k: all((within_tol(bundles[g].arrays[k], bundles[4].arrays[k], rtol=1e-3)) for g in (32, 64)) for k in per_ray}
    blocks = {g: int(bundles[g].cells) for g in bundles}
    report['partition'] = dict(
        per_ray_identical=inv, all_identical=all(inv.values()), blocks=blocks, rays=int(bundles[4].rays)
    )
    # 3. boxes: fixed bundle path vs accepted path
    cases = []
    for grid, field in (('selftest', src), ('full', None)):
        s, scope, _ = create(work / f'boxes_{grid}', grid, [550.0], 'numpy', 128)
        if field is None:
            with np.load(validate.full_source_path(full_source), allow_pickle=False) as z:
                full = z['source']
            field = np.ascontiguousarray(full[..., ::2], 'f4')
        solver = Solver(s, TransientGeometry(s, scope), fixed_boxes=True)
        t = time.perf_counter()
        a = solver.boxes(field)
        ta = time.perf_counter() - t
        t = time.perf_counter()
        b = solver.boxes_fixed(field)
        tb = time.perf_counter() - t
        t = time.perf_counter()
        c = solver.boxes_fixed(field)
        tc = time.perf_counter() - t
        r = dict(
            grid=grid,
            shape=list(a.shape),
            max_relative=rel(b, a),
            within_tol=(within_tol(b, a, rtol=1e-3)),
            repeat_within_tol=(within_tol(c, b, rtol=1e-5)),
            seconds_boxes=ta,
            seconds_fixed_first=tb,
            seconds_fixed_repeat=tc,
        )
        braws = solver._box_rays[0]
        bb = {g: Bundle(tf._groups(braws, g)[0], np) for g in (4, 32, 64)}
        r['partition_per_ray_identical'] = all(
            (within_tol(bb[g].arrays[k], bb[4].arrays[k], rtol=1e-3)) for k in per_ray for g in (32, 64)
        )
        r['partition_blocks'] = {g: int(bb[g].cells) for g in bb}
        r['partition_rays'] = int(bb[4].rays)
        r['pass_'] = bool(r['within_tol'] and r['repeat_within_tol'] and r['partition_per_ray_identical'])
        cases.append(r)
        print(
            f"A5 boxes {grid}: rel={r['max_relative']:.3g} within_tol={r['within_tol']} t={ta:.2f}/{tb:.2f}/{tc:.2f}s",
            flush=True,
        )
    report['boxes'] = cases
    report['pass_'] = bool(
        report['table']['within_tol_32']
        and report['table']['within_tol_64']
        and report['partition']['all_identical']
        and all(c['pass_'] for c in cases)
    )
    out.write_text(json.dumps(report, indent=1) + '\n')
    print('A5 PASS' if report['pass_'] else 'A5 FAIL', flush=True)
    return report['pass_']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('source_npz', type=Path)
    parser.add_argument('--full-source', type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if main(args.out, args.source_npz, args.full_source) else 1)
