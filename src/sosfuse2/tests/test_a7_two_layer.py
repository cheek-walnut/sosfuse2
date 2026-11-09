"""Two-layer geometry cache (A3 implemented) on the full grid, NumPy host path.

For scale heights 1.0/1.2/3.0 km:
 (a) TwoLayerGeometry bundle + ground_bundle arrays (all 12 cached endpoint
     arrays, end_col = composite) within numerical tolerance to the 1-layer ResidentGeometry
     packed path of that height;
 (c) cache-hit state preparation time: common load + aerosol column
     (computed exactly / cached per height), vs the pack() route column.
The common layer is converted from an original v1 cache of the NORMAL state
(1.2 km), placed under its original signature with the identity.json the
original constructor writes (the v1 source folder is MOVED into place, not
copied: it is consumed by the run).  Writes only under the scratch folder argument.
"""

import copy
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2 import geometry as G
from sosfuse2.timers import TIMERS

try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol

F = np.float32


def same(a, b, step=1 << 26):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    x = a.reshape(-1)
    y = b.reshape(-1)
    return all(within_tol(x[i : i + step], y[i : i + step]) for i in range(0, len(x), step))


def od(end_col, ext, rays):
    """kernel.cu boundary attenuation operand: od+=end_col[3*ray+k]*ext[k*FW+w] (FP32, k order)."""
    e = np.asarray(end_col)[rays].astype('f4')
    out = np.zeros((len(rays), ext.shape[1]), 'f4')
    for k in range(3):
        out = out + e[:, k, None] * ext[k][None, :].astype('f4')
    return out


def main(work, v1_source):
    work = Path(work)
    cache = work / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    G.ResidentGeometry._host_test = True
    solvers = {}
    for H in (1.0, 1.2, 3.0):
        st = copy.deepcopy(G.normal_state())
        st['parameters']['scale_height_km'] = H
        solvers[H] = G.create(work / f'prepare_H{H}', 'full', [550.0], 'numpy', 128, st)[:2]
    s12 = solvers[1.2][0]
    content, v1sig = G.identity(s12, 1)
    v1 = cache / ('endpoints_' + v1sig)
    if not v1.exists():
        shutil.move(str(v1_source), str(v1))
        (v1 / 'identity.json').write_text(json.dumps(content, indent=2) + '\n')
    common_sigs = {H: G.identity(solvers[H][0], 3)[1] for H in solvers}
    report = dict(
        v1_signature=v1sig,
        common_signature=common_sigs[1.2],
        common_signature_height_free=len(set(common_sigs.values())) == 1,
        one_layer_signatures={str(H): G.identity(solvers[H][0], 2)[1] for H in solvers},
        cases=[],
    )
    print('A7 v1', v1sig, 'common', common_sigs, flush=True)
    oracle = {}
    for H in (1.0, 1.2, 3.0):
        s, scope = solvers[H]
        TIMERS.reset()
        t = time.perf_counter()
        g1 = G.ResidentGeometry(s, scope, cache)
        g1.prepare()
        oracle[H] = g1
        print(
            f'A7 1-layer H={H} seconds={time.perf_counter() - t:.1f} legacy_read={g1.stats["legacy_cells_read"]}',
            flush=True,
        )
    for H, label in (
        (1.0, 'first: convert v1(1.2) -> common, compute 1.0'),
        (1.2, 'common hit, same-height column'),
        (3.0, 'common hit, compute 3.0'),
        (1.0, 'common hit, cached 1.0 column'),
    ):
        s, scope = solvers[H]
        TIMERS.reset()
        t = time.perf_counter()
        g2 = G.TwoLayerGeometry(s, scope, cache)
        g2.prepare()
        seconds = time.perf_counter() - t
        g1 = oracle[H]
        cmp = {k: same(g1.bundle.arrays[k], g2.bundle.arrays[k]) for k in g1.bundle.arrays}
        gcmp = {k: same(g1.ground_bundle.arrays[k], g2.ground_bundle.arrays[k]) for k in g1.ground_bundle.arrays}
        ec = g2.bundle.arrays['end_col']
        rays = np.random.default_rng(1).choice(len(ec), 20000, replace=False)
        ext = s.ext.astype('f4')
        od2 = od(ec, ext, rays)
        od1 = od(g1.bundle.arrays['end_col'], ext, rays)
        common_col = np.asarray(G.PackedCache(g2.cache / 'packed').open()[0]['end_col'])
        odc = od(common_col, ext, rays)
        row = dict(
            height=H,
            label=label,
            seconds=seconds,
            arrays_identical=all(cmp.values()),
            ground_identical=all(gcmp.values()),
            arrays=cmp,
            end_col_contiguous=bool(np.asarray(ec).flags.c_contiguous),
            end_col_dtype=str(np.asarray(ec).dtype),
            boundary_od_identical=bool((within_tol(od2, od1))),
            boundary_od_if_common_column_unreplaced_max_rel=float(np.max(abs(odc - od1) / np.maximum(abs(od1), 1e-30))),
            stats={
                k: g2.stats.get(k)
                for k in (
                    'aerosol_column',
                    'aerosol_column_seconds',
                    'load_seconds',
                    'build_seconds',
                    'legacy_cells_read',
                    'common_source_height',
                    'signature',
                )
            },
            timers=TIMERS.report()['wall_seconds'],
        )
        report['cases'].append(row)
        print(
            f"A7 2-layer H={H} [{label}] identical={row['arrays_identical']}/{row['ground_identical']} od={row['boundary_od_identical']} "
            f"s={seconds:.2f} column={row['stats']['aerosol_column']} {row['stats']['aerosol_column_seconds']:.2f}s "
            f"legacy_read={row['stats']['legacy_cells_read']}",
            flush=True,
        )
    # pack() route for the column, full grid, as the 91% reference
    s, scope = solvers[3.0]
    host, _ = G.PackedCache(G.TwoLayerGeometry(s, scope, cache).cache / 'packed').open()
    t = time.perf_counter()
    c_pack = G.aerosol_column_pack_route(host, scope.b.sh.old)
    t_pack = time.perf_counter() - t
    t = time.perf_counter()
    c1 = G.aerosol_column(host, 3.0, workers=1)
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    c4 = G.aerosol_column(host, 3.0, workers=4)
    t4 = time.perf_counter() - t
    report['column_timing_full_grid_H3'] = dict(
        rays=len(c4),
        pack_route_seconds=t_pack,
        exact_1thread_seconds=t1,
        exact_4threads_seconds=t4,
        identical=bool(
            (
                within_tol(c_pack, c1)
                and within_tol(c1, c4)
                and within_tol(c4, np.ascontiguousarray(np.asarray(oracle[3.0].bundle.arrays['end_col'])[:, 1]))
            )
        ),
    )
    print('A7 column timing', report['column_timing_full_grid_H3'], flush=True)
    report['pass_'] = bool(
        report['common_signature_height_free']
        and all(c['arrays_identical'] and c['ground_identical'] and c['boundary_od_identical'] for c in report['cases'])
        and report['column_timing_full_grid_H3']['identical']
        and report['cases'][0]['stats']['legacy_cells_read'] == 1500
    )
    G.ResidentGeometry._host_test = False
    (work / 'a7_two_layer.json').write_text(json.dumps(report, indent=1) + '\n')
    print('A7 PASS' if report['pass_'] else 'A7 FAIL', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
