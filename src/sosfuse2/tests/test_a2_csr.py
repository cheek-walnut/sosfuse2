"""A2: vectorized CSR packing is within numerical tolerance to the original per-ray loop.

Oracle = the unmodified scripts/sosfuse/geometry.py Bundle (read-only import).
Synthetic 100k rays (duplicate endpoints, 1e-12 positive intervals, rays with
no positive interval, random lower bits, two edge widths) + real cells.
Writes only the JSON log given on the command line.
"""

import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.checks import testkit

_SKY = testkit.sky_sos_root()
if _SKY is None:  # the v1 oracle lives in sky-sos scripts/sosfuse (manifest: sibling-import test)
    import pytest

    pytest.skip(f'v1 oracle scripts/sosfuse needs {testkit.ENV}', allow_module_level=True)
sys.path.append(
    str(testkit.sky_sos('scripts', 'sosfuse').parent)
)  # appended: the installed sosfuse2 package stays first
from sosfuse.geometry import Bundle as OldBundle  # read-only oracle
from sosfuse2.solve.ray_geometry import Bundle, create, TransientGeometry

try:
    from sosfuse2.checks.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


def synthetic(n=100_000, seed=7):
    rng = np.random.default_rng(seed)
    groups = []
    made = 0
    widths = (12, 9)
    while made < n:
        size = int(min(n - made, rng.integers(1, 400)))
        m1 = widths[len(groups) % 3 == 2]
        step = rng.exponential(10.0, (size, m1 - 1))
        kind = rng.random((size, m1 - 1))
        step[kind < 0.25] = 0.0  # duplicate endpoints
        step[(kind >= 0.25) & (kind < 0.35)] = 1e-12  # tiny positive intervals (kept; kernel skips <=1e-8)
        none = rng.random(size) < 0.05
        step[none] = 0.0  # no positive interval at all
        start = rng.uniform(0, 50, (size, 1))
        edges = np.concatenate([start, start + np.cumsum(step, 1)], 1)
        lower = rng.random((size, m1 - 1)) < 0.5
        p = np.broadcast_to(rng.normal(size=3), (size, 3)).copy()
        d = rng.normal(size=(size, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        target = np.broadcast_to(rng.normal(size=(3, 3)), (size, 3, 3)).copy()
        groups.append(
            dict(
                p=p,
                d=d,
                edges=edges,
                lower=lower,
                target=target,
                weight=rng.random(size),
                nq=12,
                end_index=rng.integers(0, 50, size).astype('i4'),
                end_fraction=rng.random(size),
                end_columns=rng.random((size, 5)),
                hits_ground=rng.random(size) < 0.5,
            )
        )
        made += size
    return groups


def compare(groups, label):
    t = time.perf_counter()
    old = OldBundle(groups, np)
    t_old = time.perf_counter() - t
    t = time.perf_counter()
    loop = Bundle(groups, np, loop=True)
    t_loop = time.perf_counter() - t
    t = time.perf_counter()
    new = Bundle(groups, np)
    t_new = time.perf_counter() - t
    same = {}
    for k in old.arrays:
        a, b, c = old.arrays[k], new.arrays[k], loop.arrays[k]
        same[k] = bool(
            a.dtype == b.dtype == c.dtype and a.shape == b.shape == c.shape and (within_tol(a, b) and within_tol(b, c))
        )
    rays = sum(len(g['d']) for g in groups)
    row = dict(
        label=label,
        rays=rays,
        groups=len(groups),
        within_tol=all(same.values()),
        arrays=same,
        seconds_original=t_old,
        seconds_vector=t_new,
        speedup=t_old / max(t_new, 1e-12),
        edges_total=int(len(old.arrays['edges'])),
        rays_without_positive=int((np.diff(old.arrays['edge_off'].astype('i8')) == 1).sum()),
    )
    print(
        f"A2 {label}: rays={rays} identical={row['within_tol']} old={t_old:.3f}s new={t_new:.3f}s x{row['speedup']:.1f}",
        flush=True,
    )
    return row


def overflow_case():
    edges = np.arange(19, dtype='f8')[None, :]
    lower = np.ones((1, 18), bool)
    g = dict(
        p=np.zeros((1, 3)),
        d=np.array([[0, 0, 1.0]]),
        edges=edges,
        lower=lower,
        target=np.eye(3)[None],
        weight=np.ones(1),
        nq=4,
        end_index=np.zeros(1, 'i4'),
        end_fraction=np.zeros(1),
        end_columns=np.zeros((1, 5)),
        hits_ground=np.zeros(1, bool),
    )
    out = {}
    for name, cls in (('original', OldBundle), ('vector', Bundle)):
        try:
            cls([g], np)
            out[name] = 'no error'
        except OverflowError as e:
            out[name] = 'OverflowError'
    # 17 kept intervals whose high bits are zero still fit uint16 in the loop.
    g2 = dict(g, lower=np.r_[np.ones(16, bool), np.zeros(2, bool)][None])
    out['fits_original'] = OldBundle([g2], np).arrays['lower'].tolist()
    out['fits_vector'] = Bundle([g2], np).arrays['lower'].tolist()
    out['pass'] = out['original'] == out['vector'] == 'OverflowError' and out['fits_original'] == out['fits_vector']
    return out


def main(log):
    rows = [compare(synthetic(), 'synthetic_100k')]
    s, scope, _ = create(Path(log).parent / 'a2_selftest_geometry', 'selftest', [550.0], 'numpy', 128)
    geo = TransientGeometry(s, scope)
    cells = [geo.cell(i) for i in range(s.geo.ns)]
    rows.append(compare(cells, 'selftest_grid_all_cells'))
    s, scope, _ = create(Path(log).parent / 'a2_full_geometry', 'full', [550.0], 'numpy', 128)
    geo = TransientGeometry(s, scope)
    pick = np.linspace(0, s.geo.ns - 1, 40).astype(int)
    cells = [geo.cell(int(i)) for i in pick]
    rows.append(compare(cells, 'full_grid_40_cells'))
    # Table path shape: many tiny 4-ray subsets of one pack.
    from sosfuse2.solve.ray_geometry import subset

    tiny = [subset(cells[5], slice(k, min(k + 4, len(cells[5]['d'])))) for k in range(0, len(cells[5]['d']), 4)]
    rows.append(compare(tiny, 'table_like_4ray_groups'))
    report = dict(cases=rows, overflow=overflow_case())
    report['pass'] = all(r['within_tol'] for r in rows) and report['overflow']['pass']
    Path(log).write_text(json.dumps(report, indent=1) + '\n')
    print('A2 PASS' if report['pass'] else 'A2 FAIL', flush=True)
    return report


if __name__ == '__main__':
    main(sys.argv[1])
