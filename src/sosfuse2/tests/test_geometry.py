"""D1 host serialization/partition checks; not CUDA execution or emulation."""

import numpy as np
from sosfuse2.solve.ray_geometry import create, TransientGeometry, ResidentGeometry, Bundle, subset

try:
    from sosfuse2.checks.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


class ArrayDevice:
    """Exercise the host's exact array transfers without allocating a GPU."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(value):
        return np.asarray(value)


def checks(output):
    s, scope, _ = create(output / 'prepare', 'selftest', [550.0], 'numpy')
    direct = TransientGeometry(s, scope)
    groups = [direct.cell(i) for i in range(s.geo.ns)]
    s.kernel.xp = ArrayDevice()
    s.kernel.sync = lambda: None
    resident = ResidentGeometry(s, scope, output / 'cache')
    resident.prepare()
    a = resident.bundle.arrays
    okay = True
    bad = []
    ray = 0
    for cell, g in enumerate(groups):
        n = len(g['d'])
        lo = int(a['cell_off'][cell])
        hi = int(a['cell_off'][cell + 1])
        cell_ok = (
            lo == ray
            and hi - lo == n
            and within_tol(a['starts'][cell], g['p'][0])
            and within_tol(a['targets'][cell], g['target'][0])
        )
        for name, source in (
            ('dirs', 'd'),
            ('weights', 'weight'),
            ('end_j', 'end_index'),
            ('end_f', 'end_fraction'),
            ('end_col', 'end_columns'),
            ('hits', 'hits_ground'),
        ):
            cell_ok &= within_tol(a[name][lo:hi], g[source].astype(a[name].dtype))
        for local in range(n):
            row = lo + local
            begin = int(a['edge_off'][row])
            end = int(a['edge_off'][row + 1])
            knots = a['edges'][begin:end]
            original = g['edges'][local]
            expected = [original[0]]
            for v in original[1:]:
                if v != expected[-1]:
                    expected.append(v)
            cell_ok &= within_tol(knots, np.asarray(expected))
        if not cell_ok:
            bad.append(cell)
        okay &= cell_ok
        ray += n
    boundary = int(a['cell_off'][len(s.geo.chi)])
    counts = (1, boundary - 1, boundary, boundary + 1, ray)
    for count in counts:
        prefix = resident.prefix(count)
        okay &= (
            prefix.rays == count and int(prefix.arrays['cell_off'][-1]) == count and len(prefix.arrays['dirs']) == count
        )
    # A positive 2e-9 gap must survive: skipping it may not move its neighbor's
    # quadrature endpoint. Exactly repeated knots, in contrast, are disposable.
    tiny = subset(groups[0], slice(0, 1))
    tiny['edges'] = np.array([[0.0, 1.0, 1.0 + 2e-9, 2.0, 2.0]], 'f8')
    tiny['lower'] = np.ones((1, 4), bool)
    packed = Bundle([tiny], np)
    gap_ok = within_tol(packed.arrays['edges'], np.array([0.0, 1.0, 1.0 + 2e-9, 2.0]))
    okay &= gap_ok
    return dict(
        pass_=bool(okay),
        bad_cells=bad,
        cells=len(groups),
        rays=ray,
        compact_bytes=resident.bundle.bytes,
        prefix_counts=list(counts),
        subthreshold_interval_preserved=bool(gap_ok),
        scope='host cache roundtrip/global CSR assembly/prefix boundary checks using array transfer shim; no CUDA claim',
    )


def test_checks(tmp_path):
    assert checks(tmp_path)['pass_']
