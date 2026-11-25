"""Regressions for a faint aerosol source and an open solar-cylinder crossing."""

import numpy as np
import pytest

from sosfuse2.backends import numpy_reference as ref
from sosfuse2.foundation.top import Domain
from sosfuse2.solve.engine_core import basis_at


def single_ray(p, length, heights):
    p = np.array([p], dtype='f8')
    d = np.array([[1., 0., 0.]])
    target = basis_at(p)
    g = dict(p=p, d=d, dloc=np.einsum('nij,ni->nj', target, d), target=target,
             weight=np.ones(1), edges=np.array([[0., length]]), lower=np.array([[True]]), nq=1)
    h = np.array(heights, 'f4')
    c = dict(domain=Domain(), L=16, W=1, u=np.array([.5]), wu=np.ones(1),
             h=h, chi=np.array([0., np.pi], 'f4'), height=np.float32(1.2),
             hr=np.exp(-h / 8), ha=np.exp(-h / np.float32(1.2)),
             mix=np.float32(.97), g=np.float32(.6), albedo=0., ext=np.zeros((5, 1), 'f4'),
             asca=np.ones(1, 'f4'), sun_h=np.array([0., 80.], 'f4'), sun_col=np.zeros((2, 2, 5), 'f4'),
             gm=np.zeros((4, 4, 1), 'f4'), gd=np.zeros((4, 0, 3), 'f4'), gv=np.zeros((4, 0, 1), 'f4'))
    src = np.zeros((4, 163, 1), 'f4')
    return g, c, src


@pytest.mark.parametrize('behind', [True, False])
def test_solar_crossing_before_fp32_rounding(behind):
    pytest.importorskip('mlx.core')
    from sosfuse2.backends import mlx_warm as warm

    # Both crossings round to the node distance in FP32.
    t = 5.0 + (1e-7 if behind else -1e-7)
    g, c, src = single_ray([6371.0 - t, 0., -300.], 20., [0., 80.])
    q = ref.sample(g, 0, 0, c)
    assert bool(ref.sun_columns(q, c)[1][0]) != behind
    ray, y = ref.transport(g, c, src, first=True, return_basis=True)
    expected, _ = ref.project(g, ray, c, y)
    actual, _ = warm.evaluate([g], [c], src, None, first=True)
    assert np.linalg.norm(actual[0] - expected) <= 1e-4 * np.linalg.norm(expected)


def test_aerosol_only_at_high_altitude():
    pytest.importorskip('mlx.core')
    from sosfuse2.backends import mlx_warm as warm

    g, c, src = single_ray([6371. + 45., 0., 0.], 1., [40., 50.])
    src[:, 10] = 1.
    ray, y = ref.transport(g, c, src, return_basis=True)
    expected, _ = ref.project(g, ray, c, y)
    assert np.linalg.norm(expected) > 0.
    actual, _ = warm.evaluate([g], [c], src, None)
    assert np.linalg.norm(actual[0] - expected) <= 1e-4 * np.linalg.norm(expected)
