"""Compare a warm source application on the fixed reduced diagnostic grid."""

from time import perf_counter

import numpy as np
import pytest

from sosfuse2.atmosphere.density import LEGACY
from sosfuse2.backends import numpy_reference as ref
from sosfuse2.checks import gate, validate
from sosfuse2.solve.ray_geometry import create, normal_state, TransientGeometry
from sosfuse2.solve.source_solver import Solver


def relative_error(actual, reference):
    actual, reference = np.asarray(actual, 'f8'), np.asarray(reference, 'f8')
    error = np.abs(actual - reference)
    return np.divide(error, abs(reference), out=np.where(error == 0, 0., np.inf), where=reference != 0)


@pytest.fixture(scope='module')
def reference_case(tmp_path_factory):
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    state = dict(normal_state(), density=dict(LEGACY))
    s, scope, _ = create(tmp_path_factory.mktemp("geometry"), 'selftest', [550.0], 'numpy', 128, state=state)
    geo = TransientGeometry(s, scope)
    solver = Solver(s, geo)
    seed = solver.apply(np.zeros((s.geo.ns, 571, 1), 'f4'))
    rays = []
    transport = ref.transport

    def observe(*args, **kwargs):
        result = transport(*args, **kwargs)
        rays.append(result[0])
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ref, 'transport', observe)
        start = perf_counter()
        expected = solver.apply(seed)
        elapsed = perf_counter() - start
    directions = validate.legacy_selftest()
    full = np.zeros((*expected.shape[:2], 2), 'f4')
    full[..., 0] = expected[..., 0]
    query = (directions['hs'], directions['el'], directions['az'])
    direct = s.query(*query, full, True)[..., 0]
    expected_rad = direct + s.query(*query, full, False)[..., 0]
    return mx, solver, seed, expected, np.concatenate(rays), elapsed, query, direct, expected_rad


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_warm_apply(reference_case, device, monkeypatch):
    mx, solver, seed, expected, expected_rays, seconds_numpy, query, direct, expected_rad = reference_case
    if device == 'gpu' and not mx.metal.is_available():
        pytest.skip('Metal device unavailable')
    from sosfuse2.backends.mlx_transport import WarmTransport

    s, geo = solver.s, solver.geometry
    cache, rays = {}, []

    def transport(g, c, field, first=False, boundary=None, statistics=None, return_basis=False):
        assert not first
        op = cache.get(id(g))
        if op is None:
            # Observer rays are packed on demand after the timed source apply.
            op = WarmTransport(g, c)
        b = None if boundary is None else mx.array(boundary.astype('f4'))
        result = op(mx.array(field), b)
        mx.eval(result)
        ray = np.asarray(result)
        if return_basis:
            rays.append(ray)
            return ray, op.basis
        return ray

    with mx.stream(getattr(mx, device)):
        mx.reset_peak_memory()
        start = perf_counter()
        for g in geo.groups.values():
            cache[id(g)] = WarmTransport(g, s.kernel.setup(s, g['nq']))
        seconds_prepare = perf_counter() - start
        monkeypatch.setattr(ref, 'transport', transport)
        start = perf_counter()
        actual = solver.apply(seed)
        seconds_apply = perf_counter() - start
        peak_mlx = mx.get_peak_memory()
        full = np.zeros((*actual.shape[:2], 2), 'f4')
        full[..., 0] = actual[..., 0]
        actual_rad = direct + s.query(*query, full, False)[..., 0]
    actual_rays = np.concatenate(rays)
    # Measure both the signed tensor and its radiometric trace before projection.
    actual_trace = actual_rays[..., :3].sum(-1)
    expected_trace = expected_rays[..., :3].sum(-1)
    source_l2 = gate.l2_by_wavelength(actual, expected).max()
    source_error = relative_error(actual, expected)
    worst = np.unravel_index(np.argmax(source_error), actual.shape)
    tensor_error = relative_error(actual_rays, expected_rays)
    ray_worst = np.unravel_index(np.argmax(tensor_error), actual_rays.shape)
    print(f'{device=} shape={actual.shape} rays={actual_trace.size} observer={expected_rad.size} '
          f'tensor_rel={tensor_error.max():.7g} tensor_worst={tuple(map(int, ray_worst))} '
          f'tensor_actual={actual_rays[ray_worst]:.8g} tensor_ref={expected_rays[ray_worst]:.8g} '
          f'trace_rel={relative_error(actual_trace, expected_trace).max():.7g} '
          f'source_rel={source_error.max():.7g} source_exceed={(source_error > 1e-3).sum()} '
          f'source_worst={tuple(map(int, worst))} actual={actual[worst]:.9g} ref={expected[worst]:.9g} '
          f'{source_l2=:.7g} observer_rel={relative_error(actual_rad, expected_rad).max():.7g} '
          f'observer_values={actual_rad.ravel().tolist()} '
          f'{peak_mlx=} {seconds_numpy=:.4f} {seconds_prepare=:.4f} {seconds_apply=:.4f}')
    assert np.isfinite(actual_rays).all() and np.isfinite(actual).all()
    np.testing.assert_allclose(actual_rays, expected_rays, rtol=1e-3, atol=0, equal_nan=False)
    np.testing.assert_allclose(actual_trace, expected_trace, rtol=1e-3, atol=0, equal_nan=False)
    np.testing.assert_allclose(actual_rad, expected_rad, rtol=1e-3, atol=0, equal_nan=False)
    assert source_l2 <= 1e-3
