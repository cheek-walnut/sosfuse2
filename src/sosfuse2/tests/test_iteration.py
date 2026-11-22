"""D1 known-fixed-point and early-converged wavelength regression checks."""

from types import SimpleNamespace
import numpy as np
import pytest
from sosfuse2.solve.source_solver import Solver


class DummyKernel:
    xp = np

    @staticmethod
    def host(value):
        return value

    @staticmethod
    def sync():
        pass

    @staticmethod
    def memory():
        pass


def checks():
    # A diagonal affine operator has an independently known exact fixed point.
    # Wave zero settles at the first apply while other waves need more cycles.
    waves = 15
    ns = 2
    contraction = np.linspace(0.25, 0.75, waves, dtype='f4')[None, None, :]
    contraction[:, :, 0] = 0
    forcing = np.linspace(0.01, 0.2, ns * 571 * waves, dtype='f4').reshape(ns, 571, waves)
    expected = forcing / (1 - contraction)
    solver = Solver(SimpleNamespace(W=waves, geo=SimpleNamespace(ns=ns), kernel=DummyKernel()), None, prepared=True)
    solver.apply = lambda field: forcing + contraction * field
    solver.boxes = lambda field: np.broadcast_to(field[0, 0], (5, 7, waves)).copy()
    field, stats = solver.solve(max_eval=16, tolerance=0.005)
    relative = float(np.max(abs(field - expected) / expected))
    # The tolerance applies to the residual, not the true error. For q <= .75,
    # e(F(x)) = q/(1-q) * residual; the conservative bound is 3 * .005.
    error_bound = float(contraction.max() / (1 - contraction.max()) * 0.005)
    first_done = next(i for i, row in enumerate(stats['history']) if row['converged'][0])
    frozen = all(
        row['source_relative'][0] == 0 and row['observer_relative'][0] == 0
        for row in stats['history'][first_done + 1 :]
    )
    try:
        unconverged = Solver(SimpleNamespace(W=waves, geo=SimpleNamespace(ns=ns), kernel=DummyKernel()), None, prepared=True)
        unconverged.apply = lambda field: field + forcing
        unconverged.boxes = solver.boxes
        unconverged.solve(max_eval=2, tolerance=0.005)
        failure_reported = False
    except RuntimeError as exc:
        failure_reported = str(exc).startswith('E_NOT_CONVERGED:')
    okay = relative <= error_bound + 1e-6 and frozen and first_done < len(stats['history']) - 1 and failure_reported
    return dict(
        pass_=bool(okay),
        known_fixed_point_max_relative=relative,
        known_residual_to_error_bound=error_bound,
        early_wave_frozen=bool(frozen),
        evaluations=stats['evaluations'],
        nonconvergence_reported=bool(failure_reported),
        scope='known diagonal affine fixed point, mixed wavelength convergence, nonconvergence signal; no physical-grid or CUDA convergence claim',
    )


def test_checks():
    assert checks()['pass_']


def test_host_history_matches_device_with_mixed_convergence(monkeypatch):
    from sosfuse2.foundation.top import Domain

    rng = np.random.default_rng(7)
    shape = (2, 660, 3)
    forcing = rng.uniform(0.1, 1, shape).astype('f4')
    contraction = rng.uniform(0.1, 0.75, shape).astype('f4')
    contraction[:, :, 0] = 0

    class HostVector(np.ndarray):
        # Storage must pass through the backend array conversion before mixing.
        __array_ufunc__ = None

    class Kernel(DummyKernel):
        L = 24
        domain = Domain(200)

        def __init__(self):
            self.saved = []

        def host(self, value):
            if value.size >= shape[0] * shape[1]:
                self.saved.append((value.shape, value.dtype))
                return value.copy().view(HostVector)
            return value

    def run(storage):
        monkeypatch.setenv('SOSFUSE2_ANDERSON_HISTORY', storage)
        kernel = Kernel()
        state = SimpleNamespace(W=shape[2], geo=SimpleNamespace(ns=shape[0]), kernel=kernel, strato={})
        solver = Solver(state, None, prepared=True)
        solver.apply = lambda x: forcing + contraction * x
        solver.observer = lambda x: np.broadcast_to(x[0, 0], (7, 7, shape[2])).copy()
        field, stats = solver.solve(max_eval=24, tolerance=1e-5)
        assert solver.timings['anderson_history'] == storage
        return field, stats['history'], kernel.saved

    reference, _, device_saved = run('device')
    field, history, host_saved = run('host')
    np.testing.assert_allclose(field, reference, rtol=1e-3, atol=0)
    np.testing.assert_allclose(field, forcing / (1 - contraction), rtol=1e-3, atol=0)
    assert history[1]['converged'] == [True, False, False]
    assert len(history) > 5
    assert all(row['source_relative'][0] == 0 for row in history[2:])
    active_updates = sum(row['converged'].count(False) for row in history)
    expected_pair = [(shape[:2], np.dtype('f4')), ((shape[0] * shape[1] + 42,), np.dtype('f8'))]
    assert not device_saved
    assert host_saved == expected_pair * active_updates


def test_invalid_anderson_storage(monkeypatch):
    monkeypatch.setenv('SOSFUSE2_ANDERSON_HISTORY', 'unknown')
    solver = Solver(SimpleNamespace(kernel=DummyKernel()), None, prepared=True)
    with pytest.raises(ValueError, match='E_ANDERSON_HISTORY'):
        solver.solve()


def test_ground_sh_order_reads_this_state_only():
    s = SimpleNamespace(W=1, geo=SimpleNamespace(ns=1))
    other = SimpleNamespace()
    kernel = SimpleNamespace(xp=np, ground_sh_l=24, device_constants={})
    s.kernel = kernel
    solver = Solver(s, None, prepared=True)
    assert solver.ground_sh_order() == 0
    kernel.device_constants[(id(s), 12)] = {'h': None}
    kernel.device_constants[(id(other), 12)] = {'ground_sh': None}
    assert solver.ground_sh_order() == 0
    kernel.device_constants[(id(s), 24)] = {'ground_sh': None}
    assert solver.ground_sh_order() == 24
    plain = SimpleNamespace(W=1, geo=SimpleNamespace(ns=1), kernel=DummyKernel())
    assert Solver(plain, None, prepared=True).ground_sh_order() == 0


def test_numpy_solve_records_no_ground_table(monkeypatch):
    from sosfuse2.backends.sh_layout import source_size
    from sosfuse2.solve.source_solver import Kernel

    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '24')
    kernel = Kernel(backend='numpy', L=16)
    assert kernel.ground_sh_l == 24
    s = SimpleNamespace(W=1, geo=SimpleNamespace(ns=1), kernel=kernel)
    solver = Solver(s, None, prepared=True)
    forcing = np.full((1, source_size(s), 1), 0.1, 'f4')
    solver.apply = lambda field: forcing + 0.5 * field
    solver.boxes = lambda field: np.broadcast_to(field[0, 0], (5, 7, 1)).copy()
    solver.solve(max_eval=16, tolerance=0.005)
    assert solver.timings['ground_sh_order'] == 0
