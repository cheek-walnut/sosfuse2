"""D1 known-fixed-point and early-converged wavelength regression checks."""

from types import SimpleNamespace
import numpy as np
from sosfuse2.fused import Solver


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
