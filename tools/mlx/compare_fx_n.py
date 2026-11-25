"""Time one fixed FX-N worker and compare its arrays to the archived NumPy run."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import pickle
from functools import lru_cache
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sosfuse2.backends import numpy_reference as ref
from sosfuse2.checks import gate
from sosfuse2.jobs import run_state


FX_N_SHA256 = "f95f1cea89345f607b541ef37d4e5b1b8a4c832b9e14144c04d493c9353c94aa"


@lru_cache(maxsize=1)
def read_cells(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def project_cells(batch):
    path, indices, field, first, boundary, ground = batch
    cells, constants = read_cells(path)
    rows = []
    for idx in indices:
        geometry = cells[idx]
        c = constants[geometry["nq"]]
        rays, basis = ref.transport(geometry, c, field, first, boundary, return_basis=True)
        moments, flux = ref.project(geometry, rays, c, basis, ground_only=ground)
        rows.append((idx, moments, flux))
    return rows


def parallel_evaluation(pool, workers, scratch):
    groups = {}
    def evaluate(geometry, s, field, first=False, phase=0, ground_only=False, boundary=None, timed=False):
        count = len(s.geo.chi) if ground_only else s.geo.ns
        if geometry not in groups:
            cells = [geometry.cell(i) for i in range(s.geo.ns)]
            constants = {}
            for n in {g["nq"] for g in cells}:
                c = s.kernel.setup(s, n)
                optics = c["density_optics"]
                compact = SimpleNamespace(geometry_segments=optics["old"].geometry_segments)
                constants[n] = dict(c, density_optics=dict(optics, old=compact))
            path = Path(scratch) / str(len(groups))
            with path.open("xb") as stream:
                pickle.dump((cells, constants), stream, protocol=pickle.HIGHEST_PROTOCOL)
            groups[geometry] = path
        moments = np.empty((count, 4 + s.kernel.K, s.W), dtype="f8")
        flux = np.empty((count, s.W), dtype="f8")
        batches = [(groups[geometry], range(i, count, workers), field, first, boundary, phase == 1)
                   for i in range(min(workers, count))]
        for rows in pool.map(project_cells, batches):
            for i, m, f in rows:
                moments[i], flux[i] = m, f
        return moments, flux
    return evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["numpy", "mlx"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, help="NumPy cell processes (default 9)")
    args = parser.parse_args()
    if args.workers is None:
        args.workers = 9 if args.backend == "numpy" else 1
    if args.workers < 1 or (args.backend == "mlx" and args.workers != 1):
        parser.error("--workers selects positive NumPy cell-worker counts")
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = "FX-N"
    spec = gate.load_fixtures()
    environment = gate.fixture_env(spec, fixture)
    baseline = gate.manifest()["golden"][fixture]
    if baseline["sha256"] != FX_N_SHA256:
        raise ValueError("FX-N manifest differs from the fixed comparison reference")
    expected = gate.baseline_npz(baseline["file"], FX_N_SHA256)
    cycle_seconds = []
    create = run_state.create_configured

    def timed_create(*args, **kwargs):
        state, scope = create(*args, **kwargs)
        query = state.query
        end_times = []
        calls = 0

        def timed_query(*args, **kwargs):
            nonlocal calls
            result = query(*args, **kwargs)
            calls += 1
            # Each iteration ends after single + multiple at every solar elevation.
            if calls % (2 * len(spec["observer_solar_deg"])) == 0:
                end_times.append(perf_counter())
                if len(end_times) > 1:
                    cycle_seconds.append(end_times[-1] - end_times[-2])
            return result

        state.query = timed_query
        return state, scope

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, environment, clear=True))
        stack.enter_context(patch.object(run_state, "create_configured", timed_create))
        if args.backend == "mlx":
            import mlx.core as mx
            from sosfuse2.backends.mlx_transport import FixtureTransport

            stack.enter_context(mx.stream(mx.gpu))
            transport = FixtureTransport()
            stack.enter_context(patch.object(ref, "transport", transport))
        start = perf_counter()
        if args.workers > 1:
            from sosfuse2.solve.ray_geometry import TransientGeometry
            scratch = stack.enter_context(TemporaryDirectory())
            pool = stack.enter_context(ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")))
            stack.enter_context(patch.object(TransientGeometry, "evaluate", parallel_evaluation(pool, args.workers, scratch)))
        gate.worker("fixed", fixture, args.output, "numpy")
        seconds = perf_counter() - start
    if len(cycle_seconds) != 9:
        raise RuntimeError(f"Expected nine warm FX-N cycles, measured {len(cycle_seconds)}")
    with np.load(args.output / "fixed.npz") as candidate:
        radiance = candidate["radiance"]
        source = candidate["source"][-1]
    if radiance.shape != expected["radiance"].shape or source.shape != expected["source"].shape:
        raise ValueError("fixed fixture output shapes differ from the archived NumPy reference")
    rad = gate.tolerance(radiance, expected["radiance"], 0.)
    source_l2 = gate.l2_by_wavelength(source, expected["source"])
    report = dict(backend=args.backend, fixture=fixture, cell_workers=args.workers, reference_sha256=baseline["sha256"],
                  worker_seconds=seconds, warm_cycle_seconds=cycle_seconds, rtol=gate.RTOL, atol=0.,
                  radiance_max_rel=rad["max_rel"], radiance_elements=radiance.size,
                  source_l2_by_wavelength=source_l2.tolist(),
                  passed=bool(rad["pass_"] and np.all(source_l2 <= gate.RTOL)))
    if args.backend == "mlx":
        report["transport_calls"] = transport.calls
        report["prepared_ray_groups"] = transport.preparations
        report["metal_peak_bytes"] = mx.get_peak_memory()
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
