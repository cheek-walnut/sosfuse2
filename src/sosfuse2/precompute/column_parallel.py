"""Opt-in process parallelism for the cold column builds (source_solver.Kernel.strato_end_columns / density_end_columns
and the host-cache endpoint02 build, host_cache.endpoint_columns).

Numerical equivalence with the serial loop is checked at rtol 1e-6: the same 2048-ray chunks (solve/source_solver.py build loops), the same
per-chunk call (stratosphere.columns / density.correct_columns) on equal input bytes, and each chunk's result is
placed at its own index; no sum crosses a chunk.  Workers are spawned processes (no GIL; never a fork of a parent
that may hold CUDA or threads).  Module functions and minimal state bindings travel by pickle.
Workers check the source hashes recorded when the parent imports this module before running any task.
Workers are used only for SOSFUSE2_COLUMN_METHOD=gl24; the default table-v1 path uses column_table.fill.

The process count is SOSFUSE2_COLUMN_WORKERS; unset = all usable CPUs (default_workers), 1 = the serial loop.
pipeline.py exports max(--threads, usable CPUs // concurrent states) unless the variable is set (pipeline.prep_workers).  The worker count enters no cache key: column_cache.cache_key hashes stratosphere.py, density.py, the
accepted sources and the engine.  test_column_parallel.py checks shape, dtype and numerical tolerance.
"""

import hashlib
import json
import os
import pickle
import time
import types
from pathlib import Path

import numpy as np

# Moved to host_limits (stdlib only, so pipeline.py can use it); re-exported for sun/ground/ray_geometry_parallel.
from ..foundation.host_limits import _quota_limits, default_workers  # noqa: F401
from ..foundation import paths

CHUNK = 2048  # solve/source_solver.py strato_end_columns / density_end_columns
TASK_CHUNKS = 32  # chunks per task (65,536 rays)
MIN_RAYS = 4 * TASK_CHUNKS * CHUNK  # smaller bundles (tables: tens of rays) stay serial; spawning costs ~0.5 s
ENV = 'SOSFUSE2_COLUMN_WORKERS'


def workers():
    text = os.environ.get(ENV, '').strip()
    if not text:
        return default_workers()
    try:
        count = int(text)
    except ValueError:
        raise ValueError('E_COLUMN_WORKERS: %s must be a positive integer' % ENV) from None
    if count < 1:
        raise ValueError('E_COLUMN_WORKERS: %s must be a positive integer' % ENV)
    return count


def freeze(namespace, names):
    """Serialize only the selected bindings, without the engine context."""
    try:
        return pickle.dumps(types.SimpleNamespace(**{name: getattr(namespace, name) for name in names}))
    except Exception as exc:
        raise TypeError('E_COLUMN_PARALLEL_FREEZE: ' + str(exc)) from exc


def thaw(frozen):
    return pickle.loads(frozen)


def source_hashes():
    """Raw hashes detect any source drift between the parent and spawned workers."""
    here = paths.PKG
    return {
        name: hashlib.sha256((here / name).read_bytes()).hexdigest()
        for name in ('rays/shell_segments.py', 'rays/incoming_quadrature.py', 'rays/incoming_rings.py', 'solve/engine_core.py')
    }


SOURCE_HASHES = source_hashes()


def source_drift(expected):
    """Return a diagnostic instead of raising inside a pool initializer."""
    try:
        actual = source_hashes()
    except OSError as exc:
        return Path(exc.filename).name if exc.filename else type(exc).__name__
    return next((name for name, digest in expected.items() if actual.get(name) != digest), None)


def check_source(drift):
    if drift is not None:
        raise RuntimeError('E_COLUMN_WORKER_SOURCE_DRIFT: ' + drift)


# ---- worker side ---------------------------------------------------------------------------------------------

_WORKER = {}


def _init(frozen, kind, starts, cell_off, parameter, sources):
    _WORKER["source_drift"] = source_drift(sources)
    if _WORKER["source_drift"] is not None:
        return
    _WORKER.update(old=thaw(frozen), kind=kind, starts=starts, cell_off=cell_off, parameter=parameter)


def _run(task):
    """The serial loop body for every 2048-ray chunk of [start, end)."""
    from ..atmosphere.density import correct_columns
    from ..atmosphere.stratosphere import columns

    check_source(_WORKER["source_drift"])
    start, end, dirs, base = task
    w = _WORKER
    old, starts, co, parameter = w['old'], w['starts'], w['cell_off'], w['parameter']
    out = []
    for lo in range(start, end, CHUNK):
        hi = min(lo + CHUNK, end)
        index = np.searchsorted(co, np.arange(lo, hi), side='right') - 1
        if w['kind'] == 'strato':
            out.append(columns(starts[index], dirs[lo - start : hi - start], old, parameter, 24))
        else:
            out.append(
                correct_columns(
                    starts[index], dirs[lo - start : hi - start], base[lo - start : hi - start], old, parameter, 24
                )
            )
    return start, out


# ---- parent side ---------------------------------------------------------------------------------------------


def event_fields():
    """Identity and clock fields carried by every PHASE line."""
    return dict(
        id=os.environ.get('SOSFUSE2_STATE_ID', 'COLUMN_CACHE'),
        pid=os.getpid(),
        timestamp_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
    )


def _event(kind, count, rays, tasks, result, reason, started):
    print(
        json.dumps(
            dict(
                event='PHASE',
                phase='column_parallel',
                **event_fields(),
                kind=kind,
                workers=count,
                rays=rays,
                tasks=tasks,
                result=result,
                reason=reason,
                seconds=time.monotonic() - started,
            )
        ),
        flush=True,
    )


def parallel_columns(out, kind, starts, cell_off, dirs, old, parameter, count=None):
    """Fill `out` in place like the serial loop and return True, or return False to let the caller run that loop.

    kind 'strato': out[a:b] = columns(...) (f8 -> f4 on assignment, as serially); parameter = strato profile.
    kind 'density': out = end_col copy; out[a:b] = correct_columns(..., out[a:b], ...); parameter = density optics.
    False: one worker, a bundle below MIN_RAYS, or a failure of the parallel machinery (reason printed;
    a numerical error then recurs in the serial loop and is raised there)."""
    from ..rays import column_table

    if column_table.method() != 'gl24':
        return column_table.fill(out, kind, starts, cell_off, dirs, old, parameter)
    count = workers() if count is None else int(count)
    rays = len(dirs)
    if count <= 1 or rays < MIN_RAYS:
        return False
    started = time.monotonic()
    span = CHUNK * TASK_CHUNKS
    tasks = (rays + span - 1) // span
    try:
        import multiprocessing
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        frozen = freeze(old, ['geometry_segments', 'probe'] if kind == 'density' else ['geometry_segments'])
        if kind == 'density':
            parameter = {k: v for k, v in parameter.items() if k != 'old'}
        starts = np.ascontiguousarray(starts)
        cell_off = np.ascontiguousarray(cell_off)
        # One snapshot of the inputs: density bases are the unmodified end_col rows of each task.
        bases = None if kind == 'strato' else out
        with ProcessPoolExecutor(
            min(count, tasks),
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init,
            initargs=(frozen, kind, starts, cell_off, parameter, SOURCE_HASHES),
        ) as pool:
            queue = iter(range(0, rays, span))
            pending = set()

            def submit():
                for lo in queue:
                    hi = min(lo + span, rays)
                    base = None if bases is None else np.array(bases[lo:hi])
                    pending.add(pool.submit(_run, (lo, hi, np.ascontiguousarray(dirs[lo:hi]), base)))
                    return True
                return False

            for _ in range(2 * count):
                submit()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.discard(future)
                    lo, parts = future.result()
                    for k, value in enumerate(parts):
                        a = lo + k * CHUNK
                        out[a : a + len(value)] = value
                    submit()
    except Exception as exc:
        # The serial loop follows and rewrites every row.  Density rows written by completed tasks keep the
        # unselected end_col columns unchanged and the selected ones equal to what the loop recomputes.
        reason = str(exc).split(':')[0] if str(exc).startswith('E_') else type(exc).__name__
        _event(kind, count, rays, tasks, 'fallback', reason, started)
        return False
    _event(kind, count, rays, tasks, 'parallel', None, started)
    return True
