"""Opt-in process parallelism for the cold column builds (fused.Kernel.strato_end_columns / density_end_columns
and the host-cache endpoint02 build, host_cache.endpoint_columns).

Numerical equivalence with the serial loop is checked at rtol 1e-6: the same 2048-ray chunks (fused.py build loops), the same
per-chunk call (stratosphere.columns / density.correct_columns) on equal input bytes, and each chunk's result is
placed at its own index; no sum crosses a chunk.  Workers are spawned processes (no GIL; never a fork of a parent
that may hold CUDA or threads).  The accepted solver functions (old.geometry_segments and the probe it calls) are
exec-defined in dict scopes (rig/engine.py definitions) and cannot be pickled by reference, so they are shipped as
marshalled code objects plus the globals their code names (freeze/thaw below); the bytecode is the parent's own.

The process count is SOSFUSE2_COLUMN_WORKERS; unset = all usable CPUs (default_workers), 1 = the serial loop.
pipeline.py exports max(--threads, usable CPUs // concurrent states) unless the variable is set (pipeline.prep_workers).  The worker count enters no cache key: column_cache.cache_key hashes stratosphere.py, density.py, the
accepted sources and rig/engine.py; host_cache keys hash host_cache.py (which calls this module), not this file.  test_column_parallel.py checks shape, dtype and numerical tolerance.
"""
import builtins
import json
import marshal
import os
import pickle
import time
import types

import numpy as np

# Moved to host_limits (stdlib only, so pipeline.py can use it); re-exported for sun/ground/geometry_parallel.
from .host_limits import _quota_limits, default_workers  # noqa: F401

CHUNK = 2048                      # fused.py strato_end_columns / density_end_columns
TASK_CHUNKS = 32                  # chunks per task (65,536 rays)
MIN_RAYS = 4*TASK_CHUNKS*CHUNK    # smaller bundles (tables: tens of rays) stay serial; spawning costs ~0.5 s
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


# ---- shipping exec-defined functions -------------------------------------------------------------------------

def _names(code):
    out = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            out |= _names(const)
    return out


class _Freezer:
    """Function -> (marshalled code, name, defaults, scope id); a scope keeps only the globals some code names."""

    def __init__(self):
        self.scopes = {}

    def value(self, v):
        if isinstance(v, types.ModuleType):
            return ('module', v.__name__)
        if isinstance(v, types.FunctionType):
            return self.function(v)
        if isinstance(v, types.SimpleNamespace):
            return ('namespace', {k: self.value(x) for k, x in vars(v).items() if not k.startswith('__')})
        pickle.dumps(v)                    # unsupported values fail here, before any worker starts
        return ('value', v)

    def function(self, f):
        if f.__closure__:
            raise TypeError('E_COLUMN_PARALLEL_FREEZE: closure in ' + f.__name__)
        sid = id(f.__globals__)
        scope = self.scopes.setdefault(sid, {})
        for name in _names(f.__code__):
            if name in f.__globals__ and name not in scope and not name.startswith('__'):
                scope[name] = None         # reserve first: recursive functions share the scope
                scope[name] = self.value(f.__globals__[name])
        return ('function', marshal.dumps(f.__code__), f.__name__, f.__defaults__, f.__kwdefaults__, sid)


def freeze(namespace, names):
    freezer = _Freezer()
    parts = {name: freezer.value(getattr(namespace, name)) for name in names}
    return dict(scopes=freezer.scopes, parts=parts)


def thaw(frozen):
    import importlib
    scopes = {sid: {'__builtins__': builtins} for sid in frozen['scopes']}

    def build(item):
        tag = item[0]
        if tag == 'module':
            return importlib.import_module(item[1])
        if tag == 'namespace':
            return types.SimpleNamespace(**{k: build(x) for k, x in item[1].items()})
        if tag == 'function':
            _, code, name, defaults, kwdefaults, sid = item
            f = types.FunctionType(marshal.loads(code), scopes[sid], name, defaults)
            f.__kwdefaults__ = kwdefaults
            return f
        return item[1]

    for sid, entries in frozen['scopes'].items():
        for name, item in entries.items():
            scopes[sid][name] = build(item)
    return types.SimpleNamespace(**{k: build(x) for k, x in frozen['parts'].items()})


# ---- worker side ---------------------------------------------------------------------------------------------

_WORKER = {}


def _init(frozen, kind, starts, cell_off, parameter):
    _WORKER.update(old=thaw(frozen), kind=kind, starts=starts, cell_off=cell_off, parameter=parameter)


def _run(task):
    """The serial loop body for every 2048-ray chunk of [start, end)."""
    from .stratosphere import columns
    from .density import correct_columns
    start, end, dirs, base = task
    w = _WORKER; old, starts, co, parameter = w['old'], w['starts'], w['cell_off'], w['parameter']
    out = []
    for lo in range(start, end, CHUNK):
        hi = min(lo+CHUNK, end); index = np.searchsorted(co, np.arange(lo, hi), side='right')-1
        if w['kind'] == 'strato':
            out.append(columns(starts[index], dirs[lo-start:hi-start], old, parameter, 24))
        else:
            out.append(correct_columns(starts[index], dirs[lo-start:hi-start], base[lo-start:hi-start], old, parameter, 24))
    return start, out


# ---- parent side ---------------------------------------------------------------------------------------------

def event_fields():
    """Identity and clock fields every PHASE line carries (pipeline_trace.py reads pid/timestamp_ns)."""
    return dict(id=os.environ.get('SOSFUSE2_STATE_ID', 'COLUMN_CACHE'), pid=os.getpid(),
                timestamp_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())


def _event(kind, count, rays, tasks, result, reason, started):
    print(json.dumps(dict(event='PHASE', phase='column_parallel', **event_fields(), kind=kind, workers=count,
        rays=rays, tasks=tasks, result=result, reason=reason, seconds=time.monotonic()-started)), flush=True)


def parallel_columns(out, kind, starts, cell_off, dirs, old, parameter, count=None):
    """Fill `out` in place like the serial loop and return True, or return False to let the caller run that loop.

    kind 'strato': out[a:b] = columns(...) (f8 -> f4 on assignment, as serially); parameter = strato profile.
    kind 'density': out = end_col copy; out[a:b] = correct_columns(..., out[a:b], ...); parameter = density optics.
    False: one worker (default), a bundle below MIN_RAYS, or a failure of the parallel machinery (reason printed;
    a numerical error then recurs in the serial loop and is raised there)."""
    from . import column_table
    if column_table.method() != 'gl24':
        return column_table.fill(out, kind, starts, cell_off, dirs, old, parameter)
    count = workers() if count is None else int(count)
    rays = len(dirs)
    if count <= 1 or rays < MIN_RAYS:
        return False
    started = time.monotonic(); span = CHUNK*TASK_CHUNKS; tasks = (rays+span-1)//span
    try:
        import multiprocessing
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        frozen = freeze(old, ['geometry_segments'])
        if kind == 'density':
            parameter = {k: v for k, v in parameter.items() if k != 'old'}
        starts = np.ascontiguousarray(starts); cell_off = np.ascontiguousarray(cell_off)
        # One snapshot of the inputs: density bases are the unmodified end_col rows of each task.
        bases = None if kind == 'strato' else out
        with ProcessPoolExecutor(min(count, tasks), mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init, initargs=(frozen, kind, starts, cell_off, parameter)) as pool:
            queue = iter(range(0, rays, span)); pending = set()

            def submit():
                for lo in queue:
                    hi = min(lo+span, rays)
                    base = None if bases is None else np.array(bases[lo:hi])
                    pending.add(pool.submit(_run, (lo, hi, np.ascontiguousarray(dirs[lo:hi]), base)))
                    return True
                return False

            for _ in range(2*count):
                submit()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.discard(future)
                    lo, parts = future.result()
                    for k, value in enumerate(parts):
                        a = lo+k*CHUNK; out[a:a+len(value)] = value
                    submit()
    except Exception as exc:
        # The serial loop follows and rewrites every row.  Density rows written by completed tasks keep the
        # unselected end_col columns unchanged and the selected ones equal to what the loop recomputes.
        reason = str(exc).split(':')[0] if str(exc).startswith('E_') else type(exc).__name__
        _event(kind, count, rays, tasks, 'fallback', reason, started)
        return False
    _event(kind, count, rays, tasks, 'parallel', None, started)
    return True
