"""Spawn cell calculations; the original packed writer owns ordering and resume.

Installed by the package initializer to preserve geometry.py's cache-key hash.
Only ResidentGeometry.cached_cell callbacks are parallelized; generic builders
and SOSFUSE2_GEOMETRY_WORKERS=1 use the accepted writer without adaptation.
"""

import json
import multiprocessing
import os
import time
import types
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace as NS

from . import column_parallel as cp
from .geometry import PackedCache as SerialPackedCache, ResidentGeometry

ENV = 'SOSFUSE2_GEOMETRY_WORKERS'
_GEOMETRY = None


def workers():
    value = os.environ.get(ENV, '').strip()
    try:
        count = cp.default_workers() if not value else int(value)
    except ValueError:
        raise ValueError('E_GEOMETRY_WORKERS: %s must be a positive integer' % ENV) from None
    if count < 1:
        raise ValueError('E_GEOMETRY_WORKERS: %s must be a positive integer' % ENV)
    return count


def freeze(g):
    b = g.scope['b']
    old = b.sh.old
    # incoming/ring_edges only need sh.old.probe.gl/frame, not the Solver,
    # exec-defined SunGeometry class, or the rest of the engine namespace.
    scope = dict(b.legacy.incoming.__globals__)
    scope['sh'] = NS(old=NS(probe=NS(gl=old.probe.gl, frame=old.probe.frame)))
    for name in ('incoming', 'ring_edges'):
        f = getattr(b.legacy, name)
        scope[name] = types.FunctionType(f.__code__, scope, f.__name__, f.__defaults__)
    return cp.freeze(
        NS(
            incoming=scope['incoming'],
            frame=old.probe.frame,
            boundaries=old.probe.boundaries,
            total_columns=old.total_columns,
            gl=old.probe.gl,
            profiles=old.profiles,
            basis_at=g.scope['basis_at'],
        ),
        ('incoming', 'frame', 'boundaries', 'total_columns', 'gl', 'profiles', 'basis_at'),
    )


def _init(frozen, geo, legacy, optics, profile):
    global _GEOMETRY
    f = cp.thaw(frozen)
    old = NS(
        probe=NS(frame=f.frame, boundaries=f.boundaries, gl=f.gl), total_columns=f.total_columns, profiles=f.profiles
    )
    g = ResidentGeometry.__new__(ResidentGeometry)
    g.s = NS(geo=geo, density_optics=optics, strato=dict(profile=profile))
    g.scope = dict(b=NS(legacy=NS(incoming=f.incoming), sh=NS(old=old)), basis_at=f.basis_at)
    g.legacy = legacy
    g.groups = {}
    g.stats = dict(cells=0, rays=0, samples=0, legacy_cells_read=0)
    _GEOMETRY = g


def _run(i):
    before = dict(_GEOMETRY.stats)
    host, meta = _GEOMETRY.cached_cell(i)
    return (
        host,
        meta,
        {k: v - before[k] for k, v in _GEOMETRY.stats.items()},
        _GEOMETRY.__dict__.pop('_joint_cells', {}).get(i),
    )


def _event(result, count, completed, reason, started):
    print(
        json.dumps(
            dict(
                event='PHASE',
                phase='geometry_parallel',
                **cp.event_fields(),
                result=result,
                workers=count,
                completed=completed,
                reason=reason,
                seconds=time.perf_counter() - started,
            )
        ),
        flush=True,
    )


class PackedCache(SerialPackedCache):
    def build(self, ncell, cell_arrays, progress=None):
        g = getattr(cell_arrays, '__self__', None)
        if (
            not isinstance(g, ResidentGeometry)
            or getattr(cell_arrays, '__func__', None) is not ResidentGeometry.cached_cell
        ):
            return super().build(ncell, cell_arrays, progress)
        count = min(workers(), ncell)
        if count <= 1:
            return super().build(ncell, cell_arrays, progress)
        started = time.perf_counter()
        completed = 0
        try:
            frozen = freeze(g)
            geo = NS(h=g.s.geo.h, chi=g.s.geo.chi, cfg=g.s.geo.cfg)
            optics = getattr(g.s, 'density_optics', None)
            optics = None if optics is None else {k: v for k, v in optics.items() if k != 'old'}
            profile = getattr(g.s, 'strato', {}).get('profile')
            with ProcessPoolExecutor(
                count,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_init,
                initargs=(frozen, geo, g.legacy, optics, profile),
            ) as pool:
                pending = deque()
                next_cell = None

                def ordered(i):
                    nonlocal next_cell, completed
                    if next_cell is None:
                        next_cell = i  # writer has already recovered/truncated its journal
                    while next_cell < ncell and len(pending) < 2 * count:
                        pending.append((next_cell, pool.submit(_run, next_cell)))
                        next_cell += 1
                    index, future = pending.popleft()
                    if index != i:
                        raise RuntimeError('E_GEOMETRY_ORDER: writer requested unexpected cell')
                    host, meta, delta, joint = future.result()
                    if joint is not None:
                        g.__dict__.setdefault('_joint_cells', {})[i] = joint
                    for k, v in delta.items():
                        g.stats[k] += v
                    g.groups.clear()
                    completed += 1
                    return host, meta

                try:
                    super().build(ncell, ordered, progress)
                finally:
                    for _, future in pending:
                        future.cancel()
        except Exception as exc:
            _event('fallback', count, completed, type(exc).__name__ + ': ' + str(exc), started)
            # Original writer resumes after the committed prefix and overwrites
            # any uncommitted bytes, including a partially written failed cell.
            return super().build(ncell, cell_arrays, progress)
        _event('parallel', count, completed, None, started)


# Also covers importing this adapter before geometry (including spawn workers).
from . import geometry

geometry.PackedCache = PackedCache
