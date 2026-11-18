"""Opt-in CuPy pool accounting. The ledger stores allocation metadata, never arrays."""

import os
import sys
from collections import Counter
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from itertools import count
from threading import RLock

current = ContextVar('vram_trace', default=None)
_NO_SCOPE = nullcontext()
EVENT_MIN_BYTES = 1024**2


def enabled(backend):
    return backend == 'cupy' and os.environ.get('SOSFUSE2_VRAM_TRACE') == '1'


def item(name):
    trace = current.get()
    return trace.label(name) if trace is not None else _NO_SCOPE


def _site():
    frame = sys._getframe(1)
    while frame is not None:
        mod = frame.f_globals.get('__name__', '')
        if mod.startswith('sosfuse2.') and mod != __name__:
            return f'{mod.removeprefix("sosfuse2.")}.{frame.f_code.co_qualname}:{frame.f_lineno}'
        frame = frame.f_back
    return 'external'


class Ledger:
    """Shared by the active threads of one device service."""

    def __init__(self):
        self.lock = RLock()
        self.blocks = {}
        self.owners = set()
        self.sequence = count(1)


class Trace:
    def __init__(self, name, emit, ledger):
        self.name, self.emit, self.ledger = name, emit, ledger
        self.live = Counter()
        self.maxima = {}
        self.peak = self.pending = self.failed = None
        self.hook = self.pool = self.oom_type = None
        self.closed = False
        self.chunk, self.stage, self.iteration = 0, 'setup', None
        self.piece = None
        self.item_name = None

    @contextmanager
    def label(self, name):
        old, self.item_name = self.item_name, name
        try:
            yield
        finally:
            self.item_name = old

    def bind(self, xp):
        if self.hook is not None:
            return
        from .device_profile import vram_pool_api

        self.pool, base, self.oom_type = vram_pool_api(xp)
        trace = self

        class Hook(base):
            name = 'sosfuse2_vram_trace'

            def malloc_preprocess(self, device_id, size, mem_size):
                trace.before(size, mem_size)

            def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
                trace.after((device_id, mem_ptr), size, mem_size, pmem_id)

            def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id):
                trace.released((device_id, mem_ptr), pmem_id)

        hook = Hook()
        hook.__enter__()
        self.hook = hook
        self.emit(
            self.name,
            'vram_trace_start',
            cupy_version=xp.__version__,
            event_min_bytes=EVENT_MIN_BYTES,
            **self.pool_sample(),
        )

    def pool_sample(self):
        return {'pool_used_bytes': int(self.pool.used_bytes()), 'pool_total_bytes': int(self.pool.total_bytes())}

    def position(self):
        return {'chunk': self.chunk, 'stage': self.stage, 'iteration': self.iteration, 'piece': self.piece}

    def composition(self, sample):
        # Include parked states when the service releases its compute gate for observer replay.
        live = {}
        for owner in self.ledger.owners:
            if owner.live:
                live.setdefault(owner.name, Counter()).update(owner.live)
        attributed = sum(sum(items.values()) for items in live.values())
        return dict(
            **sample,
            live_by_state_bytes={k: dict(v) for k, v in live.items()},
            unattributed_used_bytes=sample['pool_used_bytes'] - attributed,
        )

    def before(self, size, mem_size):
        if not size:
            return
        site = _site()
        with self.ledger.lock:
            self.pending = dict(
                item=self.item_name or site,
                site=site,
                requested_bytes=int(size),
                bytes=int(mem_size),
                **self.position(),
                **self.composition(self.pool_sample()),
            )

    def after(self, key, size, mem_size, pmem_id):
        if not size:
            return
        with self.ledger.lock:
            request, self.pending = self.pending, None
            if not key[1]:
                # CuPy may release cached blocks during the failed request. Sample before Python unwinds temporaries.
                self.failed = dict(request, **self.composition(self.pool_sample()))
                return
            name, site, n = request['item'], request['site'], int(mem_size)
            self.failed = None
            serial = next(self.ledger.sequence)
            if key in self.ledger.blocks:
                # A non-traced thread can return a block. Reuse proves the old lease ended.
                self.released(key, None)
            self.ledger.blocks[key] = (self, name, site, n, int(size), serial, self.chunk, pmem_id)
            self.live[name] += n
            self.maxima[name] = max(self.maxima.get(name, 0), self.live[name])
            sample = self.pool_sample()
            if self.peak is None or sample['pool_used_bytes'] > self.peak['pool_used_bytes']:
                self.peak = dict(item=name, site=site, **self.position(), **self.composition(sample))
            if n >= EVENT_MIN_BYTES:
                self.emit(
                    self.name,
                    'vram_alloc',
                    item=name,
                    site=site,
                    bytes=n,
                    requested_bytes=int(size),
                    allocation=serial,
                    **self.position(),
                    **sample,
                )

    def released(self, key, pmem_id):
        with self.ledger.lock:
            entry = self.ledger.blocks.get(key)
            if entry is None or (pmem_id is not None and entry[-1] != pmem_id):
                return
            del self.ledger.blocks[key]
            owner, name, site, n, requested, serial, chunk, _ = entry
            owner.live[name] -= n
            if not owner.live[name]:
                del owner.live[name]
            if owner.closed and not owner.live:
                self.ledger.owners.discard(owner)
            if n >= EVENT_MIN_BYTES:
                owner.emit(
                    owner.name,
                    'vram_free' if pmem_id is not None else 'vram_free_inferred_on_reuse',
                    item=name,
                    site=site,
                    bytes=n,
                    requested_bytes=requested,
                    allocation=serial,
                    allocation_chunk=chunk,
                    released_by=self.name,
                    **self.position(),
                    **self.pool_sample(),
                )

    def report_oom(self, exc):
        if self.oom_type is None or not isinstance(exc, self.oom_type):
            return
        with self.ledger.lock:
            snapshot = self.pending or self.failed
            if snapshot is None:
                origin = 'exception_handler'
                snapshot = dict(**self.position(), **self.composition(self.pool_sample()))
            else:
                origin = 'malloc_begin' if self.pending is not None else 'malloc_failed'
            self.emit(
                self.name,
                'vram_oom',
                reason=f'{type(exc).__name__}: {exc}',
                snapshot_at=origin,
                **snapshot,
                **self.summary(),
            )

    def summary(self):
        return {'max_resident_bytes': dict(self.maxima), 'peak': self.peak}

    def close(self):
        if self.hook is not None:
            self.hook.__exit__(None, None, None)
            self.hook = None
        with self.ledger.lock:
            self.closed = True
            if not self.live:
                self.ledger.owners.discard(self)


@contextmanager
def session(name, emit, ledger=None):
    trace = Trace(name, emit, ledger or Ledger())
    token = current.set(trace)
    with trace.ledger.lock:
        trace.ledger.owners.add(trace)
    try:
        yield trace
    except BaseException as exc:
        trace.report_oom(exc)
        raise
    finally:
        try:
            trace.close()
        finally:
            current.reset(token)


def metadata(chunks=()):
    trace = current.get()
    if trace is None:
        return {}
    summaries = [trace.summary()] if trace.hook is not None else chunks
    maxima = Counter()
    peak = None
    for summary in summaries:
        maxima |= Counter(summary['max_resident_bytes'])
        candidate = summary['peak']
        if candidate is not None and (peak is None or candidate['pool_used_bytes'] > peak['pool_used_bytes']):
            peak = candidate
    return {'vram_trace': {'max_resident_bytes': dict(maxima), 'peak': peak}}
