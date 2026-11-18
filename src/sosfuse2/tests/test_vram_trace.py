"""Pool lifecycle, partial uploads, and state reports without requiring a CUDA device."""

import json
from types import SimpleNamespace

import pytest

from sosfuse2.backends import vram_trace as vt
from sosfuse2.jobs import run_state
from sosfuse2.jobs.pipeline_execution import promote_bundle

MIB = 1024**2


class OutOfMemoryError(MemoryError):
    pass


class Pool:
    def __init__(self, limit):
        self.limit = limit
        self.used = self.total = 0
        self.hooks = []
        pool = self

        class Hook:
            def __enter__(self):
                pool.hooks.append(self)

            def __exit__(self, *_):
                assert pool.hooks.pop() is self

        self.xp = SimpleNamespace(
            __version__='test',
            cuda=SimpleNamespace(MemoryHook=Hook, memory=SimpleNamespace(OutOfMemoryError=OutOfMemoryError)),
            get_default_memory_pool=lambda: self,
            asarray=self.allocate,
        )

    def used_bytes(self):
        return self.used

    def total_bytes(self):
        return self.total

    def allocate(self, size):
        n = (size + 511) // 512 * 512
        kw = {'device_id': 0, 'size': size, 'mem_size': n}
        hook = self.hooks[-1]
        hook.malloc_preprocess(**kw)
        if self.used + n > self.limit:
            self.total = self.used
            hook.malloc_postprocess(**kw, mem_ptr=0, pmem_id=0)
            raise OutOfMemoryError(f'request={n} limit={self.limit}')
        block = Block(self, n)
        self.used += n
        self.total = max(self.used, self.total)
        hook.malloc_postprocess(**kw, mem_ptr=id(block), pmem_id=id(block))
        return block


class Block:
    def __init__(self, pool, n):
        self.pool, self.n = pool, n

    def __del__(self):
        self.pool.used -= self.n
        if self.pool.hooks:
            self.pool.hooks[-1].free_postprocess(device_id=0, mem_size=self.n, mem_ptr=id(self), pmem_id=id(self))


def events():
    rows = []

    def emit(state, phase, **fields):
        rows.append(dict(id=state, phase=phase, **fields))

    return rows, emit


def test_named_upload_peak_free_and_cached_blocks():
    rows, emit = events()
    pool = Pool(4 * MIB)
    with vt.session('sky', emit) as t:
        t.bind(pool.xp)
        t.bind(pool.xp)
        t.chunk, t.stage, t.piece = 2, 'table_transport', 5
        with vt.item('query.table_geometry'):
            bundle = promote_bundle(SimpleNamespace(arrays={'dirs': MIB - 1, 'edges': 33}), pool.xp)
        alias = bundle.arrays['dirs']
        del bundle
        assert pool.used == MIB
        del alias
        assert pool.used == 0 and pool.total == MIB + 512
        summary = vt.metadata()['vram_trace']
    assert pool.hooks == [] and vt.current.get() is None
    assert [r['phase'] for r in rows] == ['vram_trace_start', 'vram_alloc', 'vram_free']
    alloc, free = rows[1:]
    assert alloc['requested_bytes'] == MIB - 1 and alloc['bytes'] == MIB
    assert alloc['allocation'] == free['allocation'] and free['pool_used_bytes'] == 0
    assert free['pool_total_bytes'] == MIB + 512
    assert summary['max_resident_bytes'] == {'query.table_geometry.dirs': MIB, 'query.table_geometry.edges': 512}
    peak = summary['peak']
    assert (peak['chunk'], peak['stage'], peak['piece']) == (2, 'table_transport', 5)
    assert peak['live_by_state_bytes'] == {'sky': summary['max_resident_bytes']}
    assert peak['unattributed_used_bytes'] == 0


def test_oom_snapshot_precedes_temporary_release_and_preserves_reason():
    rows, emit = events()
    pool = Pool(2 * MIB)
    with pytest.raises(OutOfMemoryError, match=f'request={2 * MIB} limit={2 * MIB}'), vt.session('sky', emit) as t:
        t.bind(pool.xp)
        t.iteration = 4
        with vt.item('source.composition_scratch'):
            # The first argument loses its last reference while the second raises.
            (pool.allocate(MIB), pool.allocate(2 * MIB))
    assert pool.used == 0 and not pool.hooks
    oom = rows[-1]
    assert oom['phase'] == 'vram_oom' and oom['snapshot_at'] == 'malloc_failed'
    assert oom['reason'] == f'OutOfMemoryError: request={2 * MIB} limit={2 * MIB}'
    assert oom['pool_used_bytes'] == MIB and oom['bytes'] == 2 * MIB
    assert oom['live_by_state_bytes'] == {'sky': {'source.composition_scratch': MIB}}
    assert oom['iteration'] == 4 and oom['peak']['pool_used_bytes'] == MIB
    assert [r['phase'] for r in rows[-2:]] == ['vram_free', 'vram_oom']


def test_service_handoff_and_cross_chunk_metadata():
    rows, emit = events()
    pool, ledger = Pool(8 * MIB), vt.Ledger()
    # Model the service's serialized handoffs while the first state holds its history.
    with vt.session('first', emit, ledger) as first:
        first.bind(pool.xp)
        with vt.item('history'):
            a, b = pool.allocate(MIB), pool.allocate(MIB)
        with vt.session('second', emit, ledger) as second:
            second.bind(pool.xp)
            second.chunk = 1
            with vt.item('moments'):
                c = pool.allocate(3 * MIB)
            del a, c
            later = second.summary()
        del b
        earlier = first.summary()
    assert later['peak']['live_by_state_bytes'] == {'first': {'history': 2 * MIB}, 'second': {'moments': 3 * MIB}}
    assert earlier['max_resident_bytes'] == {'history': 2 * MIB}
    assert [r for r in rows if r.get('released_by') == 'second' and r['id'] == 'first']
    with vt.session('worker', emit):
        meta = vt.metadata([later, earlier])['vram_trace']
    assert meta['peak'] == later['peak']
    assert meta['max_resident_bytes'] == {'history': 2 * MIB, 'moments': 3 * MIB}
    assert not ledger.blocks and not ledger.owners


def test_late_free_is_charged_to_original_state():
    rows, emit = events()
    pool, ledger = Pool(4 * MIB), vt.Ledger()
    with vt.session('first', emit, ledger) as first:
        first.bind(pool.xp)
        with vt.item('lut'):
            cached = pool.allocate(MIB)
    with vt.session('second', emit, ledger) as second:
        second.bind(pool.xp)
        with vt.item('source'):
            field = pool.allocate(MIB)
        del cached, field
    assert second.peak['live_by_state_bytes'] == {'first': {'lut': MIB}, 'second': {'source': MIB}}
    assert next(r for r in rows if r['phase'] == 'vram_free')['id'] == 'first'
    assert not ledger.blocks and not ledger.owners


@pytest.mark.parametrize(
    ('backend', 'flag', 'active'), [('numpy', '1', False), ('cupy', '0', False), ('cupy', '1', True)]
)
def test_runner_opt_in(monkeypatch, backend, flag, active):
    monkeypatch.setenv('SOSFUSE2_VRAM_TRACE', flag)

    @run_state.trace_state
    def execute(_):
        assert (vt.current.get() is not None) == active
        assert bool(vt.metadata()) == active
        raise ValueError('original')

    with pytest.raises(ValueError, match='original'):
        execute(SimpleNamespace(backend=backend, out_name='sky', id='sky'))
    assert vt.current.get() is None


def test_installed_cupy_pool():
    if not vt.enabled('cupy'):
        pytest.skip('enable tracing to run the installed CUDA pool probe')
    cp = pytest.importorskip('cupy')
    rows, emit = events()
    pool = cp.get_default_memory_pool()
    old_limit = pool.get_limit()
    # Request more than the entire capped pool. The driver never needs to satisfy it.
    cap = pool.total_bytes() + 4 * MIB
    try:
        pool.set_limit(size=cap)
        with pytest.raises(cp.cuda.memory.OutOfMemoryError), vt.session('probe', emit) as t:
            t.bind(cp)
            with vt.item('probe.live'):
                block = cp.empty(MIB, dtype='u1')
            try:
                with vt.item('probe.request'):
                    cp.empty(cap + MIB, dtype='u1')
            finally:
                del block
    finally:
        pool.set_limit(size=old_limit)
    oom = next(r for r in rows if r['phase'] == 'vram_oom')
    assert oom['snapshot_at'] in ('malloc_begin', 'malloc_failed') and oom['item'] == 'probe.request'
    assert oom['live_by_state_bytes'] == {'probe': {'probe.live': MIB}}
    assert any(r['phase'] == 'vram_free' and r['item'] == 'probe.live' for r in rows)
    print(json.dumps(rows, allow_nan=False))
