"""Persistent-service IPC error path (pipeline_execution.wire_recv): a service failure string in place of the array
descriptor keeps its reason when an ENOSPC error surfaces as a ValueError."""

import multiprocessing as mp
import numpy as np
import pytest
from sosfuse2.jobs.pipeline_execution import wire_recv, wire_send


def _quiet(*_, **__):
    pass


def test_service_error_string_raises_with_its_reason():
    a, b = mp.Pipe()
    a.send('E_PERSISTENT_STATE: OSError: [Errno 28] No space left on device')
    with pytest.raises(RuntimeError, match=r'^E_PERSISTENT_STATE: OSError: \[Errno 28\]'):
        wire_recv(b, _quiet, 'S')
    a.send('unexpected text')
    with pytest.raises(RuntimeError, match='^E_PERSISTENT_STATE: unexpected text$'):
        wire_recv(b, _quiet, 'S')


def test_descriptor_round_trip_still_works(tmp_path):
    a, b = mp.Pipe()
    value = dict(x=np.arange(5.0), n=3)
    with wire_send(a, value, tmp_path, _quiet, 'S'):
        got = wire_recv(b, _quiet, 'S')
    assert got['n'] == 3 and np.array_equal(got['x'], value['x'])


def test_state_geometry_views_and_dynamic_columns(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sosfuse2.jobs.pipeline_execution import geometry_buffers

    monkeypatch.setenv('SOS_PERSISTENT_ROOT', str(tmp_path))
    packed = tmp_path / 'packed.bin'
    np.arange(16, dtype='u4').tofile(packed)
    mapped = np.asarray(np.memmap(packed, 'u4', mode='r', offset=4))[2:9]
    paths = np.arange(32, dtype='f4')
    column = np.ones(7, dtype='f4')
    bundle = SimpleNamespace(arrays=dict(mapped=mapped, end_col=column), _device_paths=dict(edges=paths))
    events = []
    emit = lambda name, phase, **fields: events.append((phase, fields['bytes']))
    a, b = mp.Pipe()
    with geometry_buffers() as shared:
        for value in (2, 3):
            column[:] = value
            shared.select(bundle)
            with wire_send(a, (mapped, paths, column), tmp_path, emit, 'S', shared):
                restored = wire_recv(b, emit, 'S')
                for actual, expected in zip(restored, (mapped, paths, column)):
                    np.testing.assert_array_equal(actual, expected)
        assert shared.file.tell() == paths.nbytes
        assert events[0][1] - events[2][1] == paths.nbytes
    assert list(tmp_path.iterdir()) == [packed]


def test_paired_replay_preserves_both_orders_and_checks_directions(monkeypatch):
    from types import SimpleNamespace
    from sosfuse2.jobs import pipeline_execution as pe
    from sosfuse2.outputs.radiance_table import _capture
    from sosfuse2.rays.shell_segments import frame
    from sosfuse2.solve.engine_core import Solver

    s = object.__new__(Solver)
    s.W = 3
    s.solar = np.array([0.7, 1.1, 1.3])
    s.active = np.ones(2 * s.W, 'i4')
    s.export_first_order = True
    s.ctx = SimpleNamespace(b=SimpleNamespace(R=6371, sh=SimpleNamespace(OBS=0, old=SimpleNamespace(
        probe=SimpleNamespace(frame=frame)))))
    s.kernel = SimpleNamespace(xp=object(), host=np.asarray, sync=lambda: None)
    s.geo = SimpleNamespace(pack=lambda p, d, n: {'d': d.copy()})
    ee, aa = np.meshgrid(np.arange(5), np.arange(17), indexing='ij')
    hs = [1., -6.]
    dummy = np.empty(0)
    captured = [_capture(s, h, ee, aa, dummy) for h in hs]
    sizes = [len(raw['d']) for raws in captured for raw in raws]
    bundle = SimpleNamespace(sizes=sizes, rays=sum(sizes))
    rng = np.random.default_rng(21)
    rays = rng.random((2, bundle.rays, s.W, 4), dtype='f4')
    s.kernel.launch = lambda bundle, s, field, first, phase: (None, None, rays[int(first)])
    monkeypatch.setattr(pe, 'promote_bundle', lambda host, xp: host)
    prepared = dict(group=32, shape=(len(hs), *ee.shape, s.W), hs=hs, ee=ee, aa=aa,
                    batches=[dict(bundle=bundle, flat=None, part=range(len(hs)), captured=captured)])
    full = pe.evaluate_tables(s, dummy, prepared)
    reference = pe.replay_tables(s, prepared, full)
    first = s.first_order_table.copy()
    paired = pe.evaluate_tables(s, dummy, prepared, paired=True)
    np.testing.assert_array_equal(pe.replay_tables(s, prepared, paired), reference)
    np.testing.assert_array_equal(s.first_order_table, first)
    captured[1][-1]['d'][0] = 0
    with pytest.raises(RuntimeError, match='E_REPLAY_DRIFT'):
        pe.replay_tables(s, prepared, paired)


@pytest.mark.parametrize('fail', [False, True])
def test_one_chunk_ahead_joins_worker_and_consumes_previous_decision(tmp_path, monkeypatch, fail):
    from threading import Event
    from types import SimpleNamespace
    from sosfuse2.jobs import run_state
    from sosfuse2.atmosphere import density

    monkeypatch.setenv('SOS_PERSISTENT_ROOT', str(tmp_path))
    monkeypatch.setattr(run_state, 'phase', lambda *args, **kw: None)
    monkeypatch.setattr(density, 'chunk_metadata', lambda *args: {})
    monkeypatch.setattr(density, 'evidence', lambda *args: {})
    started, release, finished = Event(), Event(), Event()
    seen = []
    job = SimpleNamespace(wl=np.arange(31), persistent=True, pipeline_lock=True, name='state',
                          trace=None, ksplit_decision=None)
    args = SimpleNamespace(backend='cupy', reduced=False)

    def build(job, a, np, index, chunk, *, prepare_table=False):
        solver = SimpleNamespace(kernel=SimpleNamespace(xp=np, profile_n=161))
        if index == 1:
            started.set()
            assert release.wait(5)
            finished.set()
        return solver, object(), [] if prepare_table else None

    def execute(job, a, np, count, configure, default_root, on_sent):
        index = job.index
        assert getattr(job.s, 'ksplit_decision', None) == (index or None)
        seen.append(job.s)
        on_sent()
        if not index:
            try:
                assert started.wait(5)
                assert job.s is seen[0]
                if fail:
                    raise RuntimeError('service failed')
            finally:
                release.set()
        job.ksplit_decision = index + 1
        return True

    monkeypatch.setattr(run_state, '_build_chunk_inputs', build)
    monkeypatch.setattr(run_state, '_prepare_chunk_geometry', lambda *args: None)
    monkeypatch.setattr(run_state, '_prepare_pipeline_chunk', execute)
    def run():
        run_state._solve_wavelength_chunks(job, args, np, None, None, None, None, None, None, None, None)
    if fail:
        with pytest.raises(RuntimeError, match='service failed'):
            run()
        assert len(seen) == 1
    else:
        run()
        assert len({id(s) for s in seen}) == 3
    assert finished.is_set() and not list(tmp_path.iterdir())
