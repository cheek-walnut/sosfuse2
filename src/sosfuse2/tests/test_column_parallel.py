"""Cold column builds with SOSFUSE2_COLUMN_WORKERS > 1 give the serial values within rtol 1e-6 (column_parallel.py).  CPU only.
Kinds: fused strato_end_columns, fused density_end_columns, host_cache.endpoint_columns (endpoint02 build).

Real rays: the first 404,496 rays (the ground bundle) of the full-grid packed geometry cache; `old` and the
usstd76 density optics come from the production solver constructor (plan_queue.probe -> create_configured).
The production methods fused.Kernel.strato_end_columns / density_end_columns run with the column cache off.
The test asserts that the parallel path really ran (its PHASE event), not a serial fallback; a second test injects a
worker exception and checks that the serial rerun gives the serial values within rtol 1e-6.

Run: python -B -m pytest -q --pyargs sosfuse2.tests.test_column_parallel
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sosfuse2.precompute import column_parallel
from sosfuse2.jobs import plan_queue
from sosfuse2.tests import test_plan_queue as tp

try:
    from sosfuse2.checks.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


def flaky_segments(segments, p, d, n):
    """The accepted segments, except a worker fails on the bundle's short last chunk (404,496 = 197 x 2048 + 1040):
    the other tasks complete and write their rows first, so the serial rerun starts from a partly written array."""
    if len(d) < 2048:
        raise RuntimeError('E_TEST_WORKER_FAILURE')
    return segments(p, d, n)


@unittest.skipIf(tp.DATA is None, 'packed full-grid geometry cache not found (set SADDLE_DATA_ROOT)')
class ColumnParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sosfuse2.solve import ray_geometry as gm
        from sosfuse2.jobs import run_state as rv
        from sosfuse2.solve.source_solver import Kernel
        from sosfuse2.atmosphere.stratosphere import profile_config

        options = plan_queue.parser().parse_args(['--states', str(tp.LUT2), '--geometry-cache', '.', '--out', 'x'])
        options.reduced = False
        state = rv.effective_state(
            next(s for s in rv.read_states(tp.LUT2) if s['id'] == 'B22s5_tau0p01_aod0p15'), options
        )
        with tempfile.TemporaryDirectory() as tmp:
            # These ray bundles are the recorded nr20/nphi64 geometry, not the new V14 default.
            options.settings_json = Path(tmp) / 'settings.json'
            options.settings_json.write_text(json.dumps(dict(nr=20, h_refine='none', nphi=64)))
            cls.p = plan_queue.probe(
                state, rv.settings_for(options), tp.DATA / 'results/sos_fuse/geometry_cache', Path(tmp)
            )
        host, _ = gm.PackedCache(tp.DATA / 'results/sos_fuse/geometry_cache' / tp.SIGNATURE / 'packed').open()
        cls.rays = rays = 404496
        co = np.asarray(host['cell_off'])
        cls.cells = cells = int(np.searchsorted(co, rays))
        off = co[: cells + 1].copy()
        off[-1] = rays
        cls.arrays = dict(
            cell_off=off,
            starts=np.asarray(host['starts'][:cells]),
            dirs=np.asarray(host['dirs'][:rays]),
            end_col=np.asarray(host['end_col'][:rays]),
        )
        cls.c = dict(strato_old=cls.p.old, strato_profile=profile_config(state), density_optics=cls.p.s.density_optics)
        cls.kernel = Kernel(backend='numpy')

    def run_kind(self, kind, workers):
        """(array, column_parallel events) of one cold build through the production method."""
        from sosfuse2.solve import ray_geometry as gm
        from sosfuse2.precompute import host_cache

        bundle = gm.Bundle.__new__(gm.Bundle)
        bundle.arrays = self.arrays
        bundle.rays = self.rays
        bundle.cells = self.cells
        text = io.StringIO()
        with (
            mock.patch.dict(os.environ, {'SOSFUSE2_COLUMN_WORKERS': str(workers), 'SOSFUSE2_COLUMN_METHOD': 'gl24'}),
            contextlib.redirect_stdout(text),
        ):
            os.environ.pop('SOSFUSE2_COLUMN_CACHE', None)
            if kind == 'strato':
                value = self.kernel.strato_end_columns(bundle, self.c)
            elif kind == 'density':
                value = self.kernel.density_end_columns(bundle, self.c)
            else:  # host-cache endpoint02 build only (no cache file is read or written)
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    mock.patch.object(host_cache, 'cached', lambda folder, key, build, compressed=False: build()),
                ):
                    value = host_cache.endpoint_columns(
                        self.p.s, self.arrays, self.c['density_optics'], Path(tmp), tp.SIGNATURE
                    )
        events = [json.loads(line) for line in text.getvalue().splitlines() if '"column_parallel"' in line]
        return np.asarray(value), events

    def test_parallel_bytes_equal_serial(self):
        self.assertGreaterEqual(self.rays, column_parallel.MIN_RAYS)
        for kind in ('strato', 'density', 'endpoint'):
            a, events = self.run_kind(kind, 1)
            self.assertEqual(events, [], kind)
            b, events = self.run_kind(kind, 3)
            self.assertEqual([e['result'] for e in events], ['parallel'], kind)
            assert_equivalent(a, b)
            self.assertTrue(np.isfinite(a).all())

    def test_worker_failure_falls_back_to_serial_bytes(self):
        freeze = column_parallel.freeze
        flaky = lambda namespace, names: freeze(
            SimpleNamespace(
                geometry_segments=partial(flaky_segments, namespace.geometry_segments), probe=namespace.probe
            ),
            names,
        )
        for kind in ('strato', 'density', 'endpoint'):
            a, _ = self.run_kind(kind, 1)
            with mock.patch.object(column_parallel, 'freeze', flaky):
                b, events = self.run_kind(kind, 3)
            self.assertEqual(
                [(e['result'], e['reason']) for e in events], [('fallback', 'E_TEST_WORKER_FAILURE')], kind
            )
            assert_equivalent(a, b)


def spawn_columns(tmp_path, monkeypatch, *, two_layer=False, drift=False):
    from sosfuse2.atmosphere import density
    from sosfuse2.solve.ray_geometry import normal_state
    from sosfuse2.tests.test_solver_promotion import context_solver

    state = normal_state()
    if two_layer:
        state['troposphere_profile'] = {'status': 'enabled', 'model': 'two_layer'}
    _, ctx, _ = context_solver(tmp_path, state=state)
    old = ctx.b.sh.old
    optics = density.build(state)
    starts = np.array([[0.0, 0.0, old.R + 0.1]])
    mu = np.linspace(-0.2, 1.0, 12)
    dirs = np.column_stack([np.sqrt(1 - mu * mu), np.zeros(len(mu)), mu])
    p = np.broadcast_to(starts, dirs.shape)
    base = old.total_columns(p, dirs, 24)
    serial = density.correct_columns(p, dirs, base, old, optics, 24)
    out = base.copy()
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'gl24')
    monkeypatch.setattr(column_parallel, 'MIN_RAYS', 1)
    monkeypatch.setattr(column_parallel, 'CHUNK', 4)
    monkeypatch.setattr(column_parallel, 'TASK_CHUNKS', 1)
    if drift:
        monkeypatch.setattr(
            column_parallel, 'SOURCE_HASHES', dict(column_parallel.SOURCE_HASHES, **{'rays/shell_segments.py': 'drift'})
        )
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        success = column_parallel.parallel_columns(
            out, 'density', starts, np.array([0, len(dirs)]), dirs, old, optics, count=2
        )
    if not success:
        out[:] = density.correct_columns(p, dirs, out, old, optics, 24)
    events = [json.loads(line) for line in log.getvalue().splitlines() if '"column_parallel"' in line]
    np.testing.assert_allclose(out, serial, rtol=1e-6, atol=0)
    return success, events


def test_column_spawn_default_and_two_layer(tmp_path, monkeypatch):
    for two_layer in (False, True):
        success, events = spawn_columns(tmp_path / str(two_layer), monkeypatch, two_layer=two_layer)
        assert success and [(e['result'], e['reason']) for e in events] == [('parallel', None)]


def test_worker_source_drift_keeps_parent_reason_and_serial_result(tmp_path, monkeypatch):
    success, events = spawn_columns(tmp_path, monkeypatch, drift=True)
    assert not success
    assert [(e['result'], e['reason']) for e in events] == [('fallback', 'E_COLUMN_WORKER_SOURCE_DRIFT')]


if __name__ == '__main__':
    unittest.main(verbosity=2)
