"""CPU-only real selftest geometry: spawn, committed-prefix recovery, failure."""

from collections import OrderedDict
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sosfuse2.atmosphere import components, density, stratosphere
from sosfuse2.foundation.top import resolve
from sosfuse2.jobs.run_state import register_geometry
from sosfuse2.rays import column_table
from sosfuse2.solve import engine, ray_geometry as gm, ray_geometry_parallel as gp
from sosfuse2.checks.numerical_checks import assert_equivalent


def fail_after_prefix(i):
    if i == 5:
        raise RuntimeError('E_TEST_GEOMETRY_WORKER_FAILURE: cell 5')
    return gp._run(i)


def files(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}


class GeometryParallelTests(unittest.TestCase):
    def test_lazy_imports(self):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        subprocess.run(
            [sys.executable, '-S', '-c', "import sys, sosfuse2.foundation.host_limits; assert 'numpy' not in sys.modules"],
            env=env,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                '-c',
                "import sosfuse2.solve.ray_geometry_parallel as p; from sosfuse2.solve.ray_geometry import PackedCache; "
                "assert PackedCache is p.PackedCache",
            ],
            env=env,
            check=True,
        )

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.s, cls.scope, _ = gm.create(cls.root / 'prepare', 'selftest', [550.0], 'numpy')
        cls.serial, cls.serial_events = cls.build('serial', 1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def resident(cls, name):
        with mock.patch.object(gm.TwoLayerGeometry, '_host_test', True):
            return gm.TwoLayerGeometry(cls.s, cls.scope, cls.root / name)

    @classmethod
    def build(cls, name, count, g=None):
        g = g or cls.resident(name)
        log = io.StringIO()
        with mock.patch.dict(os.environ, {gp.ENV: str(count)}), contextlib.redirect_stdout(log):
            g.build_packed()
        events = [json.loads(line) for line in log.getvalue().splitlines() if '"geometry_parallel"' in line]
        return g, events

    def assert_same(self, g):
        a, _ = gm.PackedCache(self.serial.cache / 'packed').open()
        b, _ = gm.PackedCache(g.cache / 'packed').open()
        self.assertEqual(set(a), set(b))
        for key in a:
            assert_equivalent(a[key], b[key])

    def test_parallel_and_resume_bytes(self):
        self.assertEqual(self.serial_events, [])
        for resume in (False, True):
            g = self.resident('resume' if resume else 'parallel')
            if resume:

                def interrupted(i):
                    if i == 5:
                        raise RuntimeError('interrupted build')
                    return g.cached_cell(i)

                with self.assertRaisesRegex(RuntimeError, 'interrupted build'):
                    gp.SerialPackedCache(g.cache / 'packed').build(self.s.geo.ns, interrupted)
                with (g.cache / 'packed' / 'dirs.bin').open('ab') as f:
                    f.write(b'uncommitted trailing bytes')
            g, events = self.build('', 3, g)
            self.assertEqual([e['result'] for e in events], ['parallel'])
            self.assertEqual(events[0]['completed'], self.s.geo.ns - (5 if resume else 0))
            self.assert_same(g)
            expected = None if resume else self.s.kernel.height  # accepted TwoLayer resume behavior
            self.assertEqual(json.loads((g.cache / 'source_height.json').read_text()), dict(height=expected))
            _, again = self.build('', 3, g)
            self.assertEqual(again, [])  # completed caches never spawn workers

    def test_worker_failure_preserves_reason_and_bytes(self):
        with mock.patch.object(gp, '_run', fail_after_prefix):
            g, events = self.build('failure', 3)
        self.assertEqual([e['result'] for e in events], ['fallback'])
        self.assertEqual(events[0]['completed'], 5)
        self.assertIn('E_TEST_GEOMETRY_WORKER_FAILURE: cell 5', events[0]['reason'])
        self.assert_same(g)

    def test_unsupported_callback_reports_serial_fallback(self):
        g = self.resident('callback')
        log = io.StringIO()
        with mock.patch.dict(os.environ, {gp.ENV: '2'}), contextlib.redirect_stdout(log):
            g.packed().build(self.s.geo.ns, lambda i: g.cached_cell(i))
        events = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual([(e['phase'], e['result']) for e in events], [('geometry_parallel', 'fallback')])
        self.assertEqual(events[0]['reason'], 'callback is not ResidentGeometry.cached_cell')
        self.assert_same(g)

    def test_mixed_legacy_statistics_and_bytes(self):
        g = self.resident('legacy')
        for i in range(2):
            arrays, meta = g.cached_cell(i)
            folder = g.legacy / f'cell_{i:05d}'
            folder.mkdir(parents=True)
            for k, v in arrays.items():
                gm.np.save(folder / (k + '.npy'), v)
            (folder / 'complete.json').write_text(json.dumps(meta))
        g.stats.update(cells=0, rays=0, samples=0)
        g, events = self.build('', 3, g)
        self.assertEqual([e['result'] for e in events], ['parallel'])
        self.assertEqual(g.stats['legacy_cells_read'], 2)
        self.assertEqual(json.loads((g.cache / 'source_height.json').read_text()), dict(height=None))
        a, _ = gm.PackedCache(self.serial.cache / 'packed').open()
        b, _ = gm.PackedCache(g.cache / 'packed').open()
        self.assertEqual(set(a), set(b))
        for key in a:
            assert_equivalent(a[key], b[key])


def init_small_column_tables(*args):
    column_table.NA = column_table.NX = 32
    gp._init(*args)


def test_extended_direct_callback_reuses_column_tables(tmp_path, monkeypatch, capsys):
    for env in ('SOSFUSE2_COLUMN_CACHE', 'SOSFUSE2_COLUMN_TABLE_CACHE'):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'table-v1')
    monkeypatch.setattr(column_table, 'NA', 32)
    monkeypatch.setattr(column_table, 'NX', 32)
    monkeypatch.setattr(column_table, '_MEMO', OrderedDict())

    domain = resolve(dict(top_km=120, upper_air='vacuum'))
    state = dict(gm.normal_state(), density=dict(density.DEFAULTS))
    with components._s1_reference():
        solver, scope, _ = engine.create(
            state,
            tmp_path / 'solver',
            'full',
            1,
            kernel_factory=lambda threads, height, g: SimpleNamespace(xp=np, height=height, domain=domain),
            grid=lambda *args: (np.array([0., 4., 120.]), np.deg2rad([0., 80., 90., 110.]), 2, 8, 4),
            sun_geometry=lambda *args: None,
            prepare_ground=False,
            domain=domain,
        )
    solver.density_optics = density.build(state, domain)
    profile = stratosphere.profile_config(state)
    profile.update(end_km=domain.lower_end, domain=domain.identity())
    solver.strato = dict(profile=profile)
    serial = register_geometry(solver, scope, tmp_path / 'serial', layers=2)
    parallel = register_geometry(solver, scope, tmp_path / 'parallel', layers=2)

    monkeypatch.setenv(gp.ENV, '1')
    with mock.patch.object(column_table, 'build_table', wraps=column_table.build_table) as build:
        serial.build_packed()
    # Two cumulative tables (ground and tangent) for each of the nine profiles.
    assert build.call_count == 18

    monkeypatch.setenv(gp.ENV, '2')
    with mock.patch.object(gp, '_init', init_small_column_tables):
        parallel.build_packed()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if '"geometry_parallel"' in line]
    assert [(e['result'], e['workers'], e['completed']) for e in events] == [('parallel', 2, solver.geo.ns)]
    expected, expected_records = serial.packed().open()
    actual, records = parallel.packed().open()
    assert records == expected_records
    assert actual.keys() == expected.keys()
    for key in expected:
        assert_equivalent(expected[key], actual[key])
    for i in range(solver.geo.ns):
        assert parallel._joint_cells[i].keys() == serial._joint_cells[i].keys()
        for key, array in serial._joint_cells[i].items():
            assert_equivalent(array, parallel._joint_cells[i][key])
