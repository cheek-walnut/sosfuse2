"""Real accepted row arithmetic, spawn failures, and the unchanged NPZ reader."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sosfuse2.atmosphere import density
from sosfuse2.solve import ray_geometry
from sosfuse2.precompute import host_cache, sun_parallel as sp

try:
    from sosfuse2.checks.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


def fail_row(task):
    if task[0] == 2:
        raise RuntimeError('E_TEST_SUN_WORKER')
    return sp._run(task)


class SunParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with mock.patch.dict(os.environ, {sp.ENV: '1'}):
            _, scope, _ = ray_geometry.create(Path(cls.tmp.name) / 'prepare', 'selftest', [550.0], 'numpy')
        cls.old = scope.b.sh.old
        cls.h = np.array([0.0, 0.1, 13.0, 40.0, 80.0])
        cls.x = np.linspace(0, 1, 21)
        cls.optics = density.build(dict(density=dict(density_profile='usstd76', ozone_profile='usstd76')))
        cls.optics['old'] = cls.old

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_rows(self, count, base=None, optics=None):
        log = io.StringIO()
        with mock.patch.dict(os.environ, {sp.ENV: str(count)}), contextlib.redirect_stdout(log):
            value = sp.rows(self.old, self.h, self.x, self.old.R, 32, base, optics)
        events = [json.loads(line) for line in log.getvalue().splitlines()]
        return value, events

    def test_rows_and_worker_fallback(self):
        base, _ = self.run_rows(1)
        for initial, optics in ((None, None), (base, self.optics)):
            expected, events = self.run_rows(1, initial, optics)
            self.assertEqual(events, [])
            actual, events = self.run_rows(2, initial, optics)
            self.assertEqual(events[0]['result'], 'parallel')
            assert_equivalent(actual, expected)
            with mock.patch.object(sp, '_run', fail_row):
                actual, events = self.run_rows(2, initial, optics)
            self.assertEqual((events[0]['result'], events[0]['reason']), ('fallback', 'E_TEST_SUN_WORKER'))
            assert_equivalent(actual, expected)

    def test_full_table_unchanged_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.dict(os.environ, {sp.ENV: '1'}):
                expected = host_cache.sun_geometry(self.old, 1, root / 'serial', 0.1)
            with mock.patch.dict(os.environ, {sp.ENV: '2'}):
                actual = host_cache.sun_geometry(self.old, 1, root / 'parallel', 0.1)
                self.assertGreater(actual.build_seconds, 0)
                with mock.patch.object(sp, 'rows', side_effect=AssertionError('cache hit')):
                    hit = host_cache.sun_geometry(self.old, 1, root / 'parallel', 0.1)
                    self.assertEqual(hit.build_seconds, 0)
            for name in ('h', 'x', 'col'):
                assert_equivalent(getattr(expected, name), getattr(actual, name))
            a = next((root / 'serial').rglob('sun_columns.npz'))
            b = next((root / 'parallel').rglob('sun_columns.npz'))
            with np.load(a) as x, np.load(b) as y:
                self.assertEqual(set(x.files), set(y.files))
                for key in x.files:
                    assert_equivalent(x[key], y[key])

    def test_worker_setting(self):
        with mock.patch.dict(os.environ, {sp.ENV: ''}), mock.patch.object(sp, 'default_workers', return_value=3):
            self.assertEqual(sp.workers(), 3)
        for value in ('0', '-1', 'wrong'):
            with mock.patch.dict(os.environ, {sp.ENV: value}):
                with self.assertRaisesRegex(ValueError, 'E_SUN_WORKERS'):
                    sp.workers()

    def test_corrected_cache_entry(self):
        base, _ = self.run_rows(1)
        sun = SimpleNamespace(h=self.h, x=self.x, col=base)
        s = SimpleNamespace(geo=SimpleNamespace(sun=sun))
        expected, _ = self.run_rows(1, base, self.optics)
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(host_cache, 'context', return_value=(Path(tmp), {})),
            mock.patch.dict(os.environ, {sp.ENV: '2'}),
        ):
            actual = host_cache.corrected_sun(s, self.optics)
            assert_equivalent(actual, expected)
            assert_equivalent(sun.col, base)
            with mock.patch.object(sp, 'rows', side_effect=AssertionError('cache hit')):
                assert_equivalent(host_cache.corrected_sun(s, self.optics), expected)
