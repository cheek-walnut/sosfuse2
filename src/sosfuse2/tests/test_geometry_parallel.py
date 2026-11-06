"""CPU-only real selftest geometry: spawn, committed-prefix recovery, failure."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from sosfuse2 import geometry as gm, geometry_parallel as gp
from sosfuse2.numerical_checks import assert_equivalent


def fail_after_prefix(i):
    if i == 5:
        raise RuntimeError('E_TEST_GEOMETRY_WORKER_FAILURE: cell 5')
    return gp._run(i)


def files(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}


class GeometryParallelTests(unittest.TestCase):
    def test_lazy_imports(self):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        subprocess.run([sys.executable, '-S', '-c',
                        "import sys, sosfuse2.host_limits; assert 'numpy' not in sys.modules"], env=env, check=True)
        subprocess.run([sys.executable, '-c',
                        "import sosfuse2.geometry_parallel as p; from sosfuse2.geometry import PackedCache; "
                        "assert PackedCache is p.PackedCache"], env=env, check=True)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.s, cls.scope, _ = gm.create(cls.root/'prepare', 'selftest', [550.], 'numpy')
        cls.serial, cls.serial_events = cls.build('serial', 1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def resident(cls, name):
        with mock.patch.object(gm.TwoLayerGeometry, '_host_test', True):
            return gm.TwoLayerGeometry(cls.s, cls.scope, cls.root/name)

    @classmethod
    def build(cls, name, count, g=None):
        g = g or cls.resident(name); log = io.StringIO()
        with mock.patch.dict(os.environ, {gp.ENV: str(count)}), contextlib.redirect_stdout(log):
            g.build_packed()
        events = [json.loads(line) for line in log.getvalue().splitlines() if '"geometry_parallel"' in line]
        return g, events

    def assert_same(self, g):
        a, _ = gm.PackedCache(self.serial.cache/'packed').open()
        b, _ = gm.PackedCache(g.cache/'packed').open()
        self.assertEqual(set(a), set(b))
        for key in a: assert_equivalent(a[key], b[key])

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
                    gp.SerialPackedCache(g.cache/'packed').build(self.s.geo.ns, interrupted)
                with (g.cache/'packed'/'dirs.bin').open('ab') as f:
                    f.write(b'uncommitted trailing bytes')
            g, events = self.build('', 3, g)
            self.assertEqual([e['result'] for e in events], ['parallel'])
            self.assertEqual(events[0]['completed'], self.s.geo.ns-(5 if resume else 0))
            self.assert_same(g)
            expected = None if resume else self.s.kernel.height  # accepted TwoLayer resume behavior
            self.assertEqual(json.loads((g.cache/'source_height.json').read_text()), dict(height=expected))
            _, again = self.build('', 3, g)
            self.assertEqual(again, [])  # completed caches never spawn workers

    def test_worker_failure_preserves_reason_and_bytes(self):
        with mock.patch.object(gp, '_run', fail_after_prefix):
            g, events = self.build('failure', 3)
        self.assertEqual([e['result'] for e in events], ['fallback'])
        self.assertEqual(events[0]['completed'], 5)
        self.assertIn('E_TEST_GEOMETRY_WORKER_FAILURE: cell 5', events[0]['reason'])
        self.assert_same(g)

    def test_mixed_legacy_statistics_and_bytes(self):
        g = self.resident('legacy')
        for i in range(2):
            arrays, meta = g.cached_cell(i); folder = g.legacy/f'cell_{i:05d}'; folder.mkdir(parents=True)
            for k, v in arrays.items():
                gm.np.save(folder/(k+'.npy'), v)
            (folder/'complete.json').write_text(json.dumps(meta))
        g.stats.update(cells=0, rays=0, samples=0)
        g, events = self.build('', 3, g)
        self.assertEqual([e['result'] for e in events], ['parallel'])
        self.assertEqual(g.stats['legacy_cells_read'], 2)
        self.assertEqual(json.loads((g.cache/'source_height.json').read_text()), dict(height=None))
        a, _ = gm.PackedCache(self.serial.cache/'packed').open()
        b, _ = gm.PackedCache(g.cache/'packed').open()
        self.assertEqual(set(a), set(b))
        for key in a: assert_equivalent(a[key], b[key])
