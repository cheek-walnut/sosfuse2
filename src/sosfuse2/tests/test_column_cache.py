"""Reduced queue runs of the column cache; shares PERSISTENT_TEST_BACKEND/DEADLINE knobs."""

from sosfuse2.foundation import paths
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from sosfuse2.tests import test_pipeline_persistent as base
import numpy as np
from sosfuse2.checks.numerical_checks import assert_equivalent

# Tiny changes around the known original; convergence still needs a measurement.
CASES = ((0.02, 0.003, 3.0), (0.0201, 0.004, 3.0), (0.02, 0.003, 3.01))  # aod, tau, sigma


class ColumnCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = base.testkit.scratch('column-tests')
        if os.environ.get('SOSFUSE2_KEEP_TEST_OUTPUT') != '1':  # runs even if setUpClass fails later
            cls.addClassCleanup(shutil.rmtree, cls.root, ignore_errors=True)
        states = []
        for i, (aod, tau, sigma) in enumerate(CASES):
            state = copy.deepcopy(base.build()['states'][1])
            state['id'] = 'C' + str(i)
            state['parameters'].update(aod550=aod, stratosphere_tau550=tau)
            state['stratosphere']['sigma_km'] = sigma
            states.append(state)
        cls.states = cls.root / 'states.json'
        cls.states.write_text(json.dumps(dict(states=states)))
        for i in range(3):
            if i < 2:
                cls.run_case(i, False)
            cls.run_case(i, True)
        print('COLUMN_TEST_OUTPUT', cls.root, flush=True)

    @classmethod
    def run_case(cls, i, cached):
        name = f'C{i}-' + ('cache' if cached else 'plain')
        out = cls.root / name
        env = base.child_environment()
        env.pop('SOSFUSE2_COLUMN_CACHE', None)
        cmd = [
            *base.paths.module_command('jobs.pipeline'),
            '--states',
            str(cls.states),
            '--ids',
            f'C{i}',
            '--work-root',
            str(out),
            '--workers',
            '1',
            '--threads',
            '1',
            '--backend',
            base.BACKEND,
            '--reduced',
            '--geometry-cache',
            str(cls.root / 'geometry'),
            *base.settings_args(cls.root),
        ]
        if cached:
            cmd += ['--column-cache', str(cls.root / 'columns')]
        with (cls.root / (name + '.log')).open('w') as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=float(os.environ.get('PERSISTENT_TEST_DEADLINE', '120')))
            except BaseException:
                proc.terminate()  # Let the controller terminate its owned child groups.
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
            if code:
                raise AssertionError(f'E_COLUMN_TEST_QUEUE: {name} exit={code}')

    def events(self, i):
        log = sorted((self.root / f'C{i}-cache/logs').glob(f'*/C{i}.log'))[-1]
        return [e for e in base.PersistentTests.events(log) if e['phase'] == 'column_cache' and e['kind'] == 'strato']

    def parity(self, i):
        for filename, key in ((f'out/C{i}/table.npz', 'radiance'), (f'tmp/C{i}/source.npz', 'source')):
            with np.load(self.root / f'C{i}-plain' / filename) as a, np.load(self.root / f'C{i}-cache' / filename) as b:
                assert_equivalent(a[key], b[key], rtol=1e-5)

    def test_1_miss_parity(self):
        self.parity(0)
        self.assertTrue(any(e['result'] == 'miss' for e in self.events(0)))

    def test_2_changed_tau_aod_hit_parity(self):
        self.parity(1)
        self.assertTrue(any(e['result'] == 'hit' for e in self.events(1)))
        summary = json.loads((self.root / 'C1-cache/pipeline_summary.json').read_text())
        self.assertGreater(summary['column_cache']['hit'], 0)

    def test_3_changed_sigma_miss(self):
        events = self.events(2)
        self.assertTrue(any(e['result'] == 'miss' for e in events))
        self.assertTrue({e['key'] for e in events}.isdisjoint(e['key'] for e in self.events(0)))


if __name__ == '__main__':
    unittest.main(verbosity=2, failfast=True)
