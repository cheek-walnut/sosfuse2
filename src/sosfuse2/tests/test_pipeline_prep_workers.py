"""Per-state preparation pool sizes exported by pipeline.py (prep_workers). CPU only, no GPU, no real state run:
the controller's subprocess.Popen is replaced and only the child environment is checked."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sosfuse2 import pipeline as pl  # noqa: E402
from sosfuse2.host_limits import default_workers  # noqa: E402

NAMES = pl.PREP_WORKER_ENV


class PrepWorkers(unittest.TestCase):
    def test_cores_shared_by_states(self):
        per, explicit, updates = pl.prep_workers(4, 1, 12, {})
        self.assertEqual((per, explicit), (3, []))
        self.assertEqual(updates, {k: '3' for k in NAMES})

    def test_explicit_env_kept(self):
        env = {'SOSFUSE2_SUN_WORKERS': '7', 'SOSFUSE2_GROUND_WORKERS': '  '}  # blank = unset, as the modules read it
        per, explicit, updates = pl.prep_workers(4, 1, 12, env)
        self.assertEqual(explicit, ['SOSFUSE2_SUN_WORKERS'])
        self.assertNotIn('SOSFUSE2_SUN_WORKERS', updates)
        self.assertEqual(updates['SOSFUSE2_GROUND_WORKERS'], '3')

    def test_minimum_one(self):
        per, _, updates = pl.prep_workers(4, 1, 2, {})
        self.assertEqual(per, 1)
        self.assertEqual(set(updates.values()), {'1'})

    def test_one_state_keeps_child_defaults(self):
        # --workers 1: every pool gets all usable CPUs, the value each child computes itself when unset
        # (sun/ground/geometry_parallel.workers() -> default_workers()); the column builds stay >= --threads.
        cores = default_workers()
        _, _, updates = pl.prep_workers(1, 1, cores, {})
        self.assertEqual(updates, {k: str(cores) for k in NAMES})
        _, _, updates = pl.prep_workers(1, 3, 2, {})
        self.assertEqual((updates['SOSFUSE2_COLUMN_WORKERS'], updates['SOSFUSE2_SUN_WORKERS']), ('3', '2'))

    def test_run_queue_child_env(self):
        tmp = Path(tempfile.mkdtemp(prefix='prep-workers-'))
        self.addCleanup(lambda: __import__('shutil').rmtree(tmp, ignore_errors=True))
        (tmp / 'states.json').write_text(
            json.dumps(dict(states=[dict(id='S0', runnable=True), dict(id='S1', runnable=True)]))
        )
        (tmp / 'runner.py').write_text('raise SystemExit(0)\n')
        seen = []

        class FakeChild:
            count = 0

            def __init__(self, cmd, env=None, **_):
                FakeChild.count += 1
                self.pid = 900000 + FakeChild.count
                seen.append(dict(id=cmd[cmd.index('--id') + 1], env={k: env.get(k) for k in NAMES}))

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        a = pl.parser().parse_args(
            [
                '--states',
                str(tmp / 'states.json'),
                '--ids',
                'S0,S1',
                '--work-root',
                str(tmp / 'work'),
                '--workers',
                '2',
                '--backend',
                'numpy',
                '--reduced',
                '--runner',
                str(tmp / 'runner.py'),
                '--gpu-lock',
                str(tmp / 'gpu.lock'),
            ]
        )
        clean = {k: v for k, v in os.environ.items() if k not in NAMES}
        with (
            mock.patch.dict(os.environ, dict(clean, SOSFUSE2_GEOMETRY_WORKERS='5'), clear=True),
            mock.patch.object(pl, 'default_workers', return_value=12),
            mock.patch.object(pl, 'cpu_budget', return_value=12),
            mock.patch.object(pl.subprocess, 'Popen', FakeChild),
            mock.patch.object(pl, 'linux_sample', return_value=None),
        ):
            code = pl.run_queue(pl.validate(a))
        self.assertEqual(code, 0)
        expected = dict(
            SOSFUSE2_COLUMN_WORKERS='6',
            SOSFUSE2_SUN_WORKERS='6',
            SOSFUSE2_GROUND_WORKERS='6',
            SOSFUSE2_GEOMETRY_WORKERS='5',
        )
        self.assertEqual(seen, [dict(id='S0', env=expected), dict(id='S1', env=expected)])
        prep = json.loads((tmp / 'work' / 'pipeline_summary.json').read_text())['prep_workers']
        self.assertEqual(
            prep,
            dict(host_cores=12, states_at_once=2, per_state=6, explicit=['SOSFUSE2_GEOMETRY_WORKERS'], env=expected),
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)


def test_column_keeps_the_cpu_budget_cap():
    # Mac cpu_budget() = 4: the column pool stays at the old max(--threads, budget // workers) while the other pools
    # take host_cores // workers (pool children are outside the RSS guard).
    _, _, updates = pl.prep_workers(1, 1, 10, {}, column_budget=4)
    assert updates['SOSFUSE2_COLUMN_WORKERS'] == '4' and updates['SOSFUSE2_SUN_WORKERS'] == '10'
    _, _, updates = pl.prep_workers(2, 3, 10, {}, column_budget=4)
    assert updates['SOSFUSE2_COLUMN_WORKERS'] == '3' and updates['SOSFUSE2_GROUND_WORKERS'] == '5'
