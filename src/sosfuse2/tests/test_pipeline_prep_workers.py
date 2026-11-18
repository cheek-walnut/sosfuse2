"""CPU checks for pipeline worker budgets and summary records using stub runners."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sosfuse2.jobs import pipeline as pl  # noqa: E402
from sosfuse2.foundation.host_limits import default_workers  # noqa: E402

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
    # macOS cpu_budget() = 4: the column pool stays at the old max(--threads, budget // workers) while the other pools
    # take host_cores // workers (pool children are outside the RSS guard).
    _, _, updates = pl.prep_workers(1, 1, 10, {}, column_budget=4)
    assert updates['SOSFUSE2_COLUMN_WORKERS'] == '4' and updates['SOSFUSE2_SUN_WORKERS'] == '10'
    _, _, updates = pl.prep_workers(2, 3, 10, {}, column_budget=4)
    assert updates['SOSFUSE2_COLUMN_WORKERS'] == '3' and updates['SOSFUSE2_GROUND_WORKERS'] == '5'


def test_pipeline_reports_ksplit_decision_per_state(tmp_path, capsys):
    states = tmp_path / 'states.json'
    states.write_text(json.dumps(dict(states=[dict(id=i, runnable=True) for i in ('off', 'fallback')])))
    runner = tmp_path / 'runner.py'
    runner.write_text(
        'import sys\n'
        'from sosfuse2.jobs.run_state import phase\n'
        'state = sys.argv[sys.argv.index("--id") + 1]\n'
        'decision = dict(requested=0, effective=0, reason="sh_order_gt_16")\n'
        'if state == "fallback":\n'
        '    decision.update(requested=3, peak=3_000_000_000, budget=2_000_000_000,\n'
        '                    reason="state_minimum_exceeds_solo_budget")\n'
        '    phase(state, "KSPLIT_FALLBACK", **decision)\n'
        'phase(state, "ksplit", decision=decision)\n'
    )
    args = ['--states', str(states), '--ids', 'off,fallback', '--work-root', str(tmp_path)]
    a = pl.parser().parse_args(args + ['--workers', '1', '--backend', 'numpy', '--reduced', '--runner', str(runner)])
    assert pl.run_queue(pl.validate(a)) == 0
    off, fallback = json.loads((tmp_path / 'pipeline_summary.json').read_text())['states']
    assert off['ksplit'] == dict(requested=0, effective=0, reason='sh_order_gt_16')
    assert 'ksplit_fallback' not in off
    assert fallback['ksplit']['reason'] == 'state_minimum_exceeds_solo_budget'
    assert fallback['ksplit']['requested'] == 3 and fallback['ksplit']['effective'] == 0
    assert len(fallback['ksplit_fallback']) == 1
    warnings = [line for line in capsys.readouterr().err.splitlines() if line.startswith('WARNING KSPLIT')]
    assert len(warnings) == 1 and warnings[0].startswith('WARNING KSPLIT fallback in state fallback: requested 3 -> 0,')
