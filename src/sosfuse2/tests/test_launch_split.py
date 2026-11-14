"""Host-only checks of the launch split (device profile, adaptive chunks, budget); no CUDA, no engine.

Run: python -B -m sosfuse2.tests.test_launch_split
"""

from sosfuse2.foundation import paths
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from sosfuse2.backends import device_profile
from sosfuse2.solve.source_solver import Kernel


class FakeXP:
    """Events whose elapsed time is scripted per launch (milliseconds)."""

    def __init__(self, seconds):
        self.seconds = list(seconds)
        self.cuda = SimpleNamespace(
            Event=lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None),
            get_elapsed_time=lambda a, b: 1000.0 * self.seconds.pop(0),
        )


def kernel(profile_env, seconds=()):
    with mock.patch.dict(os.environ, profile_env, clear=False):
        k = Kernel(backend='numpy')
        k.xp = FakeXP(seconds)
        k._device = device_profile.select(None)
        k.launch_log = dict(
            profile=k._device[0],
            reason='test',
            budget_seconds=k._device[1]['launch_budget_seconds'],
            target_seconds=k._device[1]['launch_target_seconds'],
            launches=0,
            gpu_seconds=0.0,
            max_seconds=0.0,
            by_kind={},
        )
    return k


ENV_KEYS = (
    'SOSFUSE2_DEVICE_PROFILE',
    'SOSFUSE2_LAUNCH_CELLS',
    'SOSFUSE2_LAUNCH_BUDGET_SECONDS',
    'SOSFUSE2_LAUNCH_TARGET_SECONDS',
    'SOSFUSE2_LAUNCH_PAUSE_MS',
)


class LaunchSplit(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.pop(k) for k in ENV_KEYS if k in os.environ}

    def tearDown(self):
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(self.saved)

    def test_profile_selection(self):
        self.assertEqual(device_profile.select('NVIDIA GeForce RTX 3080')[0], 'eda')
        self.assertEqual(device_profile.select('NVIDIA GeForce RTX 4090')[0], 'rig')
        self.assertEqual(device_profile.select(None)[0], 'mac')
        with mock.patch.dict(os.environ, {'SOSFUSE2_DEVICE_PROFILE': 'eda'}):
            name, p, _ = device_profile.select('NVIDIA GeForce RTX 4090')
            self.assertEqual((name, p['launch_budget_seconds']), ('eda', 30.0))
        with mock.patch.dict(os.environ, {'SOSFUSE2_DEVICE_PROFILE': 'nope'}):
            with self.assertRaises(ValueError):
                device_profile.select(None)
        with mock.patch.dict(os.environ, {'SOSFUSE2_LAUNCH_CELLS': '408'}):
            self.assertEqual(device_profile.select('NVIDIA GeForce RTX 3080')[1]['launch_cells_fixed'], 408)

    def test_modes(self):
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'eda'})
        self.assertEqual(k.launch_mode(1500), ('adaptive', None))
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'rig'})
        self.assertEqual(k.launch_mode(1500), ('single', 1500))
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'rig', 'SOSFUSE2_LAUNCH_CELLS': '408'})
        self.assertEqual(k.launch_mode(1500), ('fixed', 408))
        self.assertEqual(k.launch_mode(300), ('single', 300))

    def run_chunks(self, k, seconds_of, n=1500, per=10.0):
        cum = np.r_[0, np.cumsum(np.full(n, per))]
        c0, sizes = 0, []
        while c0 < n:
            c1 = k.next_chunk_end('kind', c0, n, None, cum)
            self.assertGreater(c1, c0)
            k.xp.seconds.append(seconds_of(c1 - c0))
            k.timed_launch('kind', lambda *a, **kw: None, (c1 - c0,), (256,), (), 0, c1 - c0, cum, c0, c1)
            sizes.append(c1 - c0)
            c0 = c1
        return sizes

    def test_adaptive_chunks_cover_grid_probe_first_and_bounded_growth(self):
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'eda'})
        sizes = self.run_chunks(k, lambda u: 0.02 * u)
        self.assertEqual(sum(sizes), 1500)
        self.assertEqual(sizes[0], k.device()[1]['launch_cells_initial'] // Kernel.PROBE_DIVISOR)
        for a, b in zip(sizes, sizes[1:-1]):
            self.assertLessEqual(b, max(int(1.5 * a), a + 1))
        self.assertLessEqual(k.launch_log['max_seconds'], k.device()[1]['launch_target_seconds'] * 1.5)

    def test_latency_floor_does_not_collapse_chunks(self):
        # t = 0.8 s + 0.01 s/cell (eda r4: one cell ~0.8 s): chunks must grow past a few cells, not stay at 1.
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'eda'})
        sizes = self.run_chunks(k, lambda u: 0.8 + 0.01 * u)
        self.assertGreater(max(sizes), 20)
        self.assertLess(len(sizes), 150)
        self.assertLessEqual(k.launch_log['max_seconds'], 4.0)

    def test_slow_measurement_halves_before_launch(self):
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'eda'}, seconds=[1.0, 4.5])
        cum = np.r_[0, np.cumsum(np.full(1500, 10.0))]
        c1 = k.next_chunk_end('kind', 0, 1500, None, cum)
        k.timed_launch('kind', lambda *a, **kw: None, (c1,), (256,), (), 0, c1, cum, 0, c1)
        c2 = k.next_chunk_end('kind', c1, 1500, None, cum)
        k.timed_launch('kind', lambda *a, **kw: None, (c2 - c1,), (256,), (), 0, c2 - c1, cum, c1, c2)
        c3 = k.next_chunk_end('kind', c2, 1500, None, cum)
        self.assertLess(c3 - c2, c2 - c1)
        self.assertLessEqual(k.predict_seconds('kind', cum[c3] - cum[c2]), 0.8 * 5.0)

    def test_one_unit_predicted_over_cap_stops_before_launch(self):
        k = kernel(
            {'SOSFUSE2_DEVICE_PROFILE': 'eda', 'SOSFUSE2_LAUNCH_BUDGET_SECONDS': '5'}, seconds=[4.9]
        )  # 5 s: the fan-curve-off budget
        cum = np.r_[0, np.cumsum(np.full(100, 10.0))]
        c1 = k.next_chunk_end('kind', 0, 100, None, cum)
        k.timed_launch('kind', lambda *a, **kw: None, (c1,), (256,), (), 0, c1, cum, 0, c1)
        with self.assertRaises(RuntimeError) as e:
            k.next_chunk_end('kind', c1, 100, None, cum)
        self.assertIn('not launched', str(e.exception))

    def test_first_light_split_rule(self):
        c2 = (
            'GROUND_PHASE_BATCH,LANE_ATTEN,LANE_GROUND,LM_LUT,PHASE1_TRACE3,PHASE2_NOBARRIER,POINT_NOINLINE,'
            'RAY_CONST_SHARED,SH_AUX,SH_AUX_PAIR,SPECIALIZE,TRANSPORT_WAVE_LANES'
        ).split(',')
        eda = device_profile.select('NVIDIA GeForce RTX 3080')
        rig = device_profile.select('NVIDIA GeForce RTX 4090')
        err = device_profile.first_light_split_error
        self.assertIsNone(err(c2, False, *eda[:2]))  # retired names select default split transport
        self.assertIn('E_LAUNCH_BUDGET_CONFIG', err(['EXP_SECTION_CLOCKS'], False, *eda[:2]))
        self.assertIsNone(err(c2 + ['ACC_GLOBAL', 'FL_RAYSPLIT'], False, *eda[:2]))
        self.assertIsNone(err(c2, False, *rig[:2]))  # rig launches whole grids
        for name in ('C3', 'C3_nostrato'):
            import json

            flags = json.loads((paths.CONFIGS / f'{name}.json').read_text())['kernel_flags']
            self.assertIsNone(err(flags, name == 'C3', *eda[:2]))

    def test_eda_never_launches_whole_grid(self):
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'eda', 'SOSFUSE2_LAUNCH_TARGET_SECONDS': '0'})
        with self.assertRaises(RuntimeError) as e:
            k.launch_mode(1500)
        self.assertIn('E_LAUNCH_BUDGET_CONFIG', str(e.exception))
        self.assertEqual(k.launch_mode(1), ('single', 1))
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'eda', 'SOSFUSE2_LAUNCH_CELLS': '408'})
        self.assertEqual(k.launch_mode(1500), ('fixed', 408))
        self.assertEqual(k.launch_mode(300), ('fixed', 300))

    def test_fixed_step_is_only_an_upper_bound_on_eda(self):
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'eda', 'SOSFUSE2_LAUNCH_CELLS': '408'})
        cum = np.r_[0, np.cumsum(np.full(1500, 10.0))]
        c0, sizes = 0, []
        while c0 < 1500:
            c1 = k.next_chunk_end('kind', c0, 1500, 408, cum)
            k.xp.seconds.append(0.79 + 0.0134 * (c1 - c0))  # r4-like warm cost
            k.timed_launch('kind', lambda *a, **kw: None, (c1 - c0,), (256,), (), 0, c1 - c0, cum, c0, c1)
            sizes.append(c1 - c0)
            c0 = c1
        self.assertEqual(sizes[0], k.device()[1]['launch_cells_initial'] // Kernel.PROBE_DIVISOR)
        self.assertLessEqual(max(sizes), 408)
        self.assertLessEqual(k.launch_log['max_seconds'], 4.0)

    def test_budget_raises_after_the_launch(self):
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'eda', 'SOSFUSE2_LAUNCH_BUDGET_SECONDS': '5'}, seconds=[5.2])
        calls = []
        with self.assertRaises(RuntimeError) as e:
            k.timed_launch('kind', lambda *a, **kw: calls.append(a), (1,), (256,), (), 0, 1)
        self.assertIn('E_LAUNCH_BUDGET', str(e.exception))
        self.assertEqual(len(calls), 1)

    def test_no_budget_launches_without_events(self):
        k = kernel({'SOSFUSE2_DEVICE_PROFILE': 'mac'}, seconds=[])
        calls = []
        self.assertEqual(k.timed_launch('kind', lambda *a, **kw: calls.append(a), (1,), (256,), (), 0, 1), 0.0)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
