"""Five integration cases; PERSISTENT_TEST_BACKEND and PERSISTENT_TEST_DEADLINE select execution.
Run with a NumPy/Pillow/zstandard Python: python -B -m sosfuse2.test_pipeline_persistent
"""
import copy
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from .numerical_checks import assert_equivalent
from sosfuse2 import paths, testing
from sosfuse2.strato_states import build
from sosfuse2.pipeline import cpu_budget
TAUS = (.003, .004, .005, .006)  # Commander may tune reduced-state convergence here.
BACKEND = os.environ.get('PERSISTENT_TEST_BACKEND', 'numpy')


def child_environment():
    if BACKEND not in ('numpy', 'cupy'): raise ValueError('E_TEST_BACKEND')
    env = paths.child_env()   # child pipelines import sosfuse2 in-tree too
    if BACKEND == 'cupy':
        config = json.loads((HERE/'configs/C3.json').read_text())
        env.update(SOSFUSE2_KERNEL_FLAGS=','.join(config['kernel_flags']), SOSFUSE2_BLOCK_SIZE=str(config['block_size']))
    return env


def interleave_env():
    """On the development GPU host (RTX 3080 10 GB), KSPLIT reserves 5.2 GB per state (1c870ff..b974ea0), so a second state waits in
    vram_wait and the service runs one state at a time. The interleaving case tests the scheduler, so it runs with KSPLIT off
    there; larger cards keep the default KSPLIT and still check interleaving with it."""
    from sosfuse2 import device_profile
    device = device_profile.smi_device_name() if BACKEND == 'cupy' else None
    if device is None and BACKEND == 'cupy':
        import cupy
        device = device_profile.cuda_device_name(cupy)
    name = device_profile.select(device)[0]
    return {'SOSFUSE2_KSPLIT': '0'} if name == 'eda' else {}


def settings_args(root):
    """C3 kernels are specialized for 15 wavelengths; the reduced default has one."""
    if BACKEND != 'cupy': return []
    path = Path(root)/'settings_fw15.json'
    if not path.exists():
        path.write_text(json.dumps(dict(wavelengths=15, wavelengths_nm=(360+(np.arange(15)+.5)*470/15).tolist())))
    return ['--settings-json', str(path)]


def numeric(value):
    if isinstance(value, dict):
        return {k: numeric(v) for k, v in value.items() if 'seconds' not in k}
    if isinstance(value, list): return [numeric(v) for v in value]
    return value


class PersistentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = testing.scratch('persistent-tests')
        if os.environ.get('SOSFUSE2_KEEP_TEST_OUTPUT') != '1':   # runs even if setUpClass fails later
            cls.addClassCleanup(shutil.rmtree, cls.root, ignore_errors=True)
        cls.states = cls.root / 'states.json'
        states = []
        for i in range(4):
            s = copy.deepcopy(build()['states'][1]); s['id'] = 'P'+str(i)
            s['parameters']['stratosphere_tau550'] = TAUS[i]
            states.append(s)
        cls.states.write_text(json.dumps(dict(states=states)))
        cls.run_queue('baseline', 4, persistent=False)
        cls.run_queue('pair', 2)

    @classmethod
    def command(cls, label, count, persistent=True):
        return [*paths.module_command('pipeline'), '--states', str(cls.states),
            '--ids', ','.join('P'+str(i) for i in range(count)), '--work-root', str(cls.root/label),
            # never more workers than the host's CPU budget: a 2-core CI runner refuses 4 (E_PIPELINE_CPU_BUDGET)
            '--workers', str(min(count, cpu_budget())), '--threads', '1', '--backend', BACKEND, '--reduced',
            '--geometry-cache', str(cls.root/'geometry'), '--gpu-lock', str(cls.root/'gpu.lock'),
            *settings_args(cls.root), *(['--persistent', '--gpu-active', '2'] if persistent else [])]

    @classmethod
    def run_queue(cls, label, count, persistent=True, extra=None, kill=False):
        env = dict(child_environment(), **(extra or {}))
        start = time.monotonic()
        with (cls.root/(label+'.log')).open('w') as log:
            proc = subprocess.Popen(cls.command(label, count, persistent), env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if kill:
                    while time.monotonic()-start < 50:
                        logs = sorted((cls.root/label/'logs').glob('*/GPU_SERVICE.log'))
                        events = cls.events(logs[-1]) if logs else []
                        hit = next((e for e in events if e['phase']=='gpu_piece_start'), None)
                        if hit:
                            os.kill(hit['pid'], signal.SIGKILL)
                            break
                        if proc.poll() is not None: raise AssertionError('service ended before death injection')
                        time.sleep(.05)
                    else: raise AssertionError('no GPU work before death injection deadline')
                code = proc.wait(timeout=max(.01, float(os.environ.get('PERSISTENT_TEST_DEADLINE','60'))-(time.monotonic()-start)))
            except BaseException:
                # Controller TERM cleans its independently grouped children/service.
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                raise
        path = cls.root/label/'pipeline_summary.json'
        if not path.is_file():
            # Keep the reason: the scratch root is removed by the class cleanup, so report the exit code and the log tail.
            tail = (cls.root/(label+'.log')).read_text(errors='replace').splitlines()[-60:]
            raise AssertionError(f'{label}: queue exited with code {code} and wrote no pipeline_summary.json; log tail:\n'
                                 + '\n'.join(tail))
        summary = json.loads(path.read_text())
        if not extra and not kill:
            assert code == 0, (label, summary, str(cls.root))
        return code, summary

    @staticmethod
    def events(path):
        result = []
        for line in path.read_text().splitlines():
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(event, dict) and event.get('event') == 'PHASE': result.append(event)
        return result

    def arrays(self, label, i):
        root = self.root/label
        with np.load(root/'out'/f'P{i}'/'table.npz') as table, np.load(root/'tmp'/f'P{i}'/'source.npz') as source:
            return table['radiance'], source['source']

    def equal_state(self, label, i, base='baseline'):
        for actual, expected in zip(self.arrays(label, i), self.arrays(base, i)):
            self.assertEqual((actual.shape, actual.dtype), (expected.shape, expected.dtype))
            assert_equivalent(actual, expected, rtol=1e-5)
        actual = json.loads((self.root/label/'out'/f'P{i}'/'meta.json').read_text())
        expected = json.loads((self.root/base/'out'/f'P{i}'/'meta.json').read_text())
        self.assertEqual(numeric(actual['source_solve']), numeric(expected['source_solve']))

    def test_1_parity(self):
        for i in range(2): self.equal_state('pair', i)

    def test_2_no_state_mixing(self):
        self.assertNotEqual(self.arrays('baseline', 0)[0].tobytes(), self.arrays('baseline', 1)[0].tobytes())
        for i in range(2):
            assert_equivalent(self.arrays('pair', i)[0], self.arrays('baseline', i)[0], rtol=1e-5)
            self.assertNotEqual(self.arrays('pair', i)[0].tobytes(), self.arrays('baseline', 1-i)[0].tobytes())

    def test_3_prepare_failure(self):
        code, summary = self.run_queue('prepare-failure', 2, extra={'SOS_PERSISTENT_TEST_PREP_FAIL':'P0'})
        self.assertNotEqual(code, 0)
        rows = {r['id']:r for r in summary['states']}
        self.assertEqual(rows['P0']['status'], 'failed')
        self.assertTrue(rows['P0']['reason'].startswith('E_'))
        self.assertEqual(rows['P1']['status'], 'succeeded')

    def test_4_service_death(self):
        code, summary = self.run_queue('death', 4, kill=True,
            extra={'SOS_PERSISTENT_TEST_OBSERVER_DELAY':'0.1'})
        self.assertNotEqual(code, 0)
        self.assertTrue(all(r['status']=='failed' and r['reason'].startswith('E_') for r in summary['states']))

    def test_5_interleaving(self):
        kernel = interleave_env()
        # Bytes are compared with a sequential run of the same kernel: KSPLIT=0 and the default KSPLIT=3 differ by ULPs
        # (max rel 7.6e-7).
        base = 'baseline'
        if kernel:
            base = 'baseline_ksplit0'; code, summary = self.run_queue(base, 4, persistent=False, extra=kernel)
            self.assertEqual(code, 0, summary)
        code, summary = self.run_queue('interleaved', 4,
            extra={'SOS_PERSISTENT_TEST_OBSERVER_DELAY':os.environ.get('PERSISTENT_TEST_OBSERVER_DELAY','0.02'), **kernel})
        self.assertEqual(code, 0, summary)
        for i in range(4): self.equal_state('interleaved', i, base)
        events = self.events(sorted((self.root/'interleaved/logs').glob('*/GPU_SERVICE.log'))[-1])
        pieces = [e['id'] for e in events if e['phase']=='gpu_piece_start' and
                  e['kind'] in ('first_light','sos_iteration','table_transport')]
        runs = [s for i,s in enumerate(pieces) if i==0 or s!=pieces[i-1]]
        self.assertGreater(len(runs), len(set(runs)), 'no return to a previous state')
        if testing.sky_sos_root() is not None:   # the timeline renderer is a sky-sos research tool (not in the package)
            trace = testing.sky_sos_optional('scripts', 'sosfuse2_dev', 'pipeline_trace.py')   # absent in some checkouts: then no timeline check
            if trace is None: self.skipTest('the timeline renderer scripts/sosfuse2_dev/pipeline_trace.py is not in this sky-sos checkout')
            subprocess.run([sys.executable, '-B', str(trace), str(self.root/'interleaved')], check=True, timeout=10)
            self.assertTrue((self.root/'interleaved/timeline.html').is_file())
            print('TIMELINE', self.root/'interleaved/timeline.html', flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2, failfast=True)
