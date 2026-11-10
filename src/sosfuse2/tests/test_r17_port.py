"""r17 port (feat/sosfuse2-r17): four host checks and the pipeline runner seam. CPU only, no CUDA.

Run: python -B -m sosfuse2.tests.test_r17_port
1 column_cache job first-use comparison: miss, same-job hit, new-job hit and a poisoned payload all return
  the direct bytes.
2 preview reuse: the repack of one attempt renders no previews; packed products are unchanged.
3 nphi: v3 default 16 since 2026-10-21 (was 32; validated <=0.003 % on V14, a4f66a1); rig/engine.py stays 64; explicit 32/64 still generate their geometry.
4 kernel.cu production configurations preprocess; r17 flags remain reachable and parse.
5 pipeline.py --persistent --workers auto --gpu-active auto hands the real runner parser its work-root,
  geometry-cache, keep-geometry and column-cache arguments and the r17 environment.
7 stage1 wiring: rig S2 NVRTC options/keys, env 0 override, separate memo, fl_project options.
"""

from sosfuse2 import paths
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sosfuse2.numerical_checks import assert_equivalent

HERE = paths.PKG  # the package directory (kernel, configs, assets)
from sosfuse2 import column_cache, fused, paths, stratosphere


def phase_events(text):
    out = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get('phase') == 'column_cache':
            out.append(event)
    return out


class ColumnCacheFirstUse(unittest.TestCase):
    """The bytes the solver receives, not only the cache bookkeeping."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='r17-colcache-'))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        from sosfuse2.testing import sun_scope

        self.old = sun_scope()
        rng = np.random.default_rng(23)
        d = rng.normal(size=(200, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        self.arrays = dict(
            starts=np.array([[0, 0, 6371.1], [0, 0, 6381.0], [0, 0, 6401.0]]),
            dirs=d,
            cell_off=np.array([0, 70, 120, 200], dtype='i8'),
        )
        self.c = dict(strato_profile=stratosphere.profile_config({}), strato_old=self.old)

    def columns(self, job, cache=True):
        """Fresh Bundle each call (its memo would otherwise answer); the production method end to end."""
        bundle = SimpleNamespace(rays=200, cells=3, arrays=dict(self.arrays))
        env = {'SOSFUSE2_COLUMN_WORKERS': '1', 'SOSFUSE2_STRATO_HOST_NOPREFIX': '1'}
        if cache:
            env['SOSFUSE2_COLUMN_CACHE'] = str(self.tmp / 'columns')
        if job:
            env['SOSFUSE2_COLUMN_CACHE_JOB'] = str(self.tmp / job)
        stdout = io.StringIO()
        with (
            patch.dict(os.environ, env),
            contextlib.redirect_stdout(stdout),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter('always')
            out = fused.Kernel.strato_end_columns(SimpleNamespace(host=np.asarray, xp=np), bundle, self.c)
        os.environ.pop('SOSFUSE2_COLUMN_CACHE', None)
        os.environ.pop('SOSFUSE2_COLUMN_CACHE_JOB', None)
        events = phase_events(stdout.getvalue())
        self.assertEqual(len(events), 1, stdout.getvalue())
        return np.asarray(out), events[0], [str(w.message) for w in caught]

    def test_final_bytes_hit_miss_mismatch(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in ('SOSFUSE2_COLUMN_CACHE', 'SOSFUSE2_COLUMN_CACHE_JOB'):
                os.environ.pop(k, None)
            direct, event, _ = self.columns(None, cache=False)
            self.assertEqual((event['result'], event['verify']), ('disabled', None))
            same = lambda a: assert_equivalent(a, direct)
            # Miss: publish, reload and compare once in job A.
            out, event, _ = self.columns('jobA')
            same(out)
            self.assertEqual((event['result'], event['verify'], event['reason']), ('miss', 'reload_compare', None))
            key = event['key']
            files = list((self.tmp / 'columns').glob(key + '*.npy'))
            self.assertEqual(len(files), 1)
            # Same job: marker says match, no second calculation.
            with patch.object(column_cache, 'direct', wraps=column_cache.direct) as built:
                out, event, _ = self.columns('jobA')
            same(out)
            self.assertEqual((event['result'], event['verify']), ('hit', 'compare_skip'))
            self.assertEqual(built.call_count, 0)
            # New job: one direct comparison.
            out, event, _ = self.columns('jobB')
            same(out)
            self.assertEqual((event['result'], event['verify'], event['reason']), ('hit', 'first_compare', None))
            # Poisoned payload with a consistent checksum: the direct bytes win, the job disables it.
            poisoned = np.load(files[0]) * np.float32(1.001)
            np.save(files[0], poisoned, allow_pickle=False)
            files[0].with_suffix('.sha256').write_text(hashlib.sha256(files[0].read_bytes()).hexdigest())
            out, event, messages = self.columns('jobC')
            same(out)
            self.assertEqual(
                (event['result'], event['verify'], event['reason']),
                ('invalid', 'first_compare', 'E_COLUMN_CACHE_MISMATCH'),
            )
            self.assertTrue(any('E_COLUMN_CACHE_MISMATCH' in m for m in messages))
            out, event, _ = self.columns('jobC')
            same(out)
            self.assertEqual(
                (event['result'], event['verify'], event['reason']), ('disabled', 'disabled', 'E_COLUMN_CACHE_MISMATCH')
            )
            # Job A's match marker belongs to the old payload SHA only.
            out, event, _ = self.columns('jobA')
            same(out)
            self.assertEqual((event['result'], event['verify']), ('invalid', 'first_compare'))
            # Without a job the previous production behaviour stays: the file is trusted as before.
            out, event, _ = self.columns(None)
            self.assertEqual(out.tobytes(), poisoned.tobytes())
            self.assertEqual((event['result'], event['verify']), ('hit', None))
            # Markers stay in the job folders, never in the shared cache folder.
            self.assertEqual(
                sorted(p.parent.parent.name for p in self.tmp.glob('*/.column_cache_checks/*.json')),
                ['jobA', 'jobA', 'jobB', 'jobC'],
            )
            self.assertFalse(list((self.tmp / 'columns').glob('*.json')))


class PreviewReuse(unittest.TestCase):
    def test_repack_renders_once_same_products(self):
        spec = importlib.util.spec_from_file_location('r17_pack_state', paths.PKG / 'host/pack_state.py')
        pm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pm)
        tmp = Path(tempfile.mkdtemp(prefix='r17-preview-'))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rng = np.random.default_rng(3)
        table = dict(
            radiance=rng.uniform(0.1, 1.0, (2, 4, 6, 1)).astype('f4'),
            wavelength_nm=np.array([550.0], 'f4'),
            solar_elevation_deg=np.array([-3.0, -6.0], 'f4'),
        )
        results = {}
        for label, reuse in (('before', False), ('after', True)):
            out = tmp / label / 'state'
            out.mkdir(parents=True)
            np.savez(out / 'table.npz', **table)
            from sosfuse2.components import physics, optics_state

            (out / 'meta.json').write_text(
                json.dumps(dict(complete=True, contract='v4', physics=physics(), optics_state=optics_state({})))
            )
            with patch.object(pm, 'previews', wraps=pm.previews) as rendered:
                pm.pack_state(out, tmp / label / 'pack', 'gzip')
                pm.pack_state(out, tmp / label / 'pack', 'gzip', reuse_previews=reuse)  # run_gpu_state_v3 repack
            members = {}
            with tarfile.open(tmp / label / 'pack' / 'state.tar.gz') as tar:
                for m in tar.getmembers():
                    members[m.name] = tar.extractfile(m).read()
            meta = json.loads(members.pop('meta.json'))
            meta.pop('peak_rss_bytes')
            results[label] = (rendered.call_count, members, meta)
        self.assertEqual((results['before'][0], results['after'][0]), (2, 1))
        self.assertEqual(sorted(results['after'][1]), ['preview_m03.png', 'preview_m06.png', 'table.npz'])
        self.assertEqual(results['before'][1:], results['after'][1:])
        with self.assertRaisesRegex(TypeError, 'positional'):
            pm.pack_state(out, tmp / 'x', 'gzip', None, True)  # keyword-only: no silent positional reuse


class Nphi(unittest.TestCase):
    def test_default_32_explicit_64_generates(self):
        from sosfuse2 import run_gpu_state_v3 as rv
        from sosfuse2 import geometry as gm
        from sosfuse2.strato_states import build

        # accepted/engine.py keeps 64 (it is in the geometry signature, geometry_contract.UPSTREAM);
        # the v3 adapter replaces its grid counts with settings_for (run_gpu_state_v3.py create_configured, the create() grid hook).
        engine = (paths.ACCEPTED / 'engine.py').read_text()
        self.assertRegex(engine, r"if mode == 'full':\s*h, ch = old\.grids\(1\)\s*nmu, nphi, path_n = 12, 64, 12")
        tmp = Path(tempfile.mkdtemp(prefix='r17-nphi-'))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith('SOSFUSE2_')}
        # This test clears tuning flags, but still needs its isolated synthetic O4 cache.
        env['SOSFUSE2_WORK_ROOT'] = str(paths.work_root())
        env['SOSFUSE2_OFFLINE'] = '1'
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(rv.settings_for(SimpleNamespace(reduced=False, settings_json=None))['nphi'], 16)
            explicit = tmp / 'phi64.json'
            explicit.write_text(json.dumps(dict(nphi=64)))
            self.assertEqual(rv.settings_for(SimpleNamespace(reduced=False, settings_json=explicit))['nphi'], 64)
            rays = {}
            for nphi in (32, 64):
                path = tmp / f'reduced{nphi}.json'
                path.write_text(json.dumps(dict(nphi=nphi)))
                cfg = rv.settings_for(SimpleNamespace(reduced=True, settings_json=path))
                s, scope = rv.create_configured(
                    tmp / f'prepare{nphi}', cfg, np.array([550.0]), 'numpy', build()['states'][1]
                )
                g = gm.TransientGeometry(s, scope)
                g.cell(0)
                self.assertEqual(s.geo.cfg['nphi'], nphi)
                rays[nphi] = g.stats['rays']
        self.assertGreater(rays[64], rays[32])
        self.assertGreater(rays[32], 0)


# Production flag sets: C3 (+automatic STRATO/PROFILE), C3_nostrato,
# C3 with the automatic TROPO_MIE/AEROSOL_2L, and none.
SPECS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (1, 2, 0), (0, 2, 0))  # precompile.py SPECS


def flag_sets():
    c3 = json.loads((HERE / 'configs/C3.json').read_text())['kernel_flags']
    off = json.loads((HERE / 'configs/C3_nostrato.json').read_text())['kernel_flags']
    return dict(
        c3=c3 + ['STRATO', 'PROFILE'],
        c3_nostrato=off + ['PROFILE'],
        c3_tropo=c3 + ['STRATO', 'PROFILE', 'TROPO_MIE', 'AEROSOL_2L'],
        none=[],
    )


def cases():
    from sosfuse2.kernel_config import resolve_config

    for name, flags in flag_sets().items():
        for W in (1, 15):
            effective = resolve_config(flags, W=W).defines
            for spec in SPECS:
                base = (
                    [f'-DFW={W}']
                    + [f'-D{f}' for f in effective]
                    + [f'-DK_FIRST={spec[0]}', f'-DK_PHASE={spec[1]}', f'-DK_BOUNDARY={spec[2]}']
                )
                yield f'{name}/W{W}/{spec}', base
                if spec == (1, 0, 0):
                    yield f'{name}/W{W}/{spec}/raylimit', base + ['-DLAUNCH_RAY_LIMIT']


def token_digest(kernel, jobs=8):
    from concurrent.futures import ThreadPoolExecutor

    def one(case):
        label, defines = case
        text = subprocess.check_output(
            ['clang++', '-E', '-P', '-x', 'c++', *defines, str(kernel)], text=True, stderr=subprocess.PIPE
        )
        return label, hashlib.sha256('\n'.join(re.findall(r'\w+|[^\s]', text)).encode()).hexdigest()

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = dict(pool.map(one, list(cases())))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), rows


@unittest.skipIf(shutil.which('clang++') is None, 'clang++ required for lexical preprocessing')
class KernelFlagsParse(unittest.TestCase):
    def test_production_and_new_flags_parse(self):
        _, rows = token_digest(HERE / 'kernel.cu')
        self.assertEqual(len(rows), 56)  # four physical sets x FW1/FW15 x six roles + ray-limit
        # Each r17 macro changes the token stream (it is reachable) and still passes a host syntax check.
        from sosfuse2.tests.test_kernel_static import STUB

        stub = Path(tempfile.mkdtemp(prefix='r17-stub-'))
        self.addCleanup(shutil.rmtree, stub, ignore_errors=True)
        (stub / 'stub.h').write_text(
            STUB + 'float __expf(float);float __sinf(float);float __cosf(float);\n'
            'template<class T>T __shfl_xor_sync(unsigned,T v,int){return v;}\n'
        )
        base = [f'-D{f}' for f in sorted(flag_sets()['c3_nostrato'])] + [
            '-DFW=15',
            '-DK_FIRST=0',
            '-DK_PHASE=0',
            '-DK_BOUNDARY=0',
        ]
        plain = subprocess.check_output(['clang++', '-E', '-P', '-x', 'c++', *base, str(HERE / 'kernel.cu')], text=True)
        for name in ('SOSFUSE2_FASTMATH',):
            on = subprocess.check_output(
                ['clang++', '-E', '-P', '-x', 'c++', *base, '-D' + name, str(HERE / 'kernel.cu')], text=True
            )
            self.assertNotEqual(on, plain, name)
            result = subprocess.run(
                [
                    'clang++',
                    '-std=c++17',
                    '-ffp-contract=off',
                    '-fsyntax-only',
                    '-x',
                    'c++',
                    '-include',
                    str(stub / 'stub.h'),
                    *base,
                    '-D' + name,
                    str(HERE / 'kernel.cu'),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, (name, result.stderr[-2000:]))


class FakeRawKernel:
    """Records what fused.Kernel hands NVRTC; compiles nothing."""

    made = []

    def __init__(self, code, name, options=(), backend=None):
        self.code, self.name, self.options, self.backend = code, name, tuple(options), backend
        self.attributes = dict(num_regs=0)
        FakeRawKernel.made.append(self)

    def compile(self):
        pass


class Stage1Wiring(unittest.TestCase):
    """Rig profile S2 (FMAD+FASTMATH) -> NVRTC options and stage1 keys, env 0 override,
    separate memo entries, identical fl_project options. Fake RawKernel; no CUDA."""

    def kernel(self, profile_name):
        from sosfuse2 import device_profile

        flags = sorted(set(json.loads((HERE / 'configs/C3.json').read_text())['kernel_flags']) | {'STRATO', 'PROFILE'})
        k = fused.Kernel(1, backend='numpy', block_size=256, flags=flags)
        props = dict(sharedMemPerBlockOptin=1 << 20, sharedMemPerBlock=49152, multiProcessorCount=68)
        k.xp = SimpleNamespace(
            RawKernel=FakeRawKernel,
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDevice=lambda: 0, getDeviceProperties=lambda device: props)
            ),
        )
        with patch.dict(os.environ, {'SOSFUSE2_DEVICE_PROFILE': profile_name}, clear=True):
            k._device = device_profile.select(None)
        return k

    def test_rig_s2_options_keys_override_fl_project(self):
        exact = ('--fmad=false', '--prec-div=true', '--prec-sqrt=true')
        s2 = ('--fmad=true', '--prec-div=false', '--prec-sqrt=false')
        base = lambda k: (tuple(k.flags), 15, 15, 0, 0, 0, ('L', k.L))
        # eda profile: the exact pre-r17 options and key.
        eda = self.kernel('eda')
        with patch.dict(os.environ, {}, clear=True):
            raw_eda = eda.raw_kernel(15, 0, 0, 0)
        self.assertEqual(raw_eda.options[1:4], exact)
        self.assertEqual([o for o in raw_eda.options if o.startswith('-DSOSFUSE2_')], ['-DSOSFUSE2_DETERMINISTIC=0'])
        self.assertEqual(list(eda.raw), [base(eda)])
        # rig profile, env unset: S2 from configs/devices.json stage1_defaults.
        rig = self.kernel('rig')
        with patch.dict(os.environ, {}, clear=True):
            raw_s2 = rig.raw_kernel(15, 0, 0, 0)
        self.assertEqual(raw_s2.options[1:4], s2)
        self.assertIn('-DSOSFUSE2_FASTMATH', raw_s2.options)
        self.assertNotIn('-DSOSFUSE2_FMAD', raw_s2.options)  # FMAD is the --fmad option only
        self.assertEqual(set(raw_s2.options) - set(raw_eda.options), set(s2) | {'-DSOSFUSE2_FASTMATH'})
        attrs = rig.attributes[str(15)]
        self.assertEqual((attrs['implicit_fma'], attrs['stage1']), (True, ['FMAD', 'FASTMATH']))
        # Explicit 0 overrides the profile: exact options, the plain key, a separate memo entry.
        with patch.dict(os.environ, {'SOSFUSE2_FMAD': '0', 'SOSFUSE2_FASTMATH': '0'}, clear=True):
            raw_off = rig.raw_kernel(15, 0, 0, 0)
        self.assertEqual(raw_off.options, raw_eda.options)
        self.assertEqual(set(rig.raw), {(base(rig), ('FMAD', 'FASTMATH')), base(rig)})
        self.assertIsNot(rig.raw[base(rig)], rig.raw[(base(rig), ('FMAD', 'FASTMATH'))])
        with patch.dict(os.environ, {}, clear=True):  # memo hit, no recompilation
            count = len(FakeRawKernel.made)
            self.assertIs(rig.raw_kernel(15, 0, 0, 0), raw_s2)
            self.assertEqual(len(FakeRawKernel.made), count)
        # fl_project: the ray-range first-light key and exactly its options, for S2 and for the override.
        for env, stage1 in (({}, ('FMAD', 'FASTMATH')), ({'SOSFUSE2_FMAD': '0', 'SOSFUSE2_FASTMATH': '0'}, ())):
            with patch.dict(os.environ, env, clear=True):
                first = rig.raw_kernel(15, 1, 0, 0, ray_limit=True)
                project = rig.fl_project_kernel(15, 0)
            key = (tuple(rig.flags), 15, 15, 1, 0, 0, ('L', rig.L), 'LAUNCH_RAY_LIMIT')
            key = (key, stage1) if stage1 else key
            self.assertIs(rig.raw[key], first)
            self.assertIs(rig.raw[key + ('fl_project',)], project)
            self.assertEqual((project.name, project.options, project.code), ('fl_project', first.options, first.code))
            self.assertIn('-DLAUNCH_RAY_LIMIT', first.options)
            self.assertEqual(first.options[1:4], s2 if stage1 else exact)
        with (
            patch.dict(os.environ, {'SOSFUSE2_FASTMATH': 'on'}, clear=True),
            self.assertRaisesRegex(ValueError, 'E_STAGE1_FLAG'),
        ):
            rig.raw_kernel(15, 0, 0, 0)


RECORDER = '''import json, os, sys
from pathlib import Path
sys.path.insert(0, {src!r})
from sosfuse2 import run_gpu_state_v3 as rv
def record(a):
    keys = ('SOSFUSE2_COLUMN_CACHE', 'SOSFUSE2_COLUMN_CACHE_JOB', 'SOSFUSE2_STATE_ID', 'SOSFUSE2_FMAD',
            'SOSFUSE2_STRATO_HOST_COMPONENT_RADIUS', 'SOSFUSE2_STRATO_ASYNC', 'SOSFUSE2_COLUMN_WORKERS')
    doc = dict(argv=sys.argv[1:], work_root=str(a.work_root), geometry_cache=str(a.geometry_cache),
               keep_geometry=a.keep_geometry, column_cache=str(a.column_cache), env={{k: os.environ.get(k) for k in keys}},
               persistent=bool(os.environ.get('SOS_PERSISTENT_ROOT')), persistent_key=bool(os.environ.get('SOS_PERSISTENT_KEY')))
    (Path(os.environ['R17_SEAM_OUT'])/(a.id+'.json')).write_text(json.dumps(doc))
rv.run = record
rv.main()
'''


class PipelineRunnerSeam(unittest.TestCase):
    def test_persistent_auto_passes_runner_arguments(self):
        from sosfuse2.strato_states import build

        tmp = Path(tempfile.mkdtemp(prefix='r17-seam-')).resolve()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        states = []
        for i in range(2):
            s = copy.deepcopy(build()['states'][1])
            s['id'] = f'R{i}'
            states.append(s)
        (tmp / 'states.json').write_text(json.dumps(dict(states=states)))
        (tmp / 'runner.py').write_text(RECORDER.format(src=str(paths.SRC)))
        (tmp / 'calls').mkdir()
        env = paths.child_env(
            {k: v for k, v in os.environ.items() if not k.startswith(('SOSFUSE2_', 'SOS_PERSISTENT', 'STRATO_ATTACH'))}
        )
        env.update(
            R17_SEAM_OUT=str(tmp / 'calls'),
            SOSFUSE2_FMAD='1',
            SOSFUSE2_STRATO_HOST_COMPONENT_RADIUS='0',
            PYTHONDONTWRITEBYTECODE='1',
        )
        work, geometry, columns = tmp / 'work', tmp / 'geometry', tmp / 'columns'
        cmd = [
            *paths.module_command('pipeline'),
            '--states',
            str(tmp / 'states.json'),
            '--ids',
            'R0,R1',
            '--work-root',
            str(work),
            '--geometry-cache',
            str(geometry),
            '--column-cache',
            str(columns),
            '--keep-geometry',
            '--persistent',
            '--workers',
            'auto',
            '--gpu-active',
            'auto',
            '--backend',
            'numpy',
            '--reduced',
            '--gpu-lock',
            str(tmp / 'gpu.lock'),
            '--runner',
            str(tmp / 'runner.py'),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
        summary = json.loads((work / 'pipeline_summary.json').read_text())
        self.assertEqual(proc.returncode, 0, (proc.stdout[-2000:], proc.stderr[-2000:], summary))
        self.assertTrue(summary['persistent'] and summary['complete'])
        for i in range(2):
            doc = json.loads((tmp / 'calls' / f'R{i}.json').read_text())
            argv = doc['argv']
            # pipeline.py: --threads 1, the column pool keeps cpu_budget() // states prepared at once (pipeline.prep_workers).
            from sosfuse2.pipeline import cpu_budget

            columns_expected = str(max(1, cpu_budget() // summary['workers']))
            for flag, value in (('--work-root', work), ('--geometry-cache', geometry), ('--column-cache', columns)):
                self.assertEqual(argv[argv.index(flag) + 1], str(value.resolve()), flag)
            self.assertIn('--keep-geometry', argv)
            self.assertEqual(
                (doc['work_root'], doc['geometry_cache'], doc['keep_geometry'], doc['column_cache']),
                (str(work.resolve()), str(geometry.resolve()), True, str(columns.resolve())),
            )
            self.assertEqual(
                doc['env'],
                dict(
                    SOSFUSE2_COLUMN_CACHE=str(columns.resolve()),
                    SOSFUSE2_COLUMN_CACHE_JOB=str(work.resolve()),
                    SOSFUSE2_STATE_ID=f'R{i}',
                    SOSFUSE2_FMAD='1',
                    SOSFUSE2_STRATO_HOST_COMPONENT_RADIUS='0',
                    SOSFUSE2_STRATO_ASYNC=None,
                    SOSFUSE2_COLUMN_WORKERS=columns_expected,
                ),
            )
            self.assertTrue(doc['persistent'] and doc['persistent_key'])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--print-main-digest':
        # Maintenance: digest of a given kernel.cu (e.g. `git show main:src/sosfuse2/kernel.cu > k.cu`).
        print(token_digest(Path(sys.argv[2]))[0])
        raise SystemExit(0)
    unittest.main(verbosity=2)
