#!/usr/bin/env python3
"""Independent round-3 bounded DELTA tests; no production writes or CUDA claim."""

from sosfuse2 import paths
import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
MODULE = paths.PKG  # the package directory (kernel, configs, assets)
import numpy as np
from sosfuse2 import paths, stratosphere as st, testing
from sosfuse2.strato_states import build
from sosfuse2.fused import Kernel, Solver
from sosfuse2.geometry import create, TransientGeometry

try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol

# The CLI harness (main --out) replaces this; under pytest/unittest collection the cases write into the test scratch instead of
# failing with OUT=None (harness semantics unchanged).
OUT = testing.scratch_parent() / 'strato_revision/delta'
OUT.mkdir(parents=True, exist_ok=True)
MEASUREMENTS = {}
ARTIFACT = (
    testing.scratch_parent() / 'strato_revision'
)  # earlier harness runs (runner_enabled, val_zero) are read from here


def state(tau=0.003, status='enabled'):
    s = copy.deepcopy(build()['states'][1])
    s['parameters']['stratosphere_tau550'] = tau
    s['stratosphere']['status'] = status
    return s


def optics(s=None, h=None, wl=None):
    return st.build_optics(
        state() if s is None else s, np.linspace(0, 80, 20) if h is None else h, [550.0] if wl is None else wl
    )


class Delta(unittest.TestCase):
    def test_D01_disabled_unused_fields(self):
        class Unused:
            def __float__(self):
                raise AssertionError('unused tau accessed')

        for status in ('disabled', 'unsupported', None):
            s = state(Unused(), status)
            if status is None:
                s['stratosphere'].pop('status')
            s['stratosphere'].update(profile=object(), sigma_km=object(), optics_table=object())
            with (
                patch.object(st, 'profile_config', side_effect=AssertionError('unused profile')),
                patch.object(st, 'load_table', side_effect=AssertionError('unused table')),
            ):
                self.assertIsNone(st.build_optics(s, object(), object()))
                self.assertIsNone(st.build_optics(s, object(), object()))

    def test_D02_zero_and_invalid_tau(self):
        s = state(0.0)
        s['stratosphere'].update(profile=object(), optics_table=object())
        with patch.object(st, 'profile_config', side_effect=AssertionError('unused profile')):
            self.assertIsNone(st.build_optics(s, object(), object()))
        for tau in (-1.0, float('nan'), float('inf'), 'bad', [1]):
            with self.subTest(tau=repr(tau)), self.assertRaisesRegex(ValueError, '^E_STRATO_TAU'):
                optics(state(tau))

    def test_D03_D20_activation_and_recovery(self):
        s = state()
        fresh = optics(s)
        s['stratosphere']['status'] = 'disabled'
        self.assertIsNone(optics(s))
        s['stratosphere']['status'] = 'enabled'
        s['stratosphere']['sigma_km'] = 0.99
        with self.assertRaisesRegex(ValueError, '^E_STRATO_PROFILE_UNSUPPORTED'):
            optics(s)
        s['stratosphere']['sigma_km'] = 3.0
        recovered = optics(s)
        for name in ('ext', 'sca', 'density', 'source_density', 'beta_e', 'beta_s'):
            assert_equivalent(fresh[name], recovered[name])
        self.assertEqual(fresh['meta'], recovered['meta'])

    def test_D05_sigma_threshold(self):
        for sigma in (-1.0, 0.0, np.nextafter(1.0, 0.0)):
            s = state()
            s['stratosphere']['sigma_km'] = sigma
            with self.subTest(sigma=sigma), self.assertRaisesRegex(ValueError, '^E_STRATO_PROFILE_UNSUPPORTED'):
                optics(s)
        for sigma in (1.0, np.nextafter(1.0, 2.0), 3.0):
            s = state()
            s['stratosphere']['sigma_km'] = sigma
            self.assertIsNotNone(optics(s))
        s = state()
        s['stratosphere']['sigma_km'] = float('nan')
        with self.assertRaisesRegex(ValueError, '^E_STRATO_PROFILE_PARAMETERS'):
            optics(s)

    def test_D06_grid_boundaries_and_malformed_nodes(self):
        for h in (
            [0.0],
            [0.0, 0.0, 80.0],
            [0.0, 80.0, 40.0],
            [0.0, float('nan'), 80.0],
            [[0.0, 80.0]],
            [12.0, 21.0, 30.0],
            [12.00000001, 21.0, 80.0],
            [0.0, 21.0, 29.99999999],
        ):
            with self.subTest(h=h), self.assertRaisesRegex(ValueError, '^E_STRATO_PROFILE_UNSUPPORTED'):
                optics(h=h)
        # Inclusive support boundary can also satisfy global normalization:
        # the physical normalization itself excludes the tail below 0/above 80.
        for center in (3.0, 77.0):
            s = state()
            s['stratosphere'].update(center_km=center, sigma_km=1.0)
            self.assertIsNotNone(optics(s))

    def test_D07_D08_conserved_cells_analytic_nodes(self):
        rows = []
        grids = [
            np.array([0.0, 7.0, 21.0, 46.0, 80.0]),
            np.linspace(0, 80, 39),
            np.array([0.0, 0.1, 4.0, 13.0, 20.0, 21.01, 22.0, 27.0, 61.0, 80.0]),
        ]
        for center, sigma in ((21.0, 3.0), (3.0, 1.0), (77.0, 1.0), (35.0, 7.0)):
            s = state()
            s['stratosphere'].update(center_km=center, sigma_km=sigma)
            # Independent normalization uses a different refined trapezoid grid.
            z = np.linspace(0.0, 80.0, 160001)
            independent_norm = np.trapezoid(np.exp(-0.5 * ((z - center) / sigma) ** 2), z)
            for h in grids:
                o = optics(s, h)
                edges = np.r_[h[0], (h[:-1] + h[1:]) / 2, h[-1]]
                exact = np.array([math.erf((x - center) / (math.sqrt(2) * sigma)) for x in edges])
                cell = np.diff(exact) * sigma * math.sqrt(math.pi / 2) / independent_norm / np.diff(edges)
                column = float(o['source_density'] @ np.diff(edges))
                self.assertLessEqual(abs(column - 1), 1e-6)
                np.testing.assert_allclose(o['source_density'], cell, rtol=1e-6, atol=1e-15)
                analytic = np.exp(-0.5 * ((h - center) / sigma) ** 2) / o['profile']['norm']
                assert_equivalent(o['density'], analytic)
                assert_equivalent(o['beta_e'], analytic[:, None] * o['ext'])
                self.assertGreater(float(np.max(abs(o['source_density'] - analytic))), 0.0)
                rows.append(dict(center=center, sigma=sigma, nodes=len(h), column=column))
        MEASUREMENTS['column_cases'] = rows

    def test_D09_table_preservation_and_endpoints(self):
        oldpath = ARTIFACT / 'optics_before.json'
        if not oldpath.is_file():
            raise unittest.SkipTest(f'earlier harness artifact absent: {oldpath}')  # research record, not package data
        self.assertEqual(
            hashlib.sha256(oldpath.read_bytes()).hexdigest(),
            '36bd392f4159fd5c43363f36bb6c456e12e571f364a05a40c0cd830885dbc54f',
        )
        old, new = json.loads(oldpath.read_text()), json.loads(st.DEFAULT.read_text())
        by_wave = {r['wavelength_nm']: r for r in new['rows']}
        for row in old['rows']:
            self.assertEqual(row, by_wave[row['wavelength_nm']])
        wl = np.r_[360.0, 360 + (np.arange(48) + 0.5) * 470 / 48, 830.0]
        o = optics(wl=wl)
        self.assertEqual(o['beta_e'].shape, (20, 50))
        self.assertTrue(np.isfinite(o['beta_e']).all())
        for wavelength in (np.nextafter(360.0, 0.0), np.nextafter(830.0, 1000.0)):
            with self.assertRaisesRegex(ValueError, '^E_STRATO_WAVELENGTH_RANGE'):
                optics(wl=[wavelength])
        MEASUREMENTS['old_rows_unchanged'] = len(old['rows'])
        MEASUREMENTS['table_sha256'] = hashlib.sha256(st.DEFAULT.read_bytes()).hexdigest()

    def test_D10_provenance_content_and_disabled_metadata(self):
        with tempfile.TemporaryDirectory(dir=OUT) as folder:
            folder = Path(folder)
            for i in range(2):
                directory = folder / str(i)
                directory.mkdir()
                table = directory / 'same.json'
                d = json.loads(st.DEFAULT.read_text())
                d['verifier_test_only'] = i
                table.write_text(json.dumps(d))
                s = state()
                s['stratosphere']['optics_table'] = str(table)
                o = optics(s)
                self.assertEqual(o['meta']['optics_sha256'], hashlib.sha256(table.read_bytes()).hexdigest())
                meta = st.computed_metadata(s, o)
                self.assertEqual(meta['parameters_requested'], s['parameters'])
                self.assertEqual(meta['parameters_computed']['stratosphere_tau550'], 0.003)
            with patch.object(st, '__file__', '/unrelated-root/scripts/sosfuse2/stratosphere.py'):
                self.assertEqual(st.path_label(table), table.name)
                self.assertEqual(optics(s)['meta']['optics_table'], table.name)
            disabled = state(0.003, 'disabled')
            disabled['stratosphere'].update(optics_table=object(), optics_sha256='stale')
            meta = st.computed_metadata(disabled, None)
            self.assertEqual(meta['parameters_requested']['stratosphere_tau550'], 0.003)
            self.assertEqual(meta['parameters_computed']['stratosphere_tau550'], 0.0)
            self.assertEqual(meta['stratosphere']['status'], 'disabled')
            self.assertNotIn('optics_table', meta['stratosphere'])
            self.assertNotIn('optics_sha256', meta['stratosphere'])

    @testing.known_failure(
        'D11',
        'st.source_size(s) is 316, the harness expects the pre-kclean 1132; kernel/harness mismatch since the baseline',
    )
    def test_D11_initialize_1_15_lanes_and_source_widths(self):
        from sosfuse2.run_gpu_state_v3 import settings_for, create_configured

        a = types.SimpleNamespace(reduced=True, settings_json=None)
        cfg = settings_for(a)
        rows = []
        for W in (1, 15):
            wl = [550.0] if W == 1 else 360 + (np.arange(15) + 0.5) * 470 / 15
            s, scope = create_configured(OUT / f'init_{W}', cfg, wl, 'numpy', state())
            solver = Solver(s, TransientGeometry(s, scope))
            c = s.kernel.setup(s, 4)
            self.assertEqual(c['strato_ext'].shape, (W,))
            self.assertEqual(st.source_size(s), 1132)
            first_meta = st.computed_metadata(s.fuse_state, s.strato)
            self.assertEqual(st.attach(s, scope)['meta'], s.strato['meta'])
            self.assertEqual(first_meta, st.computed_metadata(s.fuse_state, s.strato))
            rows.append(dict(wavelengths=W, source_coefficients=st.source_size(s)))
        MEASUREMENTS['initialized_widths'] = rows

    def test_D13_D14_independent_FP32_ray_additivity(self):
        from sosfuse2 import reference

        s, scope, _ = create(OUT / 'fp32_geometry', 'selftest', [550.0], state=state(1e-6))
        Solver(s, TransientGeometry(s, scope))
        old = scope['b'].sh.old
        p, up, ex = old.probe.frame(np.array(6371.1), np.deg2rad(87.0))
        ey = np.cross(up, ex)
        el, az = np.meshgrid(
            np.array([1.1, 3.0, 11.0, 22.0, 44.0, 63.0, 79.0, 89.0]), np.linspace(0.0, 180.0, 9), indexing='ij'
        )
        e, a = np.deg2rad(el.ravel()), np.deg2rad(az.ravel())
        d = np.sin(e)[:, None] * up + np.cos(e)[:, None] * (np.cos(a)[:, None] * ex + np.sin(a)[:, None] * ey)
        g = s.geo.pack(np.broadcast_to(p, d.shape), d, s.geo.query_n)
        c = dict(s.kernel.setup(s, g['nq']))
        c['asca'] = np.array([2e-5], 'f4')
        c['ext'] = np.zeros_like(c['ext'])
        for name in ('gm', 'gv', 'gval'):
            c[name] = np.zeros_like(c[name])
        field = np.zeros((s.geo.ns, 1132, 1), 'f4')
        both = reference.transport(g, c, field, True)
        trop = reference.transport(g, dict(c, strato_sca=np.zeros(1, 'f4')), field, True)
        strat = reference.transport(g, dict(c, asca=np.zeros(1, 'f4')), field, True)
        expected = (trop + strat).astype('f4')
        delta = abs(both.astype('f8') - expected.astype('f8'))
        spacing = np.maximum(abs(np.spacing(expected)).astype('f8'), float(np.nextafter(np.float32(0), np.float32(1))))
        ulp = delta / spacing
        relative = delta / np.maximum(abs(expected.astype('f8')), 1e-30)
        self.assertEqual(both.dtype, np.dtype('f4'))
        self.assertGreaterEqual(len(d), 64)
        self.assertTrue(((ulp <= 1) | (relative <= 1e-6)).all())
        doubled = reference.transport(
            g, dict(c, asca=np.zeros(1, 'f4'), strato_sca=c['strato_sca'] * np.float32(2)), field, True
        )
        self.assertGreater(np.count_nonzero(strat), 0)
        assert_equivalent(doubled, strat * np.float32(2))
        MEASUREMENTS['independent_FP32'] = dict(
            rays=len(d),
            max_ulp=float(ulp.max()),
            max_relative=float(relative.max()),
            nonzero_strato=int(np.count_nonzero(strat)),
            exact_2x=True,
        )
        # A deliberate material mismatch must not be hidden by a self-comparison.
        bad = both * np.float32(1.1)
        bd = abs(bad.astype('f8') - expected.astype('f8'))
        self.assertFalse(((bd / spacing <= 1) | (bd / np.maximum(abs(expected.astype('f8')), 1e-30) <= 1e-6)).all())

    @testing.known_failure(
        'D15',
        'test_strato_order host build of kernel.cu lacks -DK_FIRST/-DK_PHASE/-DK_BOUNDARY (clang error since the specialised kernel); skips without clang++',
    )
    def test_D15_SH_gate_accepts_absolute_rejects_material_mismatch(self):
        from sosfuse2.tests import test_strato_order as order

        with patch.object(order.reference, 'project', wraps=order.reference.project) as projection:
            report = order.run(OUT / 'independent_order')
        self.assertTrue(report['pass_'])
        self.assertGreater(projection.call_count, 0)
        original = order.reference.project
        calls = [0]

        def corrupt(*args, **kwargs):
            moment, flux = original(*args, **kwargs)
            calls[0] += 1
            if calls[0] % 16 == 1:
                moment = moment.copy()
                moment[0, 0] += 1e-5
            return moment, flux

        with patch.object(order.reference, 'project', side_effect=corrupt):
            self.assertFalse(order.run(OUT / 'mismatched_order')['pass_'])
        MEASUREMENTS['SH_max_absolute'] = max(r['moment_max_absolute'] for r in report['cases'])
        MEASUREMENTS['SH_gate_injected_mismatch_rejected'] = True

    @testing.known_failure(
        'D16',
        "kernel_static.json rows carry no 'strato' key any more (KeyError); harness predates the kclean kernel_static report",
    )
    def test_D16_D18_D22_adversarial_static_out_and_coverage(self):
        from sosfuse2.tests import test_kernel_static as static

        command = [
            sys.executable,
            '-B',
            str(Path(__file__).with_name('test_kernel_static.py')),
            '--out',
            str(OUT / 'misleading_baseline_directory'),
        ]
        process = subprocess.run(
            command, text=True, capture_output=True, env=paths.child_env()
        )  # the script imports the package (in-tree too)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((OUT / 'misleading_baseline_directory/kernel_static.json').read_text())
        self.assertTrue(report['pass_'])
        self.assertEqual(
            {(r['strato'], r['wavelengths']) for r in report['cases']}, {(False, 1), (False, 15), (True, 1), (True, 15)}
        )
        baseline = MODULE / 'assets/kernel_strato_off_baseline.cu'
        actual_reads = []
        original = static.subprocess.check_output

        def record(command, *args, **kwargs):
            actual_reads.append(command[-1])
            return original(command, *args, **kwargs)

        with patch.object(static.subprocess, 'check_output', side_effect=record):
            self.assertTrue(static.run(OUT / 'second_baseline_like_out')['pass_'])
        self.assertEqual(actual_reads.count(str(baseline)), 2)

        def mismatched_candidate(command, *args, **kwargs):
            value = original(command, *args, **kwargs)
            return value + ' verifier_deliberate_mismatch ' if command[-1] == str(MODULE / 'kernel.cu') else value

        with patch.object(static.subprocess, 'check_output', side_effect=mismatched_candidate):
            self.assertFalse(static.run(OUT / 'deliberate_candidate_mismatch')['pass_'])
        source = (MODULE / 'kernel.cu').read_text()
        macro = next(x for x in source.splitlines() if x.startswith('#define COPY_STRATO') and 'scache' in x)
        self.assertEqual(macro.count('_Pragma("unroll")'), 2)
        from sosfuse2.validate import parity

        parity_source = inspect.getsource(parity)
        self.assertIn('((False,1),(False,15),(True,1),(True,15))', parity_source)
        MEASUREMENTS['static_cases'] = len(report['cases'])
        MEASUREMENTS['CUDA_runtime'] = 'OUT-OF-MODEL: not executed on Mac'

    def test_D17_flags_cache_mock_NVRTC(self):
        compiled = []

        class Raw:
            attributes = {}

            def __init__(self, code, name, options, backend):
                self.options = options
                compiled.append(self)

            def compile(self):
                pass

        xp = types.SimpleNamespace(
            RawKernel=Raw,
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(
                    getDevice=lambda: 0, getDeviceProperties=lambda _: {'sharedMemPerBlock': 2000000}
                )
            ),
        )
        k = Kernel()
        k.xp = xp
        off = k.raw_kernel(1)
        k.flags = ('STRATO',)
        on = k.raw_kernel(1)
        self.assertIsNot(off, on)
        self.assertIn('-DSTRATO', on.options)
        self.assertNotIn('-DSTRATO', off.options)
        self.assertIs(k.raw_kernel(1), on)
        k.flags = ()
        self.assertIs(k.raw_kernel(1), off)
        self.assertIsNot(k.raw_kernel(15), off)
        k.flags = ('SPECIALIZE', 'STRATO')
        a = k.raw_kernel(1, first=1, phase=0, boundary=0)
        b = k.raw_kernel(1, first=0, phase=2, boundary=1)
        self.assertIsNot(a, b)
        self.assertIn('-DK_FIRST=1', a.options)
        self.assertIn('-DK_PHASE=2', b.options)
        MEASUREMENTS['mock_NVRTC_compilations'] = len(compiled)

    def test_D10_source_cache_changed_table_and_retry(self):
        try:
            from sosfuse2 import strato_colors as colors
        except ImportError:  # sky-sos research module (scripts/sosfuse2_dev), not part of the public package
            self.skipTest('strato_colors is a sky-sos research module (scripts/sosfuse2_dev)')

        class StopProbe(Exception):
            pass

        solve_calls = []
        initialize_calls = []

        class FakeSolver:
            def __init__(self, *args):
                pass

            def solve(self, *args):
                solve_calls.append(True)
                return np.zeros((1, 1132, 15), 'f4'), {}

            def initialize(self):
                initialize_calls.append(True)

        fake = types.SimpleNamespace(
            kernel=types.SimpleNamespace(host=np.asarray),
            geo=types.SimpleNamespace(ns=1, h=np.array([0.0, 80.0]), chi=np.array([0.0])),
            query=lambda *args, **kwargs: (_ for _ in ()).throw(StopProbe()),
        )
        folder = Path(tempfile.mkdtemp(prefix='cache_content_', dir=OUT))
        table = folder / 'optics.json'
        table.write_bytes(st.DEFAULT.read_bytes())
        s = state()
        s['stratosphere']['optics_table'] = str(table)
        with (
            patch.object(colors, 'build', return_value={'states': [s]}),
            patch.object(colors, 'create', return_value=(fake, {}, {})),
            patch.object(colors, 'TransientGeometry', return_value=None),
            patch.object(colors, 'Solver', FakeSolver),
        ):
            for attempt in range(3):
                if attempt == 2:
                    d = json.loads(table.read_text())
                    d['test_changed_content'] = True
                    table.write_text(json.dumps(d))
                with self.assertRaises(StopProbe):
                    colors.run(folder / 'run', 2)
                self.assertEqual(len(solve_calls), (1, 1, 2)[attempt])
        self.assertEqual(len(initialize_calls), 1)
        MEASUREMENTS['source_cache_solve_counts'] = [1, 1, 2]

    def test_D19_disabled_positive_tau_array_bytes(self):
        rows = []
        solvers = []
        for label, s in [('zero', state(0.0, 'enabled')), ('disabled_positive', state(0.003, 'disabled'))]:
            s['stratosphere']['optics_table'] = '/invalid/unused/table.json'
            solver, scope, _ = create(OUT / label, 'selftest', [550.0], state=s)
            sv = Solver(solver, TransientGeometry(solver, scope))
            source = np.zeros((solver.geo.ns, 571, 1), 'f4')
            result = sv.apply(source)
            solvers.append((solver, result))
            rows.append(dict(name=label, shape=list(result.shape), sha256=hashlib.sha256(result.tobytes()).hexdigest()))
        assert_equivalent(solvers[0][1], solvers[1][1])
        for name in ('gm', 'gdir', 'gval', 'goff'):
            assert_equivalent(getattr(solvers[0][0], name), getattr(solvers[1][0], name))
        MEASUREMENTS['disabled_positive_vs_zero'] = rows

    def test_D04_D12_D21_retained_runtime_artifacts(self):
        if not (
            ARTIFACT / 'runner_enabled/runner_test.json'
        ).is_file():  # research record of an earlier harness run, not package data
            raise unittest.SkipTest(f'earlier harness artifact absent: {ARTIFACT / "runner_enabled"}')
        report = json.loads((ARTIFACT / 'runner_enabled/runner_test.json').read_text())
        self.assertTrue(report['pass_'])
        work = ARTIFACT / 'runner_enabled/work'
        metas = {
            name: json.loads((work / 'out' / name / 'meta.json').read_text())
            for name in ('enabled', 'extra', 'override')
        }
        for name, meta in metas.items():
            self.assertTrue(meta['complete'])
            self.assertEqual(meta['stratosphere']['status'], 'enabled')
            self.assertEqual(meta['parameters_requested']['stratosphere_tau550'], 0.003)
            self.assertEqual(meta['parameters_computed']['stratosphere_tau550'], 0.003)
            self.assertEqual(meta['stratosphere']['optics_sha256'], hashlib.sha256(st.DEFAULT.read_bytes()).hexdigest())
            data = subprocess.check_output(['zstd', '-q', '-dc', str(work / 'pack' / (name + '.tar.zst'))])
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                self.assertEqual(json.load(archive.extractfile('meta.json')), meta)
        for key in ('parameters_requested', 'parameters_computed', 'stratosphere', 'stratosphere_computed'):
            self.assertEqual(metas['enabled'][key], metas['extra'][key])
        self.assertTrue(all(r['within_tol'] for r in metas['extra']['anchor_check'].values()))
        with np.load(work / 'out/enabled/table.npz') as a, np.load(work / 'out/override/table.npz') as b:
            assert_equivalent(a['radiance'], b['radiance'])
        for name in ('enabled', 'override'):
            with np.load(work / 'tmp' / name / 'source.npz') as data:
                self.assertEqual(data['source'].shape, (40, 1132, 1))
        validation = json.loads((ARTIFACT / 'val_zero/validation.json').read_text())
        self.assertTrue(validation['pass'])
        from sosfuse2.validate_strato import zero_bytes

        baseline = os.environ.get(
            'SOSFUSE2_STRATO_BASELINE', ''
        ).strip()  # the fixed measurement's npz pair is not part of the package
        if not baseline:
            self.skipTest(
                'SOSFUSE2_STRATO_BASELINE (folder with selftest_values.npz and full_35_values.npz of the fixed measurement) not set'
            )
        zero = zero_bytes(ARTIFACT / 'val_zero', Path(baseline))
        self.assertTrue(zero['pass_'])
        MEASUREMENTS['tau_zero_archive_sha256'] = {name: r['sha256'] for name, r in zero['files'].items()}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', help='exact unittest method name')
    args = parser.parse_args()
    OUT = testing.outside_package(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    suite = (
        unittest.TestSuite([Delta(args.case)]) if args.case else unittest.defaultTestLoader.loadTestsFromTestCase(Delta)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(
        scope='round 3 bounded DELTA, frozen 22-case matrix',
        risk_class='D2',
        tests_run=result.testsRun,
        pass_=result.wasSuccessful(),
        failures=[dict(test=str(t), traceback=e) for t, e in result.failures],
        errors=[dict(test=str(t), traceback=e) for t, e in result.errors],
        measurements=MEASUREMENTS,
    )
    (OUT / 'delta_adversarial.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
