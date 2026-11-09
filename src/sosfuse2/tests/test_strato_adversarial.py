#!/usr/bin/env python3
"""Spec-first independent round-1 tests; never edits production inputs.

Run: python -B -m sosfuse2.tests.test_strato_adversarial --out <dir>
"""

from sosfuse2 import paths
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import numpy as np
from sosfuse2 import stratosphere as st, mie, reference, testing

try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol

OUT = testing.scratch_parent() / 'strato/astra'


def _strato_colors(test):
    """strato_colors (colour figures, Display P3 ICC) is a sky-sos research module (scripts/sosfuse2_dev), not in the package."""
    try:
        from sosfuse2 import strato_colors as color
    except ImportError:
        test.skipTest('strato_colors is a sky-sos research module (scripts/sosfuse2_dev)')
    return color


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', type=Path, default=OUT)
    args, remaining = parser.parse_known_args()
    OUT = args.out
    sys.argv = [sys.argv[0], *remaining]
OUT.mkdir(parents=True, exist_ok=True)
MEASUREMENTS = {}


def state(tau=0.03, **cfg):
    return {'parameters': {'stratosphere_tau550': tau}, 'stratosphere': dict(status='enabled', **cfg)}


def independent_density(z, center=21.0, sigma=3.0):
    # Analytic truncated Gaussian normalization, independent of production trapz.
    norm = (
        sigma
        * math.sqrt(math.pi / 2)
        * (math.erf((80 - center) / (math.sqrt(2) * sigma)) - math.erf(-center / (math.sqrt(2) * sigma)))
    )
    return np.exp(-0.5 * ((np.asarray(z) - center) / sigma) ** 2) / norm


def table(chi=None):
    if chi is None:
        chi = [1.0, 0.6, 0.2] + [0.0] * 30
    return {
        'version': 1,
        'wavelengths_nm': [400.0, 550.0, 800.0],
        'rows': [
            dict(wavelength_nm=w, C_ext_um2=2.0, C_sca_um2=2.0, SSA=1.0, g=chi[1] / 3, chi_l=list(chi))
            for w in (400.0, 550.0, 800.0)
        ],
    }


def fixture(name, value):
    dest = OUT / (name + '.json')
    dest.write_text(json.dumps(value))
    return dest


def accurate_legendre_rule(n):
    """Independent Newton roots/derivative weights avoid eigensolver weight error.

    The initial 768-node numpy rule produced ~3e-11 weight-integration error
    on sharply forward-peaked x=50 phase. This changes the test integrator,
    not the frozen 1e-11 physical assertion; 128/256/768 Newton rules agree.
    """
    x = np.cos(np.pi * (np.arange(1, n + 1, dtype=np.longdouble) - 0.25) / (n + 0.5))
    for _ in range(12):
        p0 = np.ones(n, dtype=np.longdouble)
        p1 = x.copy()
        for l in range(2, n + 1):
            p0, p1 = p1, ((2 * l - 1) * x * p1 - (l - 1) * p0) / l
        derivative = n * (x * p1 - p0) / (x * x - 1)
        x = x - p1 / derivative
    return x, 2 / ((1 - x * x) * derivative * derivative)


class OpticalContract(unittest.TestCase):
    def test_A01_A02_zero_short_circuit(self):
        for tau in (0.0, -0.0):
            with (
                self.subTest(tau=tau),
                patch.object(st, 'profile_config', side_effect=AssertionError('profile touched')),
                patch.object(st, 'load_table', side_effect=AssertionError('table touched')),
                patch.object(st.np, 'asarray', side_effect=AssertionError('optical array allocated')),
            ):
                self.assertIsNone(
                    st.build_optics(
                        state(tau, profile='wrong', sigma_km=-1, optics_table='/missing'), object(), object()
                    )
                )
        self.assertIsNone(st.build_optics({}, [], []))
        self.assertEqual(st.source_size(type('S', (), {})()), 571)

    def test_A03_invalid_tau(self):
        for tau in (-1.0, float('nan'), float('inf'), -float('inf')):
            with self.subTest(tau=tau), self.assertRaisesRegex(ValueError, '^E_STRATO_TAU'):
                st.build_optics(state(tau), [0.0, 21.0, 80.0], [550.0])
        self.assertIsNotNone(st.build_optics(state(np.nextafter(0.0, 1.0)), [0.0, 21.0, 80.0], [550.0]))

    def test_A04_gaussian_normalization(self):
        errors = {}
        for center, sigma in ((21.0, 3.0), (21.0, 1.0), (40.0, 5.0)):
            with self.subTest(center=center, sigma=sigma):
                p = st.profile_config(state(center_km=center, sigma_km=sigma))
                z = np.linspace(max(0.0, center - 12 * sigma), min(80.0, center + 12 * sigma), 120001)
                q = st.density(z, p)
                err = abs(np.trapezoid(q, z) - 1.0)
                errors[str((center, sigma))] = err
                self.assertLessEqual(err, 1e-8)
                np.testing.assert_allclose(q, independent_density(z, center, sigma), rtol=1e-8, atol=1e-12)
        MEASUREMENTS['gaussian_normalization_errors'] = errors

    def test_A05_profile_rejections(self):
        cases = [
            ({'profile': 'exponential'}, 'E_STRATO_PROFILE'),
            *[
                ({'sigma_km': v}, 'E_STRATO_PROFILE_UNSUPPORTED' if np.isfinite(v) else 'E_STRATO_PROFILE_PARAMETERS')
                for v in (0.0, -1.0, np.nan, np.inf)
            ],
            *[
                ({'center_km': v}, 'E_STRATO_PROFILE_UNSUPPORTED' if np.isfinite(v) else 'E_STRATO_PROFILE_PARAMETERS')
                for v in (-1.0, 81.0, np.nan, np.inf)
            ],
            ({'sigma_km': 0.003}, 'E_STRATO_PROFILE_UNSUPPORTED'),
        ]
        for cfg, code in cases:
            with self.subTest(cfg=cfg), self.assertRaisesRegex(ValueError, '^' + code):
                st.build_optics(state(**cfg), [0.0, 21.0, 80.0], [550.0])

    def test_A05_extreme_sigma_stable_error(self):
        with self.assertRaisesRegex(ValueError, '^E_STRATO_PROFILE_UNSUPPORTED'):
            st.profile_config(state(sigma_km=1e-320))

    def test_A06_corrupt_table(self):
        mutations = {
            'missing': lambda d: d.pop('rows'),
            'version': lambda d: d.update(version=2),
            'negative_ext': lambda d: d['rows'][0].update(C_ext_um2=-1),
            'negative_sca': lambda d: d['rows'][0].update(C_sca_um2=-1),
            'gain': lambda d: d['rows'][0].update(C_sca_um2=3),
            'nan': lambda d: d['rows'][0].update(SSA=float('nan')),
            'ratio': lambda d: d['rows'][0].update(SSA=0.3),
            'chi0': lambda d: d['rows'][0]['chi_l'].__setitem__(0, 0.5),
            'chi1_bound': lambda d: d['rows'][0]['chi_l'].__setitem__(1, 4.0),
            'g': lambda d: d['rows'][0].update(g=0.8),
            'L31': lambda d: d['rows'][0]['chi_l'].pop(),
            'duplicate': lambda d: d['wavelengths_nm'].__setitem__(1, 400.0),
            'row_mismatch': lambda d: d['rows'][0].update(wavelength_nm=401.0),
        }
        for name, mutate in mutations.items():
            d = table()
            mutate(d)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, '^E_STRATO_TABLE_SCHEMA'):
                st.load_table(fixture('bad_' + name, d))
        p = OUT / 'corrupt.json'
        p.write_text('{')
        for bad in (p, OUT / 'does_not_exist.json'):
            with self.subTest(path=bad), self.assertRaisesRegex(ValueError, '^E_STRATO_TABLE_READ'):
                st.load_table(bad)

    def test_A06_scalar_table_fields_reject_arrays(self):
        for field in ('C_ext_um2', 'C_sca_um2', 'SSA', 'g'):
            d = table()
            for row in d['rows']:
                row[field] = [row[field]]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, '^E_STRATO_TABLE_SCHEMA'):
                st.load_table(fixture('rank_' + field, d))

    def test_A06_wavelength_range(self):
        p = fixture('valid', table())
        for wl in ([399.0], [801.0], [np.nan], [[550.0]]):
            with self.subTest(wl=wl), self.assertRaisesRegex(ValueError, '^E_STRATO_WAVELENGTH_RANGE'):
                st.build_optics(state(optics_table=str(p)), [0.0, 21.0, 80.0], wl)
        opt = st.build_optics(state(optics_table=str(p)), [0.0, 21.0, 80.0], [400.0, 550.0, 800.0])
        self.assertEqual(opt['beta_e'].shape, (3, 3))
        self.assertEqual(opt['chi'].shape, (33, 3))
        np.testing.assert_allclose(opt['ext'], 0.03, rtol=0, atol=0)

    def test_A07_A16_L32_phase_and_eigen(self):
        chi = [1.0, 0.6, 0.2] + [0.0] * 29 + [0.03]
        p = fixture('L32', table(chi))
        opt = st.build_optics(state(optics_table=str(p)), [0.0, 21.0, 80.0], [550.0])
        np.testing.assert_allclose(opt['eigen'][:, 0], np.array(chi) / (2 * np.arange(33) + 1), rtol=0, atol=0)
        self.assertEqual(st.eigen_by_sh(opt).shape, (561, 1))
        mu = np.array([-1.0, -0.72, 0.1, 0.65, 1.0], 'f4')
        expected = np.polynomial.legendre.legval(mu.astype('f8'), chi) / (4 * np.pi)
        np.testing.assert_allclose(
            reference.mie_phase(mu, opt['chi'].astype('f4'))[:, 0], expected, rtol=8e-7, atol=1e-8
        )
        mu, w = np.polynomial.legendre.leggauss(128)
        phase = np.polynomial.legendre.legval(mu, chi)
        self.assertLess(abs(w @ phase / 2 - 1), 1e-12)
        self.assertLess(abs((w * mu) @ phase / 2 - 0.2), 1e-12)

    def test_A08_rayleigh_and_A09_large(self):
        m = 1.43 + 0j
        values = np.array([mie.efficiencies(x, m)[1] for x in (0.001, 0.002)])
        theory = (8 / 3) * np.array([0.001, 0.002]) ** 4 * abs((m * m - 1) / (m * m + 2)) ** 2
        error = float(max(abs(values / theory - 1)))
        self.assertLess(error, 1e-5)
        self.assertLess(abs(np.log2(values[1] / values[0]) - 4), 1e-5)
        large = {str(x): mie.efficiencies(x, m)[0] for x in (1000.0, 2000.0, 4000.0)}
        self.assertLess(max(abs(v - 2) for v in large.values()), 0.04)
        MEASUREMENTS['mie_limits'] = dict(rayleigh_relative=error, large_Qext=large)

    def test_A10_A11_passivity_normalization_g(self):
        mu, w = accurate_legendre_rule(768)
        residuals = []
        for x in (0.001, 0.1, 1.0, 2.5, 10.0, 50.0):
            for k in (0.0, 0.001, 0.1):
                with self.subTest(x=x, k=k):
                    qe, qs, g, a, b = mie.efficiencies(x, 1.43 + 1j * k)
                    self.assertTrue(np.isfinite([qe, qs, g]).all())
                    self.assertGreaterEqual(qe, 0.0)
                    self.assertGreaterEqual(qs, 0.0)
                    self.assertLessEqual(qs, qe * (1 + 1e-10))
                    phase = mie.phase(mu, x, qs, a, b)
                    normerr = abs(w @ phase / 2 - 1)
                    gerr = abs((w * mu) @ phase / 2 - g)
                    self.assertLess(normerr, 1e-11)
                    self.assertLess(gerr, 1e-11)
                    if k == 0:
                        self.assertLess(abs(qs / qe - 1), 1e-10)
                    residuals.append([x, k, float(normerr), float(gerr)])
        MEASUREMENTS['mie_angular_residuals'] = residuals

    def test_A11_independent_quadrature_consistency(self):
        values = []
        qe, qs, g, a, b = mie.efficiencies(50.0, 1.43 + 0.1j)
        for nodes in (128, 256, 768):
            mu, w = accurate_legendre_rule(nodes)
            p = mie.phase(mu, 50.0, qs, a, b)
            values.append([float(w @ p / 2), float((w * mu) @ p / 2)])
        np.testing.assert_allclose(values, np.tile([1.0, g], (3, 1)), rtol=0, atol=1e-11)
        MEASUREMENTS['independent_quadrature_128_256_768'] = values

    def test_A09_ensemble_convergence_and_supplied_table(self):
        coarse = mie.ensemble(550.0, size_nodes=1024, angle_nodes=512)
        fine = mie.ensemble(550.0, size_nodes=1536, angle_nodes=768)
        errors = {key: abs(coarse[key] / fine[key] - 1) for key in ('C_ext_um2', 'C_sca_um2', 'g')}
        errors['chi_max_absolute'] = float(np.max(abs(np.array(coarse['chi_l']) - fine['chi_l'])))
        for key, error in errors.items():
            with self.subTest(metric=key):
                self.assertLess(error, 1e-5)
        supplied = next(row for row in json.loads(st.DEFAULT.read_text())['rows'] if row['wavelength_nm'] == 550.0)
        for key in ('C_ext_um2', 'C_sca_um2', 'g'):
            self.assertLess(abs(supplied[key] / fine[key] - 1), 1e-5)
        self.assertLess(np.max(abs(np.array(supplied['chi_l']) - fine['chi_l'])), 1e-5)
        MEASUREMENTS['ensemble_convergence'] = errors

    def test_A17_conservative_source(self):
        z = np.linspace(0, 80, 16001)
        opt = st.build_optics(state(), z, [550.0])
        self.assertEqual(float(np.trapezoid((opt['beta_e'] - opt['beta_s'])[:, 0], z)), 0.0)
        mu, w = np.polynomial.legendre.leggauss(128)
        self.assertLess(abs(w @ np.polynomial.legendre.legval(mu, opt['chi'][:, 0]) / 2 - 1), 1e-12)

    def test_A19_failure_does_not_poison_next_build(self):
        before = st.build_optics(state(), [0.0, 21.0, 80.0], [550.0])
        with self.assertRaises(ValueError):
            st.build_optics(state(optics_table=str(OUT / 'absent')), [0.0, 21.0, 80.0], [550.0])
        after = st.build_optics(state(), [0.0, 21.0, 80.0], [550.0])
        for key in ('beta_e', 'beta_s', 'chi', 'eigen', 'density'):
            assert_equivalent(before[key], after[key])


class TransportContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sosfuse2.geometry import create, normal_state, TransientGeometry
        from sosfuse2.fused import Solver

        cfg = copy.deepcopy(normal_state())
        cfg['stratosphere']['status'] = 'enabled'
        cfg['parameters'].update(stratosphere_tau550=0.03)
        cls.s, cls.scope, _ = create(OUT / 'geometry', 'selftest', [550.0], state=cfg)
        cls.solver = Solver(cls.s, TransientGeometry(cls.s, cls.scope))
        cls.old = cls.scope.b.sh.old
        cls.c = cls.s.kernel.setup(cls.s, cls.s.geo.query_n)

    def pack(self, sun=2.0, el=20.0, az=0.0, altitude=0.1):
        # Construct global frame directly, independent of production frame().
        hs, ev, av = np.deg2rad([sun, el, az])
        up = np.array([np.cos(hs), 0.0, np.sin(hs)])
        ex = np.array([-np.sin(hs), 0.0, np.cos(hs)])
        ey = np.cross(up, ex)
        p = (6371.0 + altitude) * up
        d = np.sin(ev) * up + np.cos(ev) * (np.cos(av) * ex + np.sin(av) * ey)
        return self.s.geo.pack(p[None, :], d[None, :], self.s.geo.query_n)

    def independent_path(self, p, d, n=32001):
        # Nearest positive sphere endpoint, with opaque ground.
        dot = float(p @ d)
        r2 = float(p @ p)
        top = -dot + math.sqrt(dot * dot + 6451.0**2 - r2)
        disc = dot * dot + 6371.0**2 - r2
        near = -dot - math.sqrt(max(disc, 0.0))
        length = near if disc >= 0 and near > 1e-8 else top
        t = np.linspace(0.0, length, n)
        q = p + t[:, None] * d
        return t, q, independent_density(np.linalg.norm(q, axis=1) - 6371.0)

    def clean_constants(self):
        c = copy.deepcopy({k: v for k, v in self.c.items() if k != 'strato_old'}, {id(self.old): self.old})
        c['strato_old'] = self.old
        for key in ('ext', 'asca', 'gm', 'gv'):
            c[key] = np.zeros_like(c[key])
        return c

    def test_A04_A15_A18_endpoint_columns_current_rays(self):
        errors = []
        for altitude, el in ((0.1, 90.0), (0.1, 20.0), (35.0, -90.0), (25.0, 90.0), (5.0, -90.0)):
            g = self.pack(altitude=altitude, el=el)
            a = st.columns(g['p'], g['d'], self.old, self.s.strato['profile'], 24)[0]
            t, q, den = self.independent_path(g['p'][0], g['d'][0])
            b = np.trapezoid(den, t)
            errors.append([altitude, el, float(a), float(b)])
            self.assertLess(abs(a - b), 1e-7 * max(1, abs(b)))
        # Repeat original geometry after other queries: no stale column cache.
        g = self.pack(altitude=0.1, el=90.0)
        self.assertEqual(st.columns(g['p'], g['d'], self.old, self.s.strato['profile'], 24)[0], errors[0][2])
        MEASUREMENTS['endpoint_columns'] = errors

    def test_A14_independent_thin_integral(self):
        g = self.pack()
        c = self.clean_constants()
        c['strato_ext'] = np.array([1e-6], 'f4')
        c['strato_sca'] = np.array([1e-6], 'f4')
        zero = np.zeros((self.s.geo.ns, 1132, 1), 'f4')
        actual = float(reference.transport(g, c, zero, True)[0, 0, :3].sum(dtype='f8'))
        independent = []
        for n in (16001, 32001):
            t, q, den = self.independent_path(g['p'][0], g['d'][0], n)
            prefix = np.r_[0.0, np.cumsum((den[1:] + den[:-1]) * np.diff(t) / 2)]
            end = np.sqrt(6451.0**2 - q[:, 0] ** 2 - q[:, 1] ** 2) - q[:, 2]
            node, weight = np.polynomial.legendre.leggauss(128)
            node = (node + 1) / 2
            weight = weight / 2
            col = np.empty(n)
            for start in range(0, n, 512):
                point = q[start : start + 512]
                dist = end[start : start + 512]
                solar = point[:, None, :] + dist[:, None, None] * node[None, :, None] * np.array([0.0, 0.0, 1.0])
                col[start : start + len(point)] = dist * (
                    independent_density(np.linalg.norm(solar, axis=2) - 6371.0) @ weight
                )
            phase = np.polynomial.legendre.legval(g['d'][0, 2], self.s.strato['chi'][:, 0]) / (4 * np.pi)
            independent.append(float(np.trapezoid(den * 1e-6 * phase * np.exp(-1e-6 * (prefix + col)), t)))
        residual = abs(actual - independent[-1]) / abs(independent[-1])
        MEASUREMENTS['thin'] = dict(actual=actual, independent=independent, relative=residual)
        self.assertLessEqual(residual, 0.01)

    def test_A12_FP32_common_operator_additivity(self):
        from sosfuse2.validate_strato import additivity

        result = additivity(self.s, self.scope)
        MEASUREMENTS['V4_production'] = result
        self.assertGreaterEqual(result['rays'], 64)
        self.assertEqual(result['element_failures'], 0)
        self.assertTrue(result['scattering_2x_within_tol'])
        self.assertTrue(result['pass_'])

    def test_A15_boundary_layer_attenuation(self):
        # A pure boundary signal isolates ground->view extinction.
        c = self.clean_constants()
        c['strato_sca'] = np.zeros(1, 'f4')
        field = np.zeros((self.s.geo.ns, 1132, 1), 'f4')
        boundary = np.ones((len(self.s.geo.chi), 1), 'f8')
        for altitude in (5.0, 25.0, 35.0):
            with self.subTest(altitude=altitude):
                g = self.pack(el=-90.0, altitude=altitude)
                value = reference.transport(g, c, field, False, boundary=boundary)
                t, q, den = self.independent_path(g['p'][0], g['d'][0])
                expected = np.exp(-0.03 * np.trapezoid(den, t))
                actual = float(value[0, 0, :3].sum(dtype='f8'))
                self.assertLess(abs(actual - expected), 2e-7)

    def test_A15_sun_extinction_has_independent_effect(self):
        # Changing only the sidecar sun column must change direct strat light by
        # exp(-additional column*ext), with view attenuation held fixed.
        c = self.clean_constants()
        c['strato_ext'] = np.array([0.03], 'f4')
        g = self.pack()
        zero = np.zeros((self.s.geo.ns, 1132, 1), 'f4')
        a = reference.transport(g, c, zero, True)
        extra = dict(c)
        extra['strato_sun_col'] = c['strato_sun_col'] + np.float32(2.0)
        b = reference.transport(g, extra, zero, True)
        np.testing.assert_allclose(b, a * np.exp(-0.06), rtol=5e-7, atol=1e-12)

    def test_A16_ground_uses_L32_not_HG(self):
        # Extinguish direct sunlight, use a synthetic single-direction ground
        # pool, then change only L32 while retaining g. Ground radiance must react.
        c = self.clean_constants()
        c['strato_sun_col'] = np.full_like(c['strato_sun_col'], 1e20)
        c['gd'] = np.broadcast_to(np.array([[[0.0, 0.0, 1.0]]], 'f4'), (self.s.geo.ns, 1, 3)).copy()
        c['gv'] = np.ones((self.s.geo.ns, 1, 1), 'f4')
        c['strato_chi'] = np.zeros((33, 1), 'f4')
        c['strato_chi'][0] = 1
        g = self.pack(el=90.0)
        zero = np.zeros((self.s.geo.ns, 1132, 1), 'f4')
        a = reference.transport(g, c, zero, True)
        changed = dict(c)
        changed['strato_chi'] = c['strato_chi'].copy()
        changed['strato_chi'][32] = 0.5
        b = reference.transport(g, changed, zero, True)
        np.testing.assert_allclose(b, 1.5 * a, rtol=2e-5, atol=1e-10)
        self.assertGreater(float(np.max(a)), 0.0)

    def test_A18_attach_idempotent(self):
        arrays = {key: getattr(self.s, key).copy() for key in ('gm', 'gval', 'gdir', 'goff')}
        optics = st.attach(self.s, self.scope)
        self.assertIs(optics, self.s.strato)
        for key, value in arrays.items():
            np.testing.assert_array_equal(getattr(self.s, key), value)

    @testing.known_failure(
        'A18',
        'E_SPECTRAL_UNRESOLVED: profile_mean block without a state q identity; assertion predates the P2 A-prime default, uninvestigated',
    )
    def test_A18_bounded_endpoint_sidecar_and_profile_identity(self):
        from types import SimpleNamespace

        heights = np.array([50.0, 0.0, 25.0, 35.0])
        offset = np.array([0, 0, 1, 2049, 4097], 'i8')
        starts = np.column_stack([np.zeros(4), np.zeros(4), 6371.0 + heights])
        dirs = np.tile([0.0, 0.0, 1.0], (4097, 1))
        bundle = SimpleNamespace(rays=len(dirs), arrays={'starts': starts, 'cell_off': offset, 'dirs': dirs})
        original = st.columns
        sizes = []

        def observed(p, d, *args):
            sizes.append(len(d))
            return original(p, d, *args)

        with patch.object(st, 'columns', side_effect=observed):
            actual = self.s.kernel.strato_end_columns(bundle, self.c)
        self.assertEqual(sizes, [2048, 2048, 1])
        starts_repeated = np.repeat(starts, np.diff(offset), axis=0)
        expected = original(starts_repeated, dirs, self.old, self.s.strato['profile'], 24).astype('f4')
        np.testing.assert_array_equal(actual, expected)
        with patch.object(st, 'columns', side_effect=AssertionError('unchanged immutable bundle should reuse sidecar')):
            self.assertIs(self.s.kernel.strato_end_columns(bundle, self.c), actual)
        changed = dict(self.c)
        changed['strato_profile'] = st.profile_config(state(center_km=30.0))
        other = self.s.kernel.strato_end_columns(bundle, changed)
        self.assertGreater(float(np.max(abs(other - actual))), 0.1)

    def test_A12_strat_SH_block_distinct_and_L32_present(self):
        c = self.clean_constants()
        c['strato_ext'] = np.zeros(1, 'f4')
        c['ha'] = np.ones_like(c['ha'])
        c['strato_density'] = np.ones_like(c['strato_density'])
        g = self.pack(el=10.0, az=90.0)
        for k in (0, 2, 560):
            with self.subTest(coefficient=k):
                a = np.zeros((self.s.geo.ns, 1132, 1), 'f4')
                a[:, 10 + k] = 1
                b = np.zeros_like(a)
                b[:, 571 + k] = 1
                trop = reference.transport(g, c, a, False)
                strat = reference.transport(g, c, b, False)
                self.assertGreater(float(np.max(abs(strat))), 0.0)
                np.testing.assert_array_equal(trop, strat)

    def test_A12_apply_uses_separate_eigenvalues(self):
        # Fixed artificial moments isolate the public source contraction from
        # transport interpolation; each component has its own known phase.
        from sosfuse2.fused import Solver

        moment = np.arange(1, self.s.geo.ns * 565 + 1, dtype='f8').reshape(self.s.geo.ns, 565, 1) * 1e-6

        class FixedGeometry:
            def evaluate(inner, s, field, phase=0, **kwargs):
                flux = np.zeros((len(s.geo.chi), 1), 'f8')
                return (moment.copy() if phase == 0 else np.zeros_like(moment)), flux

        solver = Solver(self.s, FixedGeometry())
        solver.first_moments = np.zeros_like(moment)
        solver.first_flux = np.zeros((len(self.s.geo.chi), 1), 'f8')
        actual = solver.apply(np.zeros((self.s.geo.ns, 1132, 1), 'f4'))
        levels = np.array([l for l in range(33) for m in range(l + 1)])
        eigen = self.s.strato['chi'][levels] / (2 * levels[:, None] + 1)
        expected = (
            moment[:, 4:] * eigen.astype('f4')[None, :, :] * self.s.strato['sca'].astype('f4')[None, None, :]
        ).astype('f4')
        np.testing.assert_array_equal(actual[:, 571:], expected)
        trop = (moment[:, 4:] * self.s.af.astype('f4')[None, :, None] * self.s.asca.astype('f4')[None, None, :]).astype(
            'f4'
        )
        np.testing.assert_array_equal(actual[:, 10:571], trop)

    def test_A15_Rayleigh_ground_recomputed_with_layer(self):
        # Set column coefficients to zero on a fresh solver with unchanged
        # scattering strengths: attach must recover the original ground tensor.
        from sosfuse2.geometry import create, normal_state

        cfg = copy.deepcopy(normal_state())
        cfg['stratosphere']['status'] = 'enabled'
        cfg['parameters']['stratosphere_tau550'] = 0.03
        # create() now eagerly attaches. Bypass only that call to preserve the
        # pre-attach fixture whose ground tensor this test must compare.
        from sosfuse2 import density

        attach_density = density.attach
        # With strato attach bypassed, also disable its deferred-ground optimization:
        # the control must contain the corrected NO2/O4 columns before comparison.
        with (
            patch.object(st, 'attach', return_value=None),
            patch.object(
                density, 'attach', side_effect=lambda s, scope, **kw: attach_density(s, scope, defer_ground=False)
            ),
        ):
            s, scope, _ = create(OUT / 'ground_control', 'selftest', [550.0], state=cfg)
        before = s.gm.copy()
        build = st.build_optics

        def no_extinction(*args, **kwargs):
            opt = build(*args, **kwargs)
            opt['ext'][:] = 0.0
            return opt

        with patch.object(st, 'build_optics', side_effect=no_extinction):
            st.attach(s, scope)
        np.testing.assert_allclose(s.gm, before, rtol=2e-7, atol=1e-12)
        base = before[:, :3].sum(axis=1, dtype='f8')
        attenuated = self.s.gm[:, :3].sum(axis=1, dtype='f8')
        self.assertTrue(np.all(attenuated <= base + 1e-12))
        self.assertGreater(float(np.max(base - attenuated)), 0.0)

    def test_A19_failed_attach_is_retryable(self):
        from sosfuse2.geometry import create, normal_state

        cfg = copy.deepcopy(normal_state())
        cfg['stratosphere']['status'] = 'enabled'
        cfg['parameters']['stratosphere_tau550'] = 0.03
        with patch.object(st, 'attach', return_value=None):
            s, scope, _ = create(OUT / 'retry', 'selftest', [550.0], state=cfg)
        before = {key: getattr(s, key).copy() for key in ('gm', 'gval', 'gdir', 'goff')}
        # _ground_geometry is memoized in-process regardless of STRATO_ATTACH_CACHE (2026-11-03); an empty memo makes the
        # injected solar_grid failure reach the compute path, and patch.dict restores the memo for the other cases.
        with patch.dict(st._GEOM_MEMO, clear=True):
            with patch.object(st, 'solar_grid', side_effect=RuntimeError('injected sidecar failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected sidecar failure'):
                    st.attach(s, scope)
            self.assertFalse(hasattr(s, 'strato'))
            for key, value in before.items():
                np.testing.assert_array_equal(getattr(s, key), value)
            st.attach(s, scope)
        for key in ('gm', 'gval', 'gdir', 'goff'):
            np.testing.assert_array_equal(getattr(s, key), getattr(self.s, key))


class BaselineContract(unittest.TestCase):
    def test_A20_frozen_math_tokens_all_declared_variants(self):
        import subprocess, re, itertools

        current = paths.PKG / 'kernel.cu'
        baseline = paths.ASSETS / 'kernel_strato_off_baseline.cu'
        self.assertTrue(baseline.exists(), 'original CUDA fixture required; no silent skip')
        testing.require_tool(
            'clang++'
        )  # the fixture is required; the compiler is an environment dependency (CHECKS.md tier 1)
        flags = [
            [],
            ['SH_AUX'],
            ['LANE_GROUND'],
            ['LM_LUT'],
            ['PHASE2_NOBARRIER'],
            ['SPECIALIZE'],
            ['SPECIALIZE', 'SH_AUX', 'LANE_GROUND', 'LM_LUT', 'PHASE2_NOBARRIER'],
        ]
        for waves, variant in itertools.product((1, 15), flags):
            with self.subTest(waves=waves, variant=variant):
                command = ['clang++', '-E', '-P', '-x', 'c++', f'-DFW={waves}', *[f'-D{f}' for f in variant]]
                if 'SPECIALIZE' in variant:
                    command += ['-DK_FIRST=1', '-DK_PHASE=2', '-DK_BOUNDARY=1']
                before = subprocess.check_output([*command, str(baseline)], text=True)
                after = subprocess.check_output([*command, str(current)], text=True)

                # B2 changes the transport layout (3 -> 5 components); the frozen
                # file remains untouched. Compare unchanged arithmetic helpers,
                # with transport covered by E1 component tests and G-P regression.
                def body(text, name):
                    start = re.search(r'\b' + name + r'\([^;{}]*\)\s*\{', text).start()
                    pos = text.index('{', start)
                    depth = 1
                    end = pos + 1
                    while depth:
                        depth += (text[end] == '{') - (text[end] == '}')
                        end += 1
                    return re.findall(r'\w+|[^\s]', text[start:end])

                for name in ('ds', 'da', 'dm', 'dr', 'dv', 'clip', 'hg'):
                    self.assertEqual(body(before, name), body(after, name), name)

    # test_A01_existing_regression_and_baseline_bytes moved to scripts/sosfuse2_dev/test_a01_baseline_bytes.py (sky-sos only): it
    # compared a harness run with another session's scratchpad folder by absolute path — not reproducible outside that machine
    # (public split S2). G-P/G-T/G-F (gate.py) cover the regression it stood for.


class ColorContract(unittest.TestCase):
    def test_A18_color_source_cache_tracks_table_contents(self):
        # Exercise the real run()/persistent-cache decision with cheap optical
        # solver doubles. A fresh run after an input-table edit must solve anew.
        from types import SimpleNamespace
        import shutil

        color = _strato_colors(self)
        folder = OUT / 'cache_probe'
        if folder.exists():
            shutil.rmtree(folder)
        path = fixture('cache_table', table())
        cfg = state(optics_table=str(path))
        cfg['id'] = 'cache_probe'
        calls = []

        def make_solver(output, grid, wavelengths, state):
            content = Path(state['stratosphere']['optics_table']).read_bytes()
            scale = json.loads(content)['rows'][0]['C_ext_um2']
            s = SimpleNamespace(
                scale=scale,
                geo=SimpleNamespace(ns=1, h=np.array([21.0]), chi=np.array([0.0])),
                kernel=SimpleNamespace(host=lambda x: x),
                strato={'meta': {'optics_sha256': hashlib.sha256(content).hexdigest()}},
            )

            def query(hs, el, az, field, first):
                out = np.zeros((*el.shape, 15, 2))
                out[..., 0] = field[0, 0, ::2] * 0.001
                return out

            s.query = query
            return s, None, {}

        class FakeSolver:
            def __init__(inner, s, geometry):
                inner.s = s

            def solve(inner, *args):
                calls.append(inner.s.scale)
                return np.full((1, 1132, 15), inner.s.scale, 'f4'), {}

            def initialize(inner):
                pass

        with (
            patch.object(color, 'build', return_value={'states': [cfg]}),
            patch.object(color, 'create', side_effect=make_solver),
            patch.object(color, 'TransientGeometry', return_value=None),
            patch.object(color, 'Solver', FakeSolver),
            patch.object(color, 'plot', return_value=[]),
        ):
            color.run(folder)
            changed = table()
            for row in changed['rows']:
                row['C_ext_um2'] = row['C_sca_um2'] = 4.0
            path.write_text(json.dumps(changed))
            color.run(folder)
        self.assertEqual(calls, [2.0, 4.0], 'optics content changed: cached source must be invalidated')

    def test_A21_local_CIE_provenance_and_bin_integrals(self):
        color = _strato_colors(self)
        wl, cmf, doc, path = color.cie_table()
        library = Path(doc['library']).read_bytes()
        header = Path(doc['header']).read_bytes()
        self.assertEqual(hashlib.sha256(library).hexdigest(), doc['library_sha256'])
        self.assertEqual(hashlib.sha256(header).hexdigest(), doc['header_sha256'])
        self.assertIn(b'ptr + MI_CIE_SAMPLES * 2', header)
        self.assertIn(b'#define MI_CIE_SAMPLES       95', header)
        extracted = np.frombuffer(library, dtype='<f4', offset=doc['byte_offset'], count=285).reshape(3, 95).T
        np.testing.assert_array_equal(extracted, cmf)
        np.testing.assert_array_equal(wl, np.arange(360.0, 831.0, 5.0))
        np.testing.assert_allclose(cmf[39], [0.5120501, 1.0, 0.005749999], rtol=0, atol=1e-7)
        self.assertLess(abs(np.trapezoid(cmf[:, 1], wl) - 106.856895), 0.01)
        weights, _ = color.color_weights()
        expected = np.zeros((15, 3))
        edges = np.linspace(360.0, 830.0, 16)
        # Integrate the linear primitive on every overlap of a native segment
        # and a spectral bin, independent of production interp/trapezoid logic.
        for b, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            for j in range(94):
                lo = max(left, wl[j])
                hi = min(right, wl[j + 1])
                if hi <= lo:
                    continue
                slope = (cmf[j + 1] - cmf[j]) / (wl[j + 1] - wl[j])
                expected[b] += (hi - lo) * cmf[j] + 0.5 * ((hi - wl[j]) ** 2 - (lo - wl[j]) ** 2) * slope
        expected /= doc['normalization']
        np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-14)
        MEASUREMENTS['CIE_provenance'] = dict(
            library_sha256=doc['library_sha256'],
            table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            Y_integral=float(np.trapezoid(cmf[:, 1], wl)),
            max_bin_error=float(np.max(abs(weights - expected))),
        )

    def test_A21_CIE_purity_known_boundary_fractions(self):
        color = _strato_colors(self)
        wl, cmf, _, _ = color.cie_table()
        locus = color.xy_of(cmf)
        boundary = [locus[int((wave - 360) / 5)] for wave in (450, 555, 600)]
        boundary.append((locus[0] + locus[-1]) / 2)
        for edge in boundary:
            for fraction in (0.25, 0.5, 1.0):
                xy = color.D65 + fraction * (edge - color.D65)
                xyz = np.r_[xy, 1.0 - xy.sum()]
                purity, _, _ = color.purity_and_hue(xyz, None)
                self.assertAlmostEqual(float(purity), fraction, places=9)

    def test_A21_exact_state_angle_spectral_matrix(self):
        color = _strato_colors(self)
        states = color.build()['states']
        pairs = [(s['parameters']['aod550'], s['parameters']['stratosphere_tau550']) for s in states]
        self.assertEqual(pairs, [(a, t) for a in (0.02, 0.15) for t in (0.0, 0.003, 0.01, 0.03)] + [(0.15, 0.1)])
        np.testing.assert_array_equal(color.HS, [-2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0])
        np.testing.assert_array_equal(color.EL, [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(color.AZ, [0.0, 30.0, 60.0])
        weights, _ = color.color_weights()
        self.assertEqual(weights.shape, (15, 3))
        patch_area = (np.sin(np.deg2rad(45)) - np.sin(np.deg2rad(5))) * np.deg2rad(90)
        MEASUREMENTS['declared_patch_sr'] = float(patch_area)

    def test_A21_numpy_scalar_host(self):
        from sosfuse2.fused import Kernel

        value = np.trace(np.eye(2))
        self.assertEqual(float(Kernel(backend='numpy').host(value)), 2.0)

    def test_A22_zero_white_purple_and_raw_preservation(self):
        color = _strato_colors(self)
        weights, _ = color.color_weights()
        white = np.array([color.D65[0] / color.D65[1], 1.0, (1 - color.D65.sum()) / color.D65[1]])
        point = color.D65 + 0.03 * np.array([np.cos(np.deg2rad(-30)), np.sin(np.deg2rad(-30))])
        purple = np.array([point[0], point[1], 1.0 - point.sum()])
        xyz = np.stack([np.zeros(3), white, purple, purple * 1e-30])
        before = xyz.tobytes()
        uv = color.uv_of(xyz)
        p, h, yes = color.purity_and_hue(xyz, weights)
        self.assertTrue(np.isnan(uv[0]).all())
        self.assertTrue(np.isnan(p[0]))
        self.assertFalse(yes[0])
        self.assertEqual(p[1], 0.0)
        self.assertFalse(yes[1])
        self.assertTrue(yes[2])
        self.assertTrue(yes[3])
        self.assertAlmostEqual(p[2], p[3], places=13)
        color.display_p3(xyz)
        self.assertEqual(xyz.tobytes(), before)

    def test_A21_P3_ICC_and_raw_input(self):
        color = _strato_colors(self)
        from PIL import Image

        im = Image.new('RGB', (2, 2), (255, 0, 0))
        dest = OUT / 'icc_probe.png'
        color.save_p3(im, dest, 'independent ICC probe')
        with Image.open(dest) as saved:
            icc = saved.info.get('icc_profile', b'')
            self.assertGreater(len(icc), 128)
            self.assertEqual(icc[36:40], b'acsp')
            self.assertEqual(saved.info['color_space'], 'Display P3')
            self.assertEqual(icc, (Path(color.__file__).with_name('assets') / 'display_p3.icc').read_bytes())


class AggregateArtifacts(unittest.TestCase):
    def test_A21_first_order_artifact_separate_from_total_gate(self):
        import csv

        color = _strato_colors(self)
        from PIL import Image

        folder = OUT.parent / 'single_colors'
        if not (folder / 'single_color_summary.json').exists():
            self.skipTest('optional first-order aggregate has not been supplied')
        summary = json.loads((folder / 'single_color_summary.json').read_text())
        self.assertFalse(summary['V5_converged_total_pass_'])
        self.assertIn('NOT converged total', summary['scope'])
        self.assertEqual(summary['sample_count'], 972)
        self.assertEqual(summary['spectra_shape'], [9, 9, 4, 3, 15])
        states = color.build()['states']
        weights, _ = color.color_weights()
        with np.load(folder / 'colors.npz', allow_pickle=False) as a:
            self.assertEqual(a['radiance'].shape, (9, 9, 4, 3, 15))
            self.assertEqual(a['state_ids'].tolist(), [s['id'] for s in states])
            self.assertTrue(np.isfinite(a['radiance']).all())
            self.assertGreaterEqual(float(a['radiance'].min()), 0.0)
            assert_equivalent(a['XYZ'], a['radiance'] @ weights)
            xyz = a['XYZ']
            den = xyz[..., 0] + 15 * xyz[..., 1] + 3 * xyz[..., 2]
            np.testing.assert_allclose(
                a['uv_prime'], np.stack([4 * xyz[..., 0] / den, 9 * xyz[..., 1] / den], -1), rtol=0, atol=1e-14
            )
            # Independent ray/segment intersection using 2x2 linear solves.
            _, cmf, _, _ = color.cie_table()
            locus = cmf[:, :2] / cmf.sum(1)[:, None]
            edges = np.roll(locus, -1, axis=0) - locus
            expected = []
            for point in xyz.reshape(-1, 3):
                direction = point[:2] / point.sum() - color.D65
                matrix = np.stack([np.broadcast_to(direction, edges.shape), -edges], axis=-1)
                good = abs(np.linalg.det(matrix)) > 1e-15
                solution = np.linalg.solve(matrix[good], (locus[good] - color.D65)[..., None])[..., 0]
                hit = solution[(solution[:, 0] > 0) & (solution[:, 1] >= -1e-10) & (solution[:, 1] <= 1 + 1e-10), 0]
                expected.append(1.0 / hit.min())
            np.testing.assert_allclose(a['purity'].ravel(), expected, rtol=0, atol=2e-12)
            xy = xyz[..., :2] / xyz.sum(-1)[..., None]
            delta = xy - color.D65
            hue = np.degrees(np.arctan2(delta[..., 1], delta[..., 0]))
            expected_purple = (hue >= -60) & (hue <= -15) & (a['purity'] > 0)
            np.testing.assert_array_equal(a['purple'], expected_purple)
            area = np.diff(np.sin(np.deg2rad([5, 15, 25, 35, 45])))[:, None] * np.deg2rad(30.0) * np.ones((1, 3))
            np.testing.assert_allclose(a['patch_area_sr'], area, rtol=0, atol=1e-15)
            np.testing.assert_allclose(
                a['purple_patch_sr'], np.sum(expected_purple * area, axis=(-2, -1)), rtol=0, atol=1e-15
            )
            self.assertAlmostEqual(summary['footprint_sr'], float(area.sum()), places=14)
            self.assertEqual(summary['undefined_metrics'], int(np.count_nonzero(~np.isfinite(a['purity']))))
            for row, config in zip(summary['states'], states):
                self.assertEqual(row['state'], config)
                if config['parameters']['stratosphere_tau550'] > 0:
                    self.assertEqual(
                        row['stratosphere']['optics_sha256'], hashlib.sha256(st.DEFAULT.read_bytes()).hexdigest()
                    )
            MEASUREMENTS['first_order_artifact'] = dict(
                shape=list(a['radiance'].shape),
                undefined=summary['undefined_metrics'],
                footprint_sr=float(area.sum()),
                max_purity=float(a['purity'].max()),
                max_purple_sr=float(a['purple_patch_sr'].max()),
                total_V5=False,
            )
        with (folder / 'colors.csv').open(newline='') as f:
            self.assertEqual(len(list(csv.reader(f))), 973)
        for name in ('uv_trajectories_p3.png', 'purity_solid_angle_p3.png'):
            with Image.open(folder / name) as img:
                self.assertEqual(
                    img.info.get('icc_profile'),
                    (Path(color.__file__).with_name('assets') / 'display_p3.icc').read_bytes(),
                )
                self.assertEqual(img.info.get('color_space'), 'Display P3')
                self.assertIn('FIRST ORDER ONLY', img.info.get('description', ''))
                self.assertIn('converged total V5 not available', img.info.get('description', ''))

    def test_A21_full_color_artifact(self):
        import csv

        color = _strato_colors(self)
        from PIL import Image, ImageCms

        folder = Path(os.environ.get('SOSFUSE2_STRATO_COLOR_ARTIFACT', str(OUT.parent / 'colors')))
        if not (folder / 'color_summary.json').exists():
            self.skipTest('full V5 aggregate not yet supplied; completion gate remains open')
        summary = json.loads((folder / 'color_summary.json').read_text())
        self.assertTrue(summary['pass_'])
        self.assertEqual(summary['spectra_shape'], [9, 9, 4, 3, 15])
        states = color.build()['states']
        weights, _ = color.color_weights()
        with np.load(folder / 'colors.npz', allow_pickle=False) as a:
            self.assertEqual(a['radiance'].shape, (9, 9, 4, 3, 15))
            self.assertEqual(a['XYZ'].shape, (9, 9, 4, 3, 3))
            self.assertTrue(np.isfinite(a['radiance']).all())
            self.assertGreaterEqual(float(a['radiance'].min()), 0.0)
            self.assertEqual(a['state_ids'].tolist(), [s['id'] for s in states])
            assert_equivalent(a['XYZ'], a['radiance'] @ weights)
            xyz = a['XYZ']
            den = xyz[..., 0] + 15 * xyz[..., 1] + 3 * xyz[..., 2]
            expected_uv = np.stack([4 * xyz[..., 0] / den, 9 * xyz[..., 1] / den], -1)
            np.testing.assert_allclose(a['uv_prime'], expected_uv, rtol=0, atol=1e-14, equal_nan=True)
            area = np.diff(np.sin(np.deg2rad([5, 15, 25, 35, 45])))[:, None] * np.deg2rad(30.0) * np.ones((1, 3))
            np.testing.assert_allclose(a['patch_area_sr'], area, rtol=0, atol=1e-15)
            np.testing.assert_allclose(
                a['purple_patch_sr'], np.sum(a['purple'] * area, axis=(-2, -1)), rtol=0, atol=1e-15
            )
            self.assertEqual(summary['undefined_metrics'], int(np.count_nonzero(~np.isfinite(a['purity']))))
            for i, state_cfg in enumerate(states):
                sub = folder / state_cfg['id']
                stats = json.loads((sub / 'solve.json').read_text())
                self.assertTrue(stats['pass_'])
                self.assertTrue(all(stats['converged']))
                self.assertLessEqual(max(stats['history'][-1]['observer_relative']), 0.005)
                self.assertLessEqual(max(stats['history'][-1]['source_relative']), 0.005)
                optics = b''
                if state_cfg['parameters']['stratosphere_tau550'] > 0:
                    path = Path(state_cfg['stratosphere']['optics_table'])
                    if not path.is_absolute():
                        path = color.ROOT / path
                    optics = path.read_bytes()
                paths = [
                    Path(color.__file__).with_name(name)
                    for name in ('reference.py', 'stratosphere.py', 'fused.py', 'kernel.cu', 'geometry.py')
                ]
                identity = hashlib.sha256(
                    json.dumps(state_cfg, sort_keys=True).encode() + b''.join(p.read_bytes() for p in paths) + optics
                ).hexdigest()
                self.assertEqual(stats['identity'], identity, 'source artifact is stale relative to current inputs')
                with np.load(sub / 'radiance.npz', allow_pickle=False) as r:
                    np.testing.assert_array_equal(a['radiance'][i], r['radiance'])
                with np.load(sub / 'source.npz', allow_pickle=False) as s:
                    self.assertEqual(s['source'].shape, (40, 1132 if optics else 571, 15))
                    self.assertTrue(np.isfinite(s['source']).all())
            xyz_copy = xyz.copy()
            color.display_p3(xyz)
            np.testing.assert_array_equal(xyz, xyz_copy)
            MEASUREMENTS['aggregate_V5'] = dict(
                shape=list(a['radiance'].shape),
                undefined=summary['undefined_metrics'],
                footprint_sr=float(area.sum()),
                max_purple_sr=float(a['purple_patch_sr'].max()),
            )
        with (folder / 'colors.csv').open(newline='') as f:
            rows = list(csv.reader(f))
            self.assertEqual(len(rows), 973)
        for name in ('uv_trajectories_p3.png', 'purity_solid_angle_p3.png'):
            with Image.open(folder / name) as img:
                profile = img.info.get('icc_profile', b'')
                self.assertEqual(profile, (Path(color.__file__).with_name('assets') / 'display_p3.icc').read_bytes())
                self.assertEqual(img.info.get('color_space'), 'Display P3')


if __name__ == '__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        (OUT / 'measurements.json').write_text(json.dumps(MEASUREMENTS, indent=2) + '\n')
