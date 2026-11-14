"""Boundary, profile, method and cache rules of the opt-in endpoint table."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sosfuse2.rays import column_table as ct
from sosfuse2.atmosphere import density, stratosphere

from sosfuse2.checks.numerical_checks import assert_equivalent


def boundary_rays():
    rows = []

    def add(name, ht, sa, length):
        rows.append((name, [sa, 0.0, ct.R + ht], [1.0, 0.0, 0.0], length))

    for side in (-1, 0, 1):
        add('rho_R', side * 1e-6, 1.0, 0.07)
        add('ht_02_oracle', 0.2 + side * 1e-6, 0.0, 0.01)
        for ground in (False, True):
            ht = -0.001 if ground else 0.1
            h = (0 if ground else ht) + (80 if ground else 80 - ht) * (0.05 + side * 1e-7) ** 2
            sa = np.sqrt((h - ht) * (2 * ct.R + h + ht))
            add('x_005_ground' if ground else 'x_005_tangent', ht, sa, 0.01)
        add('L_1m', 0.200001, 0.0, 0.001 + side * 1e-9)
        add('ht_79', 79 + side * 1e-6, 0.01, 0.01)
    for ht in (0.0, 0.200001, 21.0, 79.5, 80.0):
        add('zero', ht, 0.0, 0.0)
    for ht in (0.2, 10.0, 21.0, 25.0, 40.0, 79.0):
        add('cross_tangent', ht, -0.01, 0.02)
    return (
        [x[0] for x in rows],
        np.array([x[1] for x in rows]),
        np.array([x[2] for x in rows]),
        np.array([x[3] for x in rows]),
    )


def supported_profiles():
    opt = density.build(dict(density=dict(density_profile='usstd76', ozone_profile='usstd76', rayleigh_column='keep')))
    from sosfuse2.atmosphere import tropo_profile

    two_layer = [
        tropo_profile.build(
            dict(
                parameters=dict(aod550=0.15),
                troposphere_profile=dict(status="enabled", model="two_layer", bl_shape=shape),
            )
        )
        for shape in ("uniform", "exponential")
    ]
    return (
        [ct.descriptor('exp', height=h) for h in (0.6, 1.2, 1.3416407864998738, 2.4, 3.0, 8.0)]
        + [ct.molecular(None, 1), ct.molecular(opt, 0), ct.molecular(opt, 1)]
        + [
            ct.strato(stratosphere.profile_config(dict(stratosphere=dict(center_km=c, sigma_km=s))))
            for c, s in ((19.0, 3.0), (21.0, 3.0), (3.0, 1.0), (77.0, 1.0))
        ]
        + [ct.descriptor("two_layer", vector=p["vector"].tolist()) for p in two_layer]
    )


def reference(spec, p, d, end):
    """Independent integral: closed form for piecewise-linear profiles, GL64 split on the XYZ radius otherwise."""
    sa = np.sum(p * d, -1)
    rho = np.sqrt(np.maximum(np.sum(p * p, -1) - sa * sa, 0))
    if spec['kind'] in ('usstd76', 'triangle'):
        return ct.linear_columns(spec, rho, sa, sa + end)
    u, w = np.polynomial.legendre.leggauss(64)
    u = (u + 1) / 2
    w = w / 2
    cuts = [np.zeros(len(d)), end, np.clip(-sa, 0, end)]
    kinks = [10.0, 25.0, 40.0] + (
        [spec['vector'][0], spec['vector'][0] + 1 / spec['vector'][1]] if spec['kind'] == 'two_layer' else []
    )
    for h in kinks:
        disc = sa * sa - np.sum(p * p, -1) + (ct.R + h) ** 2
        for sign in (-1, 1):
            cuts.append(np.clip(-sa + sign * np.sqrt(np.maximum(disc, 0)), 0, end))
    cuts = np.sort(np.column_stack(cuts), 1)
    out = np.zeros(len(d))
    for k in range(cuts.shape[1] - 1):
        a, b = cuts[:, k], cuts[:, k + 1]
        m = b > a
        if not m.any():
            continue
        t = a[m, None] + (b - a)[m, None] * u
        h = np.linalg.norm(p[m, None, :] + t[..., None] * d[m, None, :], axis=-1) - ct.R
        out[m] += np.sum(ct.values(h, spec) * w, axis=1) * (b - a)[m]
    return out


class ColumnTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, SOSFUSE2_COLUMN_METHOD='table-v1', SOSFUSE2_COLUMN_TABLE_CACHE=cls.tmp.name)
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        ct._MEMO.clear()
        cls.env.stop()
        cls.tmp.cleanup()

    def test_boundaries_all_profiles(self):
        names, p, d, end = boundary_rays()
        for spec in supported_profiles():
            with self.subTest(profile=spec['kind'], height=spec.get('height'), center=spec.get('center')):
                got = ct.column(spec, p, d, end)
                ref = ct.direct_columns(spec, p, d, end, 96)
                self.assertTrue(np.isfinite(got).all())
                self.assertTrue((got >= 0).all())
                # Conservative per-column coefficient >= each bake44 coefficient.
                coefficient = 2.0 if spec['kind'] == 'exp' and spec['height'] < 8 else 0.1
                np.testing.assert_allclose(got * coefficient, ref * coefficient, rtol=0, atol=1e-4)

    def test_stable_tangent_coordinates(self):
        p = np.array([[0.0, 0.0, ct.R + 0.200001]])
        d = np.array([[1.0, 0.0, 0.0]])
        c = ct.coordinates(p, d, np.array([0.01]))
        self.assertFalse(c['direct'][0])
        self.assertEqual(c['xa'][0], 0)
        self.assertGreater(c['xb'][0], 0)
        for h in (0.6, 1.2):
            spec = ct.descriptor('exp', height=h)
            got = ct.column(spec, p, d, np.array([0.01]))
            np.testing.assert_allclose(got, ct.direct_columns(spec, p, d, np.array([0.01]), 96), rtol=0, atol=1e-8)

    def test_fallback_predicates_both_sides(self):
        names, p, d, end = boundary_rays()
        c = ct.coordinates(p, d, end)
        for name in ('ht_02_oracle', 'x_005_tangent', 'x_005_ground', 'ht_79', 'L_1m'):
            idx = np.array([i for i, v in enumerate(names) if v == name])
            if name == 'L_1m':
                self.assertTrue(c['short'][idx[0]])
                self.assertFalse(c['short'][idx[-1]])
            elif name == 'ht_79':
                self.assertFalse(c['direct'][idx[0]])
                self.assertTrue(c['direct'][idx[-1]])
            else:
                self.assertTrue(c['direct'][idx[0]])
                self.assertFalse(c['direct'][idx[-1]])

    def test_us1976_fallback_exact(self):
        _, p, d, end = boundary_rays()
        c = ct.coordinates(p, d, end)
        for spec in supported_profiles()[7:9]:
            got = ct.column(spec, p, d, end)
            exact = ct.linear_columns(spec, c['rho'], c['sa'], c['sb'])
            assert_equivalent(got[c['direct']], exact[c['direct']])

    def test_profile_cache_H_A_B_A_and_empty(self):
        _, p, d, end = boundary_rays()
        end = np.maximum(end, 0.01)
        outputs = []
        for h in (1.2, 0.6, 1.2):
            outputs.append(ct.column(ct.descriptor('exp', height=h), p, d, end).astype('f4'))
        for h, expected in zip((1.2, 0.6, 1.2), outputs):
            with tempfile.TemporaryDirectory() as fresh, patch.dict(os.environ, SOSFUSE2_COLUMN_TABLE_CACHE=fresh):
                got = ct.column(ct.descriptor('exp', height=h), p, d, end).astype('f4')
                assert_equivalent(got, expected)
        assert_equivalent(outputs[0], outputs[2])

    def test_method_separation_disk_and_memory(self):
        from sosfuse2.solve.ray_geometry import Bundle

        bundle = Bundle.__new__(Bundle)
        keys = []
        for method in ('gl24', 'table-v1'):
            with patch.dict(os.environ, SOSFUSE2_COLUMN_METHOD=method):
                keys.append(ct.table_key(ct.descriptor('exp', height=1.2)))
                self.assertNotIn('x', bundle.strato_columns)
                bundle.strato_columns['x'] = method
                bundle.density_columns = {'x': method}
        self.assertNotEqual(keys[0], keys[1])
        for method in ('gl24', 'table-v1'):
            with patch.dict(os.environ, SOSFUSE2_COLUMN_METHOD=method):
                self.assertEqual(bundle.strato_columns['x'], method)
                self.assertEqual(bundle.density_columns['x'], method)

    def test_ssa_and_g_reuse(self):
        for key in ('ssa', 'g'):
            a = dict(parameters=dict(scale_height_km=1.2, ssa=0.92, g=0.65))
            b = json.loads(json.dumps(a))
            b['parameters'][key] = 0.98
            specs = [ct.descriptor('exp', height=s['parameters']['scale_height_km']) for s in (a, b)]
            self.assertEqual(ct.table_key(specs[0]), ct.table_key(specs[1]))
            self.assertIs(ct.tables(specs[0]), ct.tables(specs[1]))

    def test_promoted_bundle_does_not_overwrite_host_memo(self):
        import copy
        from sosfuse2.solve.ray_geometry import Bundle

        host = Bundle.__new__(Bundle)
        host.strato_columns = {'x': np.array([1.0])}
        device = copy.copy(host)
        device.strato_columns = {'x': np.array([2.0])}
        self.assertEqual(host.strato_columns['x'][0], 1.0)
        self.assertEqual(device.strato_columns['x'][0], 2.0)

    def test_default_method(self):
        with patch.dict(os.environ):
            os.environ.pop('SOSFUSE2_COLUMN_METHOD', None)
            self.assertEqual(ct.method(), 'table-v1')

    def test_exact_reference_matches_independent_integral(self):
        # Gate B reference path vs the gate A reference (closed form / GL64 on XYZ radius), boundary rays plus random rays.
        names, p, d, end = boundary_rays()
        rng = np.random.default_rng(7)
        n = 400
        q = np.column_stack([np.zeros(n), np.zeros(n), ct.R + rng.uniform(0, 40, n)])
        mu = rng.uniform(-0.3, 1, n)
        v = np.column_stack([np.sqrt(1 - mu * mu), np.zeros(n), mu])
        p = np.vstack([p, q])
        d = np.vstack([d, v])
        end = np.concatenate([end, rng.uniform(0, 300, n)])
        with patch.dict(os.environ, SOSFUSE2_COLUMN_METHOD='exact'):
            for spec in supported_profiles():
                with self.subTest(profile=spec['kind'], height=spec.get('height'), center=spec.get('center')):
                    got = ct.column(spec, p, d, end)
                    want = np.maximum(reference(spec, p, d, end), 0)
                    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)

    def test_invalid_method_is_visible(self):
        with patch.dict(os.environ, SOSFUSE2_COLUMN_METHOD='typo'):
            with self.assertRaisesRegex(ValueError, 'E_COLUMN_METHOD'):
                ct.method()


if __name__ == '__main__':
    unittest.main()
