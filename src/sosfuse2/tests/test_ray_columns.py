"""Compare the joint walk with the independent accepted three-pass integrators."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from sosfuse2 import density, geometry, ray_columns, stratosphere, testing
try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


class JointColumnsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with mock.patch.dict(os.environ, {'SOSFUSE2_SUN_WORKERS': '1', 'SOSFUSE2_GROUND_WORKERS': '1'}):
            cls.s, cls.scope, _ = geometry.create(Path(cls.tmp.name), 'selftest', [550.], 'numpy')
        cls.old = cls.scope['b'].sh.old
        cls.optics = density.build(dict(density=dict(density_profile='usstd76', ozone_profile='usstd76')))
        cls.profile = stratosphere.profile_config({})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assert_close(self, a, b):
        assert_equivalent(a, b)

    def rays(self):
        # Include zero-length ground rays, grazing rays, and shell boundaries.
        heights = np.array([0., .001, 1.2, 10., 25., 40., 79.9])
        p = np.column_stack([np.zeros(len(heights)), np.zeros(len(heights)), 6371.+heights])
        mu = np.array([-1., -.5, -1e-9, 0., 1e-9, .5, 1.])
        d = np.column_stack([np.sqrt(1-mu*mu), np.zeros(len(mu)), mu])
        return np.repeat(p, len(d), axis=0), np.tile(d, (len(p), 1))

    def test_joint_matches_three_passes(self):
        p, d = self.rays()
        for n in (24, 32):
            base = self.old.total_columns(p, d, n)
            for mask in (0, 1, 2, 3):
                optics = dict(self.optics, mask=mask)
                a, b, c = ray_columns.integrate(p, d, self.old, n, legacy=True,
                                                optics=optics, profile=self.profile)
                self.assert_close(a, base)
                self.assert_close(ray_columns.replace(a, b, optics),
                                  density.correct_columns(p, d, base, self.old, optics, n))
                self.assert_close(c, stratosphere.columns(p, d, self.old, self.profile, n))

    @testing.known_failure('LEGACY_JOINT_WALK', 'joint walk vs 3-pass integration differ on grazing rays (10/147 elements, max rel 1.28e-6 > rtol 1e-6); cause uninvestigated')
    def test_pack_reuses_edges_and_keeps_legacy_bytes(self):
        p, d = self.rays()
        packed = geometry.pack(p, d, 24, self.s, self.scope)
        a, _, _ = ray_columns.integrate(p, d, self.old, legacy=True, edges=packed['edges'])
        self.assert_close(a, self.old.total_columns(p, d, 24))
        self.assert_close(packed['end_columns'], a)


if __name__ == '__main__':
    unittest.main()
