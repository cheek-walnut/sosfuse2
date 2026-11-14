"""Compare the joint walk with the independent accepted three-pass integrators."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from sosfuse2.atmosphere import density, stratosphere
from sosfuse2.solve import ray_geometry
from sosfuse2.rays import ray_columns

from sosfuse2.checks.numerical_checks import assert_equivalent


class JointColumnsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with mock.patch.dict(os.environ, {'SOSFUSE2_SUN_WORKERS': '1', 'SOSFUSE2_GROUND_WORKERS': '1'}):
            cls.s, cls.scope, _ = ray_geometry.create(Path(cls.tmp.name), 'selftest', [550.0], 'numpy')
        cls.old = cls.scope.b.sh.old
        cls.optics = density.build(dict(density=dict(density_profile='usstd76', ozone_profile='usstd76', rayleigh_column='keep')))
        cls.profile = stratosphere.profile_config({})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assert_close(self, a, b):
        assert_equivalent(a, b)

    def rays(self):
        # Include zero-length ground rays, grazing rays, and shell boundaries.
        heights = np.array([0.0, 0.001, 1.2, 10.0, 25.0, 40.0, 79.9])
        p = np.column_stack([np.zeros(len(heights)), np.zeros(len(heights)), 6371.0 + heights])
        mu = np.array([-1.0, -0.5, -1e-9, 0.0, 1e-9, 0.5, 1.0])
        d = np.column_stack([np.sqrt(1 - mu * mu), np.zeros(len(mu)), mu])
        return np.repeat(p, len(d), axis=0), np.tile(d, (len(p), 1))

    def test_joint_matches_three_passes(self):
        p, d = self.rays()
        for n in (24, 32):
            base = self.old.total_columns(p, d, n)
            for mask in (0, 1, 2, 3):
                optics = dict(self.optics, mask=mask)
                a, b, c = ray_columns.integrate(p, d, self.old, n, legacy=True, optics=optics, profile=self.profile)
                self.assert_close(a, base)
                self.assert_close(
                    ray_columns.replace(a, b, optics), density.correct_columns(p, d, base, self.old, optics, n)
                )
                self.assert_close(c, stratosphere.columns(p, d, self.old, self.profile, n))

    # pack() takes the main-ray columns from column_table unless the method is gl24 (the default is table-v1), and the
    # table method differs from the 24-point walk by up to 7.5e-4 on rays that start on a shell and run horizontally,
    # where the walk is the less accurate one. This case compares the two walks, so it pins the walk.
    @mock.patch.dict(os.environ, {'SOSFUSE2_COLUMN_METHOD': 'gl24'})
    def test_pack_reuses_edges_and_keeps_legacy_bytes(self):
        p, d = self.rays()
        packed = ray_geometry.pack(p, d, 24, self.s, self.scope)
        a, _, _ = ray_columns.integrate(p, d, self.old, legacy=True, edges=packed['edges'])
        self.assert_close(a, self.old.total_columns(p, d, 24))
        self.assert_close(packed['end_columns'], a)


if __name__ == '__main__':
    unittest.main()
