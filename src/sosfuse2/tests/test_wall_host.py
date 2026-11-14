"""r11 host port: exact NOPREFIX geometry parity."""

import itertools
import os
import unittest
from unittest.mock import patch

import numpy as np
from sosfuse2.atmosphere import stratosphere


class WallHost(unittest.TestCase):
    def test_noprefix_real_geometry_bytes(self):
        from sosfuse2.checks.testkit import sun_scope

        old = sun_scope()
        profile = stratosphere.profile_config({})
        rng = np.random.default_rng(17)
        for height, n in itertools.product((0.0, 0.1, 10.0, 21.0, 25.0, 40.0, 79.0), (24, 32)):
            d = rng.normal(size=(65, 3))
            d /= np.linalg.norm(d, axis=1)[:, None]
            d[:5] = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [1, 0, 1e-10], [1, 0, -1e-10]]
            p = np.broadcast_to([0.0, 0.0, 6371.0 + height], d.shape)
            with patch.dict(os.environ, {'SOSFUSE2_STRATO_HOST_NOPREFIX': '0'}):
                expected = stratosphere.columns(p, d, old, profile, n)
            with patch.dict(os.environ, {'SOSFUSE2_STRATO_HOST_NOPREFIX': '1'}):
                with patch.object(
                    old, 'geometry_segments', side_effect=AssertionError('unused optical prefix evaluated')
                ):
                    actual = stratosphere.columns(p, d, old, profile, n)
            self.assertEqual(actual.tobytes(), expected.tobytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
