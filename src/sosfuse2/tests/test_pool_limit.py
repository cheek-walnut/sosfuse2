"""Host-only checks of the CuPy memory-pool cap and the admission field reserve (device_profile.pool_limit_bytes,
reserved_field_bytes); no CUDA, no engine.

Run: python -B -m sosfuse2.tests.test_pool_limit
"""

import os
import unittest
from pathlib import Path
from unittest import mock

from sosfuse2.backends import device_profile
from sosfuse2.foundation import paths

GIB = 1024**3


def cap(free, value=None):
    env = {k: v for k, v in os.environ.items() if k != 'SOSFUSE2_POOL_LIMIT_GIB'}
    if value is not None:
        env['SOSFUSE2_POOL_LIMIT_GIB'] = value
    with mock.patch.dict(os.environ, env, clear=True):
        return device_profile.pool_limit_bytes(free)


class PoolLimit(unittest.TestCase):
    def test_default_is_free_minus_one_gib(self):
        # 47 GB free on a 48 GB card: no fixed 20 GiB term any more.
        self.assertEqual(cap(47_000_000_000), 47_000_000_000 - GIB)
        self.assertEqual(cap(9_500_000_000), 9_500_000_000 - GIB)
        self.assertEqual(cap(47_000_000_000, '  '), 47_000_000_000 - GIB)  # blank means unset

    def test_environment_lowers_the_cap(self):
        self.assertEqual(cap(47_000_000_000, '20'), 20 * GIB)  # the 1.3.4 cap on this card
        self.assertEqual(cap(47_000_000_000, '30.5'), int(30.5 * GIB))

    def test_environment_never_exceeds_free_minus_one_gib(self):
        self.assertEqual(cap(9_500_000_000, '20'), 9_500_000_000 - GIB)
        self.assertEqual(cap(47_000_000_000, '1000'), 47_000_000_000 - GIB)
        self.assertEqual(cap(47_000_000_000, '1e300'), 47_000_000_000 - GIB)  # no OverflowError

    def test_invalid_environment_is_refused(self):
        for value in ('0', '-1', 'abc', 'inf', 'nan', 'off', '1e-10'):  # 1e-10 GiB rounds to 0 bytes = no limit
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'E_DEVICE_PROFILE'):
                cap(47_000_000_000, value)

    def test_field_reserve(self):
        def reserved(field, value=None):
            env = {k: v for k, v in os.environ.items() if k != 'SOSFUSE2_FIELD_RESERVE'}
            if value is not None:
                env['SOSFUSE2_FIELD_RESERVE'] = value
            with mock.patch.dict(os.environ, env, clear=True):
                return device_profile.reserved_field_bytes(field)

        for field in (1_790_000_000, 3_426_000_000_034_260):  # default: the 1.3.x integer arithmetic exactly
            self.assertEqual(reserved(field), 30 * field)
            self.assertEqual(reserved(field, ' '), 30 * field)
        self.assertEqual(reserved(1_790_000_000, '10'), 17_900_000_000)
        self.assertEqual(reserved(1_000, '12.5'), 12_500)
        for value in ('0', '-1', 'abc', 'inf', 'nan', 'off', '1e300'):  # 1e300 x field overflows to inf
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'E_DEVICE_PROFILE'):
                reserved(1_790_000_000, value)
        source = (paths.PKG / 'jobs/pipeline_execution.py').read_text()
        self.assertNotIn('30 * field_bytes', source)
        self.assertIn('reserved_field_bytes(field_bytes)', source)

    def test_no_fixed_cap_left_in_the_package(self):
        root = paths.PKG
        modules = paths.production_files()
        left = [p.relative_to(root).as_posix() for p in modules if '20 * 1024**3' in p.read_text()]
        self.assertEqual(left, [])
        users = sorted(p.relative_to(root).as_posix() for p in modules if 'set_limit(size=' in p.read_text())
        for name in users:
            with self.subTest(file=name):
                self.assertIn('pool_limit_bytes', (root / name).read_text())


if __name__ == '__main__':
    unittest.main()
