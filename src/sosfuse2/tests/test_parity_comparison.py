"""D1 NumPy tests for the CUDA/NumPy moment comparison rules."""

import unittest
import numpy as np
from sosfuse2.validate import moment_comparison


class MomentComparisonTests(unittest.TestCase):
    def test_near_zero_component_uses_vector_floor(self):
        expected = np.array([[2.0], [-1e-8]])
        actual = np.array([[2.0], [-1e-8 + 1e-9]])
        report = moment_comparison(actual, expected)
        self.assertAlmostEqual(report['moment_max_relative'], 0.1)
        self.assertAlmostEqual(report['moment_scaled_relative'], 5e-7)
        self.assertAlmostEqual(report['moment_l2_relative'], 5e-10)
        self.assertTrue(report['moment_pass_'])
        diagnostic = report['moment_worst_relative']
        self.assertEqual(diagnostic['index'], [1, 0])
        self.assertEqual(diagnostic['numpy_abs'], 1e-8)
        self.assertEqual(diagnostic['vector_max_abs'], 2.0)
        self.assertEqual(diagnostic['numpy_fraction_of_vector_max'], 5e-9)

    def test_each_wavelength_has_its_own_scale_and_l2(self):
        # A large wavelength must not hide an error in a much smaller one.
        expected = np.array([[1e6, 1.0], [0.0, 0.0]])
        actual = expected.copy()
        actual[1, 1] = 2e-8
        report = moment_comparison(actual, expected)
        self.assertAlmostEqual(report['moment_scaled_relative'], 2e-5)
        self.assertAlmostEqual(report['moment_l2_relative'], 2e-8)
        self.assertFalse(report['moment_pass_'])

    def test_l2_gate_is_required_even_if_scaled_gate_passes(self):
        expected = np.array([[1.0], [0.001]])
        actual = expected.copy()
        actual[0, 0] += 2e-6
        report = moment_comparison(actual, expected)
        self.assertLess(report['moment_scaled_relative'], 1e-5)
        self.assertGreater(report['moment_l2_relative'], 1e-6)
        self.assertFalse(report['moment_pass_'])

    def test_scaled_limit_is_inclusive_and_excess_fails(self):
        expected = np.array([[1.0], [0.0]])
        actual = np.array([[1.0], [1e-8]])
        self.assertTrue(moment_comparison(actual, expected)['moment_pass_'])
        actual[1, 0] = 1.000001e-8
        self.assertFalse(moment_comparison(actual, expected)['moment_pass_'])

    def test_zero_vectors(self):
        expected = np.zeros((3, 2))
        self.assertTrue(moment_comparison(expected, expected)['moment_pass_'])
        actual = expected.copy()
        actual[0, 1] = 1e-30
        self.assertFalse(moment_comparison(actual, expected)['moment_pass_'])

    def test_nonfinite_errors_cannot_pass(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                self.assertFalse(moment_comparison([[value]], [[1.0]])['moment_pass_'])


if __name__ == '__main__':
    unittest.main()
