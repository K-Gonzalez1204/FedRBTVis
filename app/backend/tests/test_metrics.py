import unittest

import numpy as np

from fedrbtvis.metrics import categorical_emd_01, lid_mle


class CategoricalDistanceTest(unittest.TestCase):
    def test_uniform_is_zero_and_single_class_is_point_nine(self) -> None:
        self.assertAlmostEqual(categorical_emd_01(np.full(10, 10)), 0.0)
        self.assertAlmostEqual(
            categorical_emd_01(np.array([100] + [0] * 9)),
            0.9,
        )

    def test_zero_histogram_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            categorical_emd_01(np.zeros(10))


class LidTest(unittest.TestCase):
    def test_lid_is_deterministic_finite_and_per_sample(self) -> None:
        probabilities = np.random.default_rng(7).dirichlet(np.ones(10), size=24)

        first = lid_mle(probabilities, k=5)
        second = lid_mle(probabilities, k=5)

        self.assertEqual(first.shape, (24,))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second)

    def test_lid_rejects_k_equal_to_sample_count(self) -> None:
        with self.assertRaises(ValueError):
            lid_mle(np.eye(4), k=4)


if __name__ == "__main__":
    unittest.main()
