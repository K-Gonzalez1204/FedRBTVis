import unittest

import numpy as np

from fedrbtvis.noise import inject_symmetric_noise


class SymmetricNoiseTest(unittest.TestCase):
    def test_exact_changed_count_and_source_immutability(self) -> None:
        source = np.arange(20, dtype=np.int64) % 10
        before = source.copy()

        result = inject_symmetric_noise(
            source,
            target_rate=0.25,
            num_classes=10,
            rng=np.random.default_rng(3),
        )

        np.testing.assert_array_equal(source, before)
        self.assertEqual(len(result.changed_indices), 5)
        self.assertAlmostEqual(result.actual_rate, 0.25)
        self.assertTrue(
            np.all(result.labels[result.changed_indices] != source[result.changed_indices])
        )

    def test_rate_zero_returns_a_copy(self) -> None:
        source = np.array([0, 1, 2])

        result = inject_symmetric_noise(
            source,
            target_rate=0.0,
            num_classes=3,
            rng=np.random.default_rng(1),
        )

        self.assertIsNot(result.labels, source)
        np.testing.assert_array_equal(result.labels, source)

    def test_out_of_range_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inject_symmetric_noise(
                np.array([0, 3]),
                target_rate=0.5,
                num_classes=3,
                rng=np.random.default_rng(1),
            )


if __name__ == "__main__":
    unittest.main()
