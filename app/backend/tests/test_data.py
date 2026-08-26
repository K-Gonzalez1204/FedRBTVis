import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fedrbtvis.data import (
    PartitionError,
    build_client_partitions,
    build_synthetic_bundle,
    load_cifar10,
)
from fedrbtvis.metrics import categorical_emd_01
from fedrbtvis.presets import build_preset


class SyntheticDatasetTest(unittest.TestCase):
    def test_fixture_data_repeats_for_the_same_seed(self) -> None:
        first = build_synthetic_bundle(7, 10, 30, 20)
        second = build_synthetic_bundle(7, 10, 30, 20)

        torch.testing.assert_close(first.train_images, second.train_images)
        np.testing.assert_array_equal(first.train_labels, second.train_labels)
        torch.testing.assert_close(first.test_images, second.test_images)
        np.testing.assert_array_equal(first.test_labels, second.test_labels)

    def test_missing_cifar_cache_is_explicit_when_download_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_cifar10(Path(directory), download=False)


class PartitionTest(unittest.TestCase):
    def test_fixture_partitions_are_disjoint_and_repeatable(self) -> None:
        config = build_preset("test-fixture", Path("data"), Path("runs"))
        bundle = build_synthetic_bundle(
            seed=config.seed,
            num_classes=10,
            train_size=300,
            test_size=100,
        )

        first = build_client_partitions(bundle.train_labels, config)
        second = build_client_partitions(bundle.train_labels, config)

        self.assertEqual([partition.indices for partition in first], [p.indices for p in second])
        flattened = [index for partition in first for index in partition.indices]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(len(first), 4)

    def test_probe_actual_emd_matches_integer_histogram(self) -> None:
        config = build_preset("test-fixture", Path("data"), Path("runs"))
        bundle = build_synthetic_bundle(
            seed=7,
            num_classes=10,
            train_size=300,
            test_size=100,
        )

        probes = [
            partition
            for partition in build_client_partitions(bundle.train_labels, config)
            if partition.role == "probe"
        ]

        for probe in probes:
            self.assertAlmostEqual(
                probe.actual_emd,
                categorical_emd_01(np.array(probe.clean_histogram)),
            )
            self.assertEqual(sum(probe.clean_histogram), probe.sample_count)

    def test_insufficient_class_capacity_is_rejected(self) -> None:
        config = build_preset("test-fixture", Path("data"), Path("runs"))

        with self.assertRaises(PartitionError):
            build_client_partitions(np.zeros(30, dtype=np.int64), config)


if __name__ == "__main__":
    unittest.main()
