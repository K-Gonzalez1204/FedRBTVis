import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedrbtvis.data import build_client_partitions, build_synthetic_bundle
from fedrbtvis.models import build_model
from fedrbtvis.presets import build_preset
from fedrbtvis.training import metrics_from_logits, train_local, weighted_fedavg


class ModelTest(unittest.TestCase):
    def test_cifar_resnet_stem_is_not_imagenet_stem(self) -> None:
        model = build_model("cifar-resnet18", num_classes=10, seed=3)

        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertIsInstance(model.maxpool, torch.nn.Identity)

    def test_model_initialization_does_not_change_global_rng_state(self) -> None:
        torch.manual_seed(41)
        before = torch.random.get_rng_state().clone()

        build_model("tiny-cnn", num_classes=10, seed=3)

        torch.testing.assert_close(torch.random.get_rng_state(), before)


class FedAvgTest(unittest.TestCase):
    def test_float_tensors_are_sample_weighted(self) -> None:
        one = {
            "weight": torch.tensor([1.0]),
            "counter": torch.tensor(1, dtype=torch.long),
        }
        three = {
            "weight": torch.tensor([3.0]),
            "counter": torch.tensor(2, dtype=torch.long),
        }

        result = weighted_fedavg([(one, 1), (three, 3)])

        torch.testing.assert_close(result["weight"], torch.tensor([2.5]))
        self.assertEqual(result["counter"].item(), 2)

    def test_aggregation_does_not_alias_client_states(self) -> None:
        state = {"weight": torch.tensor([1.0])}

        result = weighted_fedavg([(state, 1)])
        result["weight"].add_(5)

        self.assertEqual(state["weight"].item(), 1.0)


class EvaluationTest(unittest.TestCase):
    def test_accuracy_is_computed_from_logits(self) -> None:
        logits = torch.tensor([[4.0, 1.0], [0.0, 2.0]])
        labels = torch.tensor([0, 1])

        metrics = metrics_from_logits(logits, labels)

        self.assertEqual(metrics.correct, 2)
        self.assertAlmostEqual(metrics.accuracy, 1.0)
        expected = torch.nn.functional.cross_entropy(logits, labels).item()
        self.assertAlmostEqual(metrics.loss, expected)


class LocalTrainingTest(unittest.TestCase):
    def test_fixture_update_contains_computed_metrics_and_owned_state(self) -> None:
        config = build_preset("test-fixture", Path("data"), Path("runs"))
        bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
        client = build_client_partitions(bundle.train_labels, config)[3]
        server = build_model(config.model, config.num_classes, config.seed).state_dict()
        test_loader = DataLoader(
            TensorDataset(
                bundle.test_images,
                torch.from_numpy(bundle.test_labels),
            ),
            batch_size=20,
            shuffle=False,
        )

        update = train_local(
            client=client,
            server_state=server,
            model_name=config.model,
            num_classes=config.num_classes,
            train_images=bundle.train_images,
            clean_labels=bundle.train_labels,
            test_loader=test_loader,
            local_epochs=config.local_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
            device=torch.device("cpu"),
        )

        self.assertEqual(update.client_id, client.client_id)
        self.assertAlmostEqual(update.actual_noise, 8 / 30)
        self.assertTrue(np.isfinite(update.lid_mean))
        self.assertTrue(np.isfinite(update.lid_std))
        self.assertGreater(update.test.samples, 0)
        self.assertTrue(0.0 <= update.test.accuracy <= 1.0)
        self.assertEqual(len(update.state_sha256), 64)
        first_key = next(iter(update.state_dict))
        self.assertIsNot(update.state_dict[first_key], server[first_key])


if __name__ == "__main__":
    unittest.main()
