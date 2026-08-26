import unittest
from pathlib import Path

from pydantic import ValidationError

from fedrbtvis.config import ProbeSpec, RunConfig
from fedrbtvis.presets import build_preset


class RunConfigTest(unittest.TestCase):
    def test_research_lite_contract(self) -> None:
        config = build_preset("research-lite", Path("data"), Path("runs"))

        self.assertEqual(config.dataset, "cifar10")
        self.assertEqual(config.model, "cifar-resnet18")
        self.assertEqual(config.background_clients, 10)
        self.assertEqual(len(config.probes), 5)
        self.assertEqual(
            [probe.target_emd for probe in config.probes],
            [0.0, 0.2, 0.4, 0.6, 0.8],
        )
        self.assertTrue(
            config.model_fields_set.issuperset({"preset", "dataset", "model"})
        )

    def test_invalid_lid_k_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ProbeSpec(
                client_id=10,
                target_noise=0.2,
                target_emd=0.4,
                sample_count=20,
                lid_k=20,
            )

    def test_invalid_background_lid_k_is_rejected_before_training(self) -> None:
        base = build_preset("test-fixture", Path("data"), Path("runs"))

        with self.assertRaises(ValidationError):
            RunConfig(**{**base.model_dump(), "background_samples": 5})

    def test_duplicate_client_ids_are_rejected(self) -> None:
        probe = ProbeSpec(
            client_id=2,
            target_noise=0.2,
            target_emd=0.2,
            sample_count=24,
            lid_k=5,
        )
        base = build_preset("test-fixture", Path("data"), Path("runs"))

        with self.assertRaises(ValidationError):
            RunConfig(**{**base.model_dump(), "probes": [probe, probe]})

    def test_configs_are_frozen(self) -> None:
        config = build_preset("test-fixture", Path("data"), Path("runs"))

        with self.assertRaises(ValidationError):
            config.seed = 99


if __name__ == "__main__":
    unittest.main()
