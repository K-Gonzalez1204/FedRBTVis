import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import numpy as np
import torch

from fedrbtvis.config import RunConfig
from fedrbtvis.data import DatasetBundle, build_synthetic_bundle
from fedrbtvis.engine import ExperimentResult, run_experiment
from fedrbtvis.events import TrainingEvent
from fedrbtvis.noise import inject_symmetric_noise
from fedrbtvis.presets import build_preset


def run_fixture_once() -> ExperimentResult:
    config = build_preset("test-fixture", Path("data"), Path("runs"))
    bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
    return run_experiment(
        "run-repeatable",
        config,
        bundle,
        torch.device("cpu"),
        lambda event: None,
        lambda: False,
    )


def run_fixture_with_stop(
    requested: Callable[[], bool],
) -> tuple[ExperimentResult, list[TrainingEvent]]:
    config = build_preset("test-fixture", Path("data"), Path("runs"))
    bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
    events: list[TrainingEvent] = []
    result = run_experiment(
        "run-stopping",
        config,
        bundle,
        torch.device("cpu"),
        events.append,
        requested,
    )
    return result, events


class EngineTest(unittest.TestCase):
    def test_fixture_emits_ordered_events_and_real_metrics(self) -> None:
        config = build_preset("test-fixture", Path("data"), Path("runs"))
        bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
        events: list[TrainingEvent] = []

        result = run_experiment(
            "run-test",
            config,
            bundle,
            torch.device("cpu"),
            events.append,
            lambda: False,
        )

        self.assertEqual(
            [event.sequence for event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[0].type, "run.started")
        self.assertEqual(events[-1].type, "run.completed")
        completed = [event for event in events if event.type == "client.completed"]
        self.assertEqual(len(completed), 4)
        self.assertTrue(
            all(0.0 <= event.payload["test_accuracy"] <= 1.0 for event in completed)
        )
        self.assertTrue(
            any(event.payload["test_accuracy"] > 0.0 for event in completed)
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(len({row.state_sha256 for row in result.client_updates}), 4)

    def test_same_seed_repeats_schedule_and_metrics(self) -> None:
        first = run_fixture_once()
        second = run_fixture_once()

        self.assertEqual(first.schedule, second.schedule)
        self.assertEqual(first.aggregations, second.aggregations)
        self.assertEqual(
            [update.state_sha256 for update in first.client_updates],
            [update.state_sha256 for update in second.client_updates],
        )
        self.assertEqual(
            [update.test for update in first.client_updates],
            [update.test for update in second.client_updates],
        )

    def test_noisy_label_views_remain_fixed_across_cycles(self) -> None:
        base = build_preset("test-fixture", Path("data"), Path("runs"))
        config = RunConfig(**{**base.model_dump(), "cycles": 2})
        bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
        captured_labels: list[np.ndarray] = []

        def recording_noise(*args: object, **kwargs: object):
            result = inject_symmetric_noise(*args, **kwargs)
            captured_labels.append(result.labels.copy())
            return result

        with patch(
            "fedrbtvis.training.inject_symmetric_noise",
            side_effect=recording_noise,
        ):
            result = run_experiment(
                "run-two-cycles",
                config,
                bundle,
                torch.device("cpu"),
                lambda event: None,
                lambda: False,
            )

        scheduled_clients = [
            client_id for group in result.schedule for client_id in group
        ]
        labels_by_client: dict[int, list[np.ndarray]] = {}
        for client_id, labels in zip(
            scheduled_clients,
            captured_labels,
            strict=True,
        ):
            labels_by_client.setdefault(client_id, []).append(labels)
        self.assertTrue(
            all(
                len(views) == 2 and np.array_equal(views[0], views[1])
                for views in labels_by_client.values()
            )
        )

    def test_misaligned_training_bundle_is_rejected_before_events(self) -> None:
        config = build_preset("test-fixture", Path("data"), Path("runs"))
        bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
        malformed = DatasetBundle(
            train_images=bundle.train_images[:10],
            train_labels=bundle.train_labels,
            test_images=bundle.test_images,
            test_labels=bundle.test_labels,
        )
        events: list[TrainingEvent] = []

        with self.assertRaisesRegex(ValueError, "train images and labels"):
            run_experiment(
                "run-malformed",
                config,
                malformed,
                torch.device("cpu"),
                events.append,
                lambda: False,
            )
        self.assertEqual(events, [])

    def test_empty_test_bundle_is_rejected_before_events(self) -> None:
        config = build_preset("test-fixture", Path("data"), Path("runs"))
        bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
        malformed = DatasetBundle(
            train_images=bundle.train_images,
            train_labels=bundle.train_labels,
            test_images=bundle.test_images[:0],
            test_labels=bundle.test_labels[:0],
        )
        events: list[TrainingEvent] = []

        with self.assertRaisesRegex(ValueError, "test split must not be empty"):
            run_experiment(
                "run-empty-test",
                config,
                malformed,
                torch.device("cpu"),
                events.append,
                lambda: False,
            )
        self.assertEqual(events, [])

    def test_stop_is_honored_at_client_boundary(self) -> None:
        calls = 0

        def requested() -> bool:
            nonlocal calls
            calls += 1
            return calls > 2

        result, events = run_fixture_with_stop(requested)

        self.assertEqual(result.status, "stopped")
        self.assertEqual(events[-1].type, "run.stopped")
        self.assertNotIn("run.completed", [event.type for event in events])
        self.assertEqual(
            [event.type for event in events].count("run.stop_requested"),
            1,
        )
        self.assertEqual(len(result.aggregations), 1)


if __name__ == "__main__":
    unittest.main()
