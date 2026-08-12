import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from fedrbtvis.artifacts import (
    ArtifactCorruptError,
    ArtifactInvariantError,
    ArtifactStore,
    sha256_file,
)
from fedrbtvis.config import RunConfig
from fedrbtvis.data import build_client_partitions, build_synthetic_bundle
from fedrbtvis.engine import AggregationRow, ExperimentResult
from fedrbtvis.events import TrainingEvent
from fedrbtvis.presets import build_preset
from fedrbtvis.training import ClientUpdate, EvaluationMetrics


def make_event(
    run_id: str,
    sequence: int,
    event_type: str,
    payload: dict,
) -> TrainingEvent:
    return TrainingEvent(
        schema_version=1,
        event_id=f"event-{sequence}",
        run_id=run_id,
        sequence=sequence,
        type=event_type,
        created_at="2026-08-01T00:00:00+00:00",
        payload=payload,
    )


def fixture_result(
    run_id: str = "run-1",
    client_ids: tuple[int, ...] = (0, 1),
) -> ExperimentResult:
    evaluation = EvaluationMetrics(loss=0.75, accuracy=0.5, correct=1, samples=2)
    updates = tuple(
        ClientUpdate(
            client_id=client_id,
            sample_count=2 + client_id,
            state_dict={"weight": torch.tensor([float(client_id + 1)])},
            train_loss=1.0 + client_id,
            test=evaluation,
            lid_mean=2.0 + client_id,
            lid_std=0.1,
            actual_noise=0.25 * client_id,
            actual_emd=0.2 * client_id,
            state_sha256=str(client_id + 1) * 64,
        )
        for client_id in client_ids
    )
    aggregation = AggregationRow(
        cycle=1,
        step=1,
        client_ids=(0, 1),
        test_loss=0.75,
        test_accuracy=0.5,
    )
    return ExperimentResult(
        run_id=run_id,
        status="completed",
        schedule=((0, 1),),
        client_updates=updates,
        aggregations=(aggregation,),
        final_server_state={"weight": torch.tensor([1.5])},
    )


def prepared_store(
    root: Path,
    run_id: str,
    status: str = "queued",
) -> ArtifactStore:
    store = ArtifactStore(root)
    config = build_preset("test-fixture", Path("data"), root)
    store.create_run(run_id, config)
    if status == "running":
        store.mark_running(run_id)
    return store


class ArtifactStoreTest(unittest.TestCase):
    def test_event_is_persisted_before_it_is_returned(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            config = build_preset("test-fixture", Path("data"), root)
            store.create_run("run-1", config)
            event = make_event(
                "run-1",
                1,
                "run.started",
                {"preset": "test-fixture"},
            )

            returned = store.append_event(event)

            lines = (root / "run-1" / "events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(json.loads(lines[0])["sequence"], 1)
            self.assertEqual(returned, event)
            self.assertEqual(store.read_events("run-1", after_sequence=0), [event])

    def test_out_of_order_event_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            store = prepared_store(Path(tmp), "run-1")

            with self.assertRaises(ArtifactInvariantError):
                store.append_event(make_event("run-1", 2, "run.started", {}))

    def test_manifest_hashes_match_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = prepared_store(root, "run-1")
            store.append_event(make_event("run-1", 1, "run.started", {}))
            store.finalize("run-1", "completed", fixture_result())

            manifest = store.read_manifest("run-1")

            for item in manifest["files"]:
                path = root / "run-1" / item["path"]
                self.assertEqual(item["sha256"], sha256_file(path))
                self.assertEqual(item["bytes"], path.stat().st_size)

    def test_interrupted_run_is_failed_during_reindex(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_store(root, "run-1", status="running")

            rebuilt = ArtifactStore(root)

            manifest = rebuilt.read_manifest("run-1")
            events = rebuilt.read_events("run-1", 0)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error_code"], "PROCESS_INTERRUPTED")
            self.assertEqual([item.type for item in events], ["run.failed"])
            self.assertEqual([item.sequence for item in events], [1])

    def test_reindex_replaces_terminal_event_from_finalize_crash_window(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = prepared_store(root, "run-1", status="running")
            store.append_event(make_event("run-1", 1, "run.started", {}))
            store.append_event(make_event("run-1", 2, "run.completed", {}))

            rebuilt = ArtifactStore(root)

            manifest = rebuilt.read_manifest("run-1")
            events = rebuilt.read_events("run-1", 0)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error_code"], "PROCESS_INTERRUPTED")
            self.assertEqual(
                [item.type for item in events],
                ["run.started", "run.failed"],
            )
            self.assertEqual([item.sequence for item in events], [1, 2])

    def test_manifest_rejects_unknown_status_and_missing_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = prepared_store(root, "run-1")
            manifest_path = root / "run-1" / "manifest.json"
            original = json.loads(manifest_path.read_text(encoding="utf-8"))
            invalid_manifests = (
                {**original, "status": "nonsense"},
                {key: value for key, value in original.items() if key != "created_at"},
            )

            for invalid in invalid_manifests:
                with self.subTest(manifest=invalid):
                    manifest_path.write_text(
                        json.dumps(invalid),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ArtifactCorruptError):
                        store.read_manifest("run-1")

    def test_manifest_read_retries_a_transient_windows_sharing_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            store.create_run(
                "run-1",
                build_preset("test-fixture", Path("data"), root),
            )
            manifest_text = (root / "run-1" / "manifest.json").read_text(
                encoding="utf-8"
            )

            with patch.object(
                Path,
                "read_text",
                side_effect=[PermissionError(13, "sharing violation"), manifest_text],
            ) as read_text:
                manifest = store.read_manifest("run-1")

            self.assertEqual(manifest["run_id"], "run-1")
            self.assertEqual(read_text.call_count, 2)

    def test_manifest_replace_retries_a_transient_windows_sharing_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            store.create_run(
                "run-1",
                build_preset("test-fixture", Path("data"), root),
            )
            real_replace = os.replace
            attempts = 0

            def replace_with_one_sharing_error(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError(13, "sharing violation")
                return real_replace(source, destination)

            with patch(
                "fedrbtvis.artifacts.os.replace",
                side_effect=replace_with_one_sharing_error,
            ):
                store.mark_running("run-1")

            manifest = store.read_manifest("run-1")
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(attempts, 2)

    def test_terminal_manifest_rejects_inventory_drift(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            config = build_preset("test-fixture", Path("data"), root)
            store.create_run("run-1", config)
            store.append_event(make_event("run-1", 1, "run.completed", {}))
            store.finalize("run-1", "completed", fixture_result())
            events_path = root / "run-1" / "events.jsonl"
            events_path.write_text(
                events_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ArtifactCorruptError):
                store.read_manifest("run-1")

    def test_metrics_reader_rejects_wrong_csv_header(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            config = build_preset("test-fixture", Path("data"), root)
            store.create_run("run-1", config)
            store.append_event(make_event("run-1", 1, "run.completed", {}))
            store.finalize("run-1", "completed", fixture_result())
            metrics_path = root / "run-1" / "metrics" / "client_updates.csv"
            lines = metrics_path.read_text(encoding="utf-8").splitlines()
            lines[0] = lines[0].replace("cycle", "wrong", 1)
            metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaises(ArtifactCorruptError):
                store.read_client_metrics("run-1")

    def test_partitions_and_metrics_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = prepared_store(root, "run-1")
            config = build_preset("test-fixture", Path("data"), root)
            bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
            partitions = build_client_partitions(bundle.train_labels, config)
            store.write_partitions("run-1", partitions)
            store.finalize("run-1", "completed", fixture_result())

            persisted_partitions = json.loads(
                (root / "run-1" / "partitions.json").read_text(encoding="utf-8")
            )
            clients = store.read_client_metrics("run-1")
            aggregations = store.read_aggregation_metrics("run-1")

            self.assertEqual(len(persisted_partitions), 4)
            self.assertEqual(clients[0]["client_id"], 0)
            self.assertEqual(clients[0]["test_accuracy"], 0.5)
            self.assertEqual(aggregations[0]["client_ids"], [0, 1])

    def test_run_id_cannot_escape_artifact_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            config = build_preset("test-fixture", Path("data"), root)

            with self.assertRaises(ArtifactInvariantError):
                store.create_run("../escape", config)

    def test_server_checkpoint_is_written_for_server_only_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = build_preset("test-fixture", Path("data"), root)
            config = RunConfig(
                **{**base.model_dump(), "checkpoint_policy": "server-only"}
            )
            store = ArtifactStore(root)
            store.create_run("run-1", config)

            store.finalize("run-1", "completed", fixture_result())

            state = torch.load(
                root / "run-1" / "checkpoints" / "server-final.pt",
                weights_only=True,
            )
            torch.testing.assert_close(state["weight"], torch.tensor([1.5]))

    def test_probe_checkpoints_keep_each_probe_owned_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = build_preset("test-fixture", Path("data"), root)
            config = RunConfig(
                **{**base.model_dump(), "checkpoint_policy": "probe-clients"}
            )
            store = ArtifactStore(root)
            store.create_run("run-1", config)
            bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
            partitions = build_client_partitions(bundle.train_labels, config)
            store.write_partitions("run-1", partitions)

            store.finalize(
                "run-1",
                "completed",
                fixture_result(client_ids=(2, 3)),
            )

            first = torch.load(
                root / "run-1" / "checkpoints" / "probes" / "client-2.pt",
                weights_only=True,
            )
            second = torch.load(
                root / "run-1" / "checkpoints" / "probes" / "client-3.pt",
                weights_only=True,
            )
            torch.testing.assert_close(first["weight"], torch.tensor([3.0]))
            torch.testing.assert_close(second["weight"], torch.tensor([4.0]))

    def test_stopped_probe_policy_keeps_only_completed_probe_states(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = build_preset("test-fixture", Path("data"), root)
            config = RunConfig(
                **{**base.model_dump(), "checkpoint_policy": "probe-clients"}
            )
            store = ArtifactStore(root)
            store.create_run("run-1", config)
            store.mark_running("run-1")
            bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
            partitions = build_client_partitions(bundle.train_labels, config)
            store.write_partitions("run-1", partitions)
            store.append_event(make_event("run-1", 1, "run.stopped", {}))
            stopped = ExperimentResult(
                run_id="run-1",
                status="stopped",
                schedule=(),
                client_updates=(),
                aggregations=(),
                final_server_state={"weight": torch.tensor([1.0])},
            )

            store.finalize("run-1", "stopped", stopped)

            manifest = store.read_manifest("run-1")
            self.assertEqual(manifest["status"], "stopped")
            self.assertTrue(
                (root / "run-1" / "checkpoints" / "server-final.pt").is_file()
            )
            self.assertFalse(
                (root / "run-1" / "checkpoints" / "probes").exists()
            )

    def test_stopped_probe_policy_keeps_a_partial_probe_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = build_preset("test-fixture", Path("data"), root)
            config = RunConfig(
                **{**base.model_dump(), "checkpoint_policy": "probe-clients"}
            )
            store = ArtifactStore(root)
            store.create_run("run-1", config)
            bundle = build_synthetic_bundle(config.seed, 10, 300, 100)
            store.write_partitions(
                "run-1",
                build_client_partitions(bundle.train_labels, config),
            )
            store.append_event(make_event("run-1", 1, "run.stopped", {}))
            partial = fixture_result(client_ids=(2,))
            stopped = ExperimentResult(
                run_id=partial.run_id,
                status="stopped",
                schedule=partial.schedule,
                client_updates=partial.client_updates,
                aggregations=partial.aggregations,
                final_server_state=partial.final_server_state,
            )

            store.finalize("run-1", "stopped", stopped)

            probe_dir = root / "run-1" / "checkpoints" / "probes"
            self.assertTrue((probe_dir / "client-2.pt").is_file())
            self.assertFalse((probe_dir / "client-3.pt").exists())


if __name__ == "__main__":
    unittest.main()
