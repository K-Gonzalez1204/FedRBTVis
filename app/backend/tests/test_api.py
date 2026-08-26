import importlib
import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from fedrbtvis.api import create_app
from fedrbtvis.artifacts import ArtifactStore
from fedrbtvis.config import RunConfig
from fedrbtvis.data import DatasetBundle, build_synthetic_bundle
from fedrbtvis.engine import ExperimentResult
from fedrbtvis.events import TrainingEvent
from fedrbtvis.legacy import LegacyRepository
from fedrbtvis.manager import RunManager


def event(run_id: str, sequence: int, event_type: str) -> TrainingEvent:
    return TrainingEvent(
        schema_version=1,
        event_id=f"{run_id}-{sequence}",
        run_id=run_id,
        sequence=sequence,
        type=event_type,
        created_at="2026-08-01T00:00:00+00:00",
        payload={},
    )


def fixture_runner(
    run_id,
    config,
    bundle,
    device,
    on_event,
    stop_requested,
):
    for sequence, event_type in enumerate(
        (
            "run.started",
            "client.completed",
            "aggregation.completed",
            "run.completed",
        ),
        start=1,
    ):
        on_event(event(run_id, sequence, event_type))
    return ExperimentResult(
        run_id=run_id,
        status="completed",
        schedule=(),
        client_updates=(),
        aggregations=(),
        final_server_state={},
    )


def path_failing_runner(
    run_id,
    config,
    bundle,
    device,
    on_event,
    stop_requested,
):
    raise FileNotFoundError(f"cache missing at {config.data_dir}")


def blocking_runner(gate: threading.Event):
    def run(
        run_id,
        config,
        bundle,
        device,
        on_event,
        stop_requested,
    ):
        on_event(event(run_id, 1, "run.started"))
        gate.wait(timeout=5)
        on_event(event(run_id, 2, "run.completed"))
        return ExperimentResult(
            run_id=run_id,
            status="completed",
            schedule=(),
            client_updates=(),
            aggregations=(),
            final_server_state={},
        )

    return run


def cooperative_shutdown_runner(
    run_id,
    config,
    bundle,
    device,
    on_event,
    stop_requested,
):
    on_event(event(run_id, 1, "run.started"))
    deadline = time.monotonic() + 5
    while not stop_requested() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not stop_requested():
        raise AssertionError("app shutdown did not request run stop")
    on_event(event(run_id, 2, "run.stop_requested"))
    on_event(event(run_id, 3, "run.stopped"))
    return ExperimentResult(
        run_id=run_id,
        status="stopped",
        schedule=(),
        client_updates=(),
        aggregations=(),
        final_server_state={},
    )


EVIDENCE_DIR = Path(__file__).resolve().parents[3] / "evidence" / "legacy"


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        store = ArtifactStore(self.root)

        def bundle_loader(config: RunConfig):
            return build_synthetic_bundle(config.seed, config.num_classes, 300, 100)

        self.manager = RunManager(
            store=store,
            bundle_loader=bundle_loader,
            runner=fixture_runner,
            device_resolver=lambda config: torch.device("cpu"),
        )
        self.app = create_app(self.manager)

    def create_and_finish(self, client: TestClient) -> str:
        response = client.post("/api/runs", json={"preset": "test-fixture"})
        self.assertEqual(response.status_code, 202)
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            manifest = client.get(f"/api/runs/{run_id}").json()
            if manifest["status"] in {"completed", "failed", "stopped"}:
                self.assertEqual(
                    manifest["status"],
                    "completed",
                    manifest.get("error_message"),
                )
                return run_id
            time.sleep(0.01)
        self.fail("fixture run did not become terminal")

    def wait_until_completed(self, client: TestClient, run_id: str) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            manifest = client.get(f"/api/runs/{run_id}").json()
            if manifest["status"] in {"completed", "failed", "stopped"}:
                self.assertEqual(
                    manifest["status"],
                    "completed",
                    manifest.get("error_message"),
                )
                return
            time.sleep(0.01)
        self.fail("fixture run did not complete")

    def test_invalid_config_returns_stable_error(self) -> None:
        with TestClient(self.app) as client:
            response = client.post(
                "/api/runs",
                json={"preset": "test-fixture", "overrides": {"cycles": 0}},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_CONFIG")

    def test_non_whitelisted_override_is_rejected(self) -> None:
        with TestClient(self.app) as client:
            response = client.post(
                "/api/runs",
                json={
                    "preset": "test-fixture",
                    "overrides": {"data_dir": "elsewhere"},
                },
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_CONFIG")

    def test_unknown_run_is_not_200(self) -> None:
        with TestClient(self.app) as client:
            response = client.get("/api/runs/missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "RUN_NOT_FOUND")

    def test_invalid_run_id_uses_stable_not_found_error(self) -> None:
        with TestClient(self.app) as client:
            response = client.get("/api/runs/bad!")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "RUN_NOT_FOUND")

    def test_second_active_run_returns_conflict(self) -> None:
        gate = threading.Event()
        manager = RunManager(
            store=ArtifactStore(self.root),
            bundle_loader=lambda config: build_synthetic_bundle(
                config.seed, config.num_classes, 300, 100
            ),
            runner=blocking_runner(gate),
            device_resolver=lambda config: torch.device("cpu"),
        )
        with TestClient(create_app(manager)) as client:
            first = client.post("/api/runs", json={"preset": "test-fixture"})
            second = client.post("/api/runs", json={"preset": "test-fixture"})
            gate.set()
            self.wait_until_completed(client, first.json()["run_id"])

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(
            second.json()["error"]["code"],
            "RUN_ALREADY_ACTIVE",
        )

    def test_run_and_metric_routes_return_envelopes(self) -> None:
        with TestClient(self.app) as client:
            presets = client.get("/api/presets")
            run_id = self.create_and_finish(client)
            runs = client.get("/api/runs")
            events = client.get(f"/api/runs/{run_id}/events")
            clients = client.get(f"/api/runs/{run_id}/metrics/clients")
            aggregations = client.get(
                f"/api/runs/{run_id}/metrics/aggregations"
            )

        self.assertEqual(presets.status_code, 200)
        self.assertEqual(len(presets.json()["items"]), 3)
        self.assertEqual(runs.json()["items"][0]["run_id"], run_id)
        self.assertEqual(len(events.json()["items"]), 4)
        self.assertEqual(clients.json(), {"items": []})
        self.assertEqual(aggregations.json(), {"items": []})

    def test_corrupt_artifact_returns_sanitized_error(self) -> None:
        with TestClient(self.app) as client:
            run_id = self.create_and_finish(client)
            events_path = self.root / run_id / "events.jsonl"
            events_path.write_text("not-json\n", encoding="utf-8")
            with self.assertLogs("fedrbtvis.api", level="ERROR"):
                response = client.get(f"/api/runs/{run_id}/events")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "ARTIFACT_CORRUPT")
        self.assertNotIn(str(self.root), response.json()["error"]["message"])

    def test_active_metrics_return_run_not_ready(self) -> None:
        gate = threading.Event()
        manager = RunManager(
            store=ArtifactStore(self.root / "active-runs"),
            bundle_loader=lambda config: build_synthetic_bundle(
                config.seed, config.num_classes, 300, 100
            ),
            runner=blocking_runner(gate),
            device_resolver=lambda config: torch.device("cpu"),
        )
        with TestClient(create_app(manager)) as client:
            created = client.post("/api/runs", json={"preset": "test-fixture"})
            run_id = created.json()["run_id"]
            try:
                response = client.get(f"/api/runs/{run_id}/metrics/clients")
            finally:
                gate.set()
                self.wait_until_completed(client, run_id)

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "RUN_NOT_READY")

    def test_app_shutdown_cooperatively_stops_active_run(self) -> None:
        manager = RunManager(
            store=ArtifactStore(self.root / "shutdown-runs"),
            bundle_loader=lambda config: build_synthetic_bundle(
                config.seed, config.num_classes, 300, 100
            ),
            runner=cooperative_shutdown_runner,
            device_resolver=lambda config: torch.device("cpu"),
        )
        app = create_app(manager)
        with TestClient(app) as client:
            created = client.post("/api/runs", json={"preset": "test-fixture"})
            run_id = created.json()["run_id"]

        self.assertEqual(manager.get_run(run_id)["status"], "stopped")

    def test_missing_terminal_metrics_are_artifact_corrupt(self) -> None:
        with TestClient(self.app) as client:
            run_id = self.create_and_finish(client)
            (self.root / run_id / "metrics" / "client_updates.csv").unlink()
            with self.assertLogs("fedrbtvis.api", level="ERROR"):
                response = client.get(f"/api/runs/{run_id}/metrics/clients")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "ARTIFACT_CORRUPT")

    def test_failed_run_api_never_exposes_local_paths(self) -> None:
        manager = RunManager(
            store=ArtifactStore(self.root / "failed-runs"),
            bundle_loader=lambda config: build_synthetic_bundle(
                config.seed, config.num_classes, 300, 100
            ),
            runner=path_failing_runner,
            device_resolver=lambda config: torch.device("cpu"),
        )
        with self.assertLogs("fedrbtvis.manager", level="ERROR"):
            with TestClient(create_app(manager)) as client:
                created = client.post(
                    "/api/runs",
                    json={"preset": "test-fixture"},
                )
                run_id = created.json()["run_id"]
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    manifest = client.get(f"/api/runs/{run_id}")
                    if manifest.json()["status"] == "failed":
                        break
                    time.sleep(0.01)
                events = client.get(f"/api/runs/{run_id}/events")
                runs = client.get("/api/runs")

        public_payload = f"{manifest.text}\n{events.text}\n{runs.text}"
        self.assertNotIn(str(self.root), public_payload)

    def test_corrupt_websocket_replay_closes_with_stable_reason(self) -> None:
        with TestClient(self.app) as client:
            run_id = self.create_and_finish(client)
            (self.root / run_id / "events.jsonl").write_text(
                "not-json\n",
                encoding="utf-8",
            )
            with self.assertLogs("fedrbtvis.api", level="ERROR"):
                with self.assertRaises(WebSocketDisconnect) as captured:
                    with client.websocket_connect(
                        f"/ws/runs/{run_id}?after_sequence=0"
                    ):
                        pass

        self.assertEqual(captured.exception.code, 1011)
        self.assertEqual(captured.exception.reason, "ARTIFACT_CORRUPT")

    def test_websocket_replays_intermediate_events(self) -> None:
        with TestClient(self.app) as client:
            response = client.post("/api/runs", json={"preset": "test-fixture"})
            self.assertEqual(response.status_code, 202)
            run_id = response.json()["run_id"]
            with client.websocket_connect(
                f"/ws/runs/{run_id}?after_sequence=0"
            ) as socket:
                received = []
                while True:
                    item = socket.receive_json()
                    received.append(item)
                    if item["type"] in {
                        "run.completed",
                        "run.failed",
                        "run.stopped",
                    }:
                        break

        self.assertEqual(
            [item["sequence"] for item in received],
            list(range(1, len(received) + 1)),
        )
        self.assertIn("client.completed", [item["type"] for item in received])
        self.assertEqual(received[-1]["type"], "run.completed")

    def test_after_sequence_avoids_duplicate_replay(self) -> None:
        with TestClient(self.app) as client:
            run_id = self.create_and_finish(client)
            events = client.get(f"/api/runs/{run_id}/events").json()["items"]
            after = events[-2]["sequence"]
            with client.websocket_connect(
                f"/ws/runs/{run_id}?after_sequence={after}"
            ) as socket:
                final = socket.receive_json()

        self.assertEqual(final["sequence"], events[-1]["sequence"])

    def test_main_uses_environment_paths(self) -> None:
        data_dir = self.root / "configured-data"
        run_dir = self.root / "configured-runs"
        environment = {
            "FEDRBTVIS_DATA_DIR": str(data_dir),
            "FEDRBTVIS_RUN_DIR": str(run_dir),
            "FEDRBTVIS_DEVICE": "cpu",
        }

        with patch.dict(os.environ, environment, clear=False):
            main = importlib.import_module("fedrbtvis.main")

        self.assertEqual(main.app.state.data_dir, data_dir.resolve())
        self.assertTrue(run_dir.is_dir())

    def test_study_routes_execute_grid_and_return_observations(self) -> None:
        study_run_root = self.root / "study-runs"

        def study_bundle(config: RunConfig) -> DatasetBundle:
            train_size = 7_000
            return DatasetBundle(
                train_images=torch.empty((train_size, 1)),
                train_labels=np.arange(train_size, dtype=np.int64) % 10,
                test_images=torch.empty((10, 1)),
                test_labels=np.arange(10, dtype=np.int64),
            )

        manager = RunManager(
            store=ArtifactStore(study_run_root),
            bundle_loader=study_bundle,
            runner=fixture_runner,
            device_resolver=lambda config: torch.device("cpu"),
        )
        app = create_app(manager)
        app.state.study_root = self.root / "studies"
        with TestClient(app) as client:
            created = client.post(
                "/api/studies",
                json={
                    "preset": "research-lite",
                    "factors": {"target_noise": [0.2]},
                    "seeds": [3],
                },
            )
            self.assertEqual(created.status_code, 202)
            study_id = created.json()["study_id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                manifest = client.get(f"/api/studies/{study_id}")
                if manifest.json()["status"] in {"completed", "failed", "stopped"}:
                    break
                time.sleep(0.01)
            observations = client.get(
                f"/api/studies/{study_id}/observations"
            )

        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(
            manifest.json()["status"],
            "completed",
            manager.list_runs(),
        )
        self.assertEqual(observations.status_code, 200)
        self.assertEqual(observations.json(), {"items": []})

    def test_study_routes_use_stable_validation_and_missing_codes(self) -> None:
        self.app.state.study_root = self.root / "studies"
        with TestClient(self.app) as client:
            invalid = client.post(
                "/api/studies",
                json={
                    "preset": "research-lite",
                    "factors": {"learning_rate": [0.1]},
                    "seeds": [1],
                },
            )
            missing = client.get("/api/studies/missing")

        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["error"]["code"], "INVALID_CONFIG")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "STUDY_NOT_FOUND")

    def test_legacy_observations_success(self) -> None:
        repository = LegacyRepository.from_directory(EVIDENCE_DIR)
        app = create_app(self.manager, legacy_repository=repository)
        with TestClient(app) as client:
            response = client.get("/api/observations/legacy")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["manifest"]["rows"], 4500)
        self.assertEqual(len(payload["items"]), 4500)
        first = payload["items"][0]
        self.assertEqual(first["source"], "legacy")
        self.assertIn("inferred_target_noise", first)

    def test_legacy_observations_not_imported(self) -> None:
        with TestClient(self.app) as client:
            response = client.get("/api/observations/legacy")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["error"]["code"],
            "LEGACY_NOT_IMPORTED",
        )

    def test_legacy_observations_corrupt(self) -> None:
        app = create_app(self.manager, legacy_error="ARTIFACT_CORRUPT")
        with TestClient(app) as client:
            with self.assertLogs("fedrbtvis.api", level="ERROR"):
                response = client.get("/api/observations/legacy")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "ARTIFACT_CORRUPT")


if __name__ == "__main__":
    unittest.main()
