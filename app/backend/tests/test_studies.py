import asyncio
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from pydantic import ValidationError

from fedrbtvis.presets import build_preset
from fedrbtvis.studies import (
    StudyConfig,
    expand_study,
    prepare_study,
    read_observations,
    read_study,
    run_study,
)
import fedrbtvis.studies as studies_module


class StudyExpansionTest(unittest.TestCase):
    def test_whitelisted_grid_expands_in_stable_order(self) -> None:
        spec = StudyConfig(
            preset="research-lite",
            factors={"target_noise": [0.0, 0.2], "lid_k": [10, 20]},
            seeds=[3, 7],
        )

        runs = expand_study(spec, Path("data"), Path("runs"))

        self.assertEqual(len(runs), 8)
        self.assertEqual([run.seed for run in runs[:4]], [3, 3, 3, 3])
        self.assertEqual(
            [
                (run.probes[0].lid_k, run.probes[0].target_noise)
                for run in runs[:4]
            ],
            [(10, 0.0), (10, 0.2), (20, 0.0), (20, 0.2)],
        )

    def test_unknown_factor_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            StudyConfig(
                preset="research-lite",
                factors={"learning_rate": [0.1]},
                seeds=[1],
            )

    def test_duplicate_seeds_and_large_grids_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            StudyConfig(
                preset="research-lite",
                factors={"target_noise": [0.0]},
                seeds=[1, 1],
            )
        with self.assertRaises(ValidationError):
            StudyConfig(
                preset="research-lite",
                factors={
                    "target_noise": [index / 20 for index in range(11)],
                    "target_emd": [index / 20 for index in range(10)],
                },
                seeds=[1],
            )

    def test_grid_revalidates_lid_k_against_sample_count(self) -> None:
        spec = StudyConfig(
            preset="research-lite",
            factors={"sample_count": [10], "lid_k": [10]},
            seeds=[1],
        )

        with self.assertRaises(ValidationError):
            expand_study(spec, Path("data"), Path("runs"))


class RecordingManager:
    def __init__(self, root: Path, statuses: list[str] | None = None) -> None:
        self.store = SimpleNamespace(root=root)
        self.statuses = statuses or ["completed", "completed"]
        self.run_statuses: dict[str, str] = {}
        self.active_runs = 0
        self.maximum_concurrent_runs = 0
        self.created_configs = []

    async def create_run(self, config):
        self.active_runs += 1
        self.maximum_concurrent_runs = max(
            self.maximum_concurrent_runs,
            self.active_runs,
        )
        run_id = f"run-{len(self.created_configs) + 1}"
        self.run_statuses[run_id] = self.statuses[len(self.created_configs)]
        self.created_configs.append(config)
        return SimpleNamespace(run_id=run_id, status="running")

    async def wait(self, run_id: str) -> None:
        self.assert_one_active()
        self.active_runs -= 1

    def assert_one_active(self) -> None:
        if self.active_runs != 1:
            raise AssertionError(f"expected one active run, got {self.active_runs}")

    def get_run(self, run_id: str) -> dict:
        return {"run_id": run_id, "status": self.run_statuses[run_id]}

    def client_metrics(self, run_id: str) -> list[dict]:
        return [
            {
                "cycle": 1,
                "step": 1,
                "client_id": 10,
                "role": "probe",
                "sample_count": 200,
                "target_noise": 0.2,
                "actual_noise": 0.2,
                "target_emd": 0.4,
                "actual_emd": 0.4,
                "lid_k": 20,
                "train_loss": 0.8,
                "test_loss": 0.7,
                "test_accuracy": 0.6,
                "test_correct": 60,
                "test_samples": 100,
                "lid_mean": 2.5,
                "lid_std": 0.3,
                "state_sha256": "abc",
            }
        ]


class HangingManager(RecordingManager):
    def __init__(self, root: Path) -> None:
        super().__init__(root, ["completed"])
        self.wait_started = asyncio.Event()

    async def wait(self, run_id: str) -> None:
        self.wait_started.set()
        await asyncio.Event().wait()


class StudyRunnerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_root = self.root / "runs"
        self.run_root.mkdir()

    def two_runs(self):
        base = build_preset("research-lite", Path("data"), self.run_root)
        return (
            base.model_copy(update={"seed": 3}),
            base.model_copy(update={"seed": 7}),
        )

    async def test_runs_are_created_sequentially(self) -> None:
        manager = RecordingManager(self.run_root)

        result = await run_study("study-1", self.two_runs(), manager)

        self.assertEqual(manager.maximum_concurrent_runs, 1)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.run_ids, ("run-1", "run-2"))
        manifest = result.manifest_path.read_text(encoding="utf-8")
        self.assertIn('"status": "completed"', manifest)

    async def test_failed_run_stops_later_runs_and_keeps_prior_ids(self) -> None:
        manager = RecordingManager(self.run_root, ["completed", "failed"])

        result = await run_study("study-1", self.two_runs(), manager)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.run_ids, ("run-1", "run-2"))
        self.assertEqual(len(manager.created_configs), 2)

    async def test_observations_include_provenance_and_measured_values(self) -> None:
        manager = RecordingManager(self.run_root)

        result = await run_study("study-1", self.two_runs()[:1], manager)
        with result.observations_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "fresh")
        self.assertEqual(rows[0]["study_id"], "study-1")
        self.assertEqual(rows[0]["run_id"], "run-1")
        self.assertEqual(rows[0]["seed"], "3")
        self.assertEqual(rows[0]["actual_noise"], "0.2")
        self.assertEqual(rows[0]["lid_mean"], "2.5")
        self.assertEqual(rows[0]["test_accuracy"], "0.6")
        loaded = read_observations(self.root / "studies", "study-1")
        self.assertEqual(loaded[0]["client_id"], 10)
        self.assertEqual(loaded[0]["test_accuracy"], 0.6)

    async def test_cancelled_study_writes_interrupted_terminal_manifest(self) -> None:
        manager = HangingManager(self.run_root)
        task = asyncio.create_task(
            run_study("study-1", self.two_runs()[:1], manager)
        )
        await asyncio.wait_for(manager.wait_started.wait(), timeout=5)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        manifest = read_study(self.root / "studies", "study-1")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["error_code"], "PROCESS_INTERRUPTED")

    def test_reindex_marks_interrupted_study_failed(self) -> None:
        configs = self.two_runs()[:1]
        prepare_study("study-1", configs, self.root / "studies")
        manifest_path = self.root / "studies" / "study-1" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        studies_module.reindex_interrupted_studies(self.root / "studies")

        recovered = read_study(self.root / "studies", "study-1")
        self.assertEqual(recovered["status"], "failed")
        self.assertEqual(recovered["error_code"], "PROCESS_INTERRUPTED")


if __name__ == "__main__":
    unittest.main()
