import asyncio
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import torch

from fedrbtvis.artifacts import ArtifactStore
from fedrbtvis.config import RunConfig
from fedrbtvis.data import build_synthetic_bundle
from fedrbtvis.engine import ExperimentResult
from fedrbtvis.events import TrainingEvent
from fedrbtvis.manager import RunAlreadyActiveError, RunManager
from fedrbtvis.presets import build_preset


def fixture_config(root: Path) -> RunConfig:
    return build_preset("test-fixture", Path("data"), root)


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


def empty_result(run_id: str, status: str = "completed") -> ExperimentResult:
    return ExperimentResult(
        run_id=run_id,
        status=status,
        schedule=(),
        client_updates=(),
        aggregations=(),
        final_server_state={},
    )


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
        return empty_result(run_id)

    return run


def replay_race_runner(ready: threading.Event, gate: threading.Event):
    def run(
        run_id,
        config,
        bundle,
        device,
        on_event,
        stop_requested,
    ):
        on_event(event(run_id, 1, "run.started"))
        ready.set()
        gate.wait(timeout=5)
        on_event(event(run_id, 2, "run.completed"))
        return empty_result(run_id)

    return run


def one_event_runner():
    def run(
        run_id,
        config,
        bundle,
        device,
        on_event,
        stop_requested,
    ):
        on_event(event(run_id, 1, "run.started"))
        on_event(event(run_id, 2, "run.completed"))
        return empty_result(run_id)

    return run


def stoppable_runner():
    def run(
        run_id,
        config,
        bundle,
        device,
        on_event,
        stop_requested,
    ):
        on_event(event(run_id, 1, "run.started"))
        if not stop_requested():
            raise AssertionError("stop flag was not visible to the runner")
        on_event(event(run_id, 2, "run.stop_requested"))
        on_event(event(run_id, 3, "run.stopped"))
        return empty_result(run_id, "stopped")

    return run


def shutdown_runner():
    def run(
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
            raise AssertionError("shutdown did not set the stop flag")
        on_event(event(run_id, 2, "run.stop_requested"))
        on_event(event(run_id, 3, "run.stopped"))
        return empty_result(run_id, "stopped")

    return run


def failing_runner(error: Exception):
    def run(
        run_id,
        config,
        bundle,
        device,
        on_event,
        stop_requested,
    ):
        raise error

    return run


class FinalizeFailingStore(ArtifactStore):
    def finalize(self, run_id, status, result, terminal_event=None) -> None:
        raise OSError("simulated finalization failure")


class RunManagerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_manager(
        self,
        runner: Callable,
        store: ArtifactStore | None = None,
    ) -> RunManager:
        def bundle_loader(config: RunConfig):
            return build_synthetic_bundle(config.seed, config.num_classes, 300, 100)

        return RunManager(
            store=store or ArtifactStore(self.root),
            bundle_loader=bundle_loader,
            runner=runner,
            device_resolver=lambda config: torch.device("cpu"),
        )

    async def test_second_active_run_is_rejected(self) -> None:
        gate = threading.Event()
        manager = self.make_manager(blocking_runner(gate))
        first = await manager.create_run(fixture_config(self.root))
        try:
            with self.assertRaises(RunAlreadyActiveError):
                await manager.create_run(fixture_config(self.root))
        finally:
            gate.set()
        await manager.wait(first.run_id)

    async def test_thread_event_is_persisted_then_broadcast(self) -> None:
        manager = self.make_manager(one_event_runner())
        run = await manager.create_run(fixture_config(self.root))
        subscription = manager.subscribe(run.run_id, after_sequence=0)

        received = await asyncio.wait_for(anext(subscription), timeout=5)

        self.assertEqual(received.type, "run.started")
        persisted = manager.store.read_events(run.run_id, 0)
        self.assertEqual(persisted[0].event_id, received.event_id)
        await subscription.aclose()
        await manager.wait(run.run_id)

    async def test_stop_transitions_to_stopped(self) -> None:
        manager = self.make_manager(stoppable_runner())
        run = await manager.create_run(fixture_config(self.root))

        await manager.request_stop(run.run_id)
        await manager.wait(run.run_id)

        self.assertEqual(manager.get_run(run.run_id)["status"], "stopped")

    async def test_shutdown_stops_and_awaits_active_run(self) -> None:
        manager = self.make_manager(shutdown_runner())
        run = await manager.create_run(fixture_config(self.root))

        await manager.shutdown()

        self.assertEqual(manager.get_run(run.run_id)["status"], "stopped")
        self.assertEqual(manager._tasks, {})
        self.assertEqual(manager._stop_flags, {})

    async def test_exception_becomes_failed_with_public_message(self) -> None:
        manager = self.make_manager(failing_runner(ValueError("bad partition")))
        run = await manager.create_run(fixture_config(self.root))

        with self.assertLogs("fedrbtvis.manager", level="ERROR"):
            await manager.wait(run.run_id)

        manifest = manager.get_run(run.run_id)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["error_code"], "RUN_EXECUTION_FAILED")
        self.assertEqual(manifest["error_message"], "run execution failed")
        events = manager.events(run.run_id, 0)
        self.assertEqual([item.type for item in events], ["run.failed"])

    async def test_failure_message_redacts_configured_absolute_paths(self) -> None:
        private_data = self.root / "private-data"
        base = fixture_config(self.root)
        config = RunConfig.model_validate(
            {**base.model_dump(), "data_dir": private_data}
        )
        manager = self.make_manager(
            failing_runner(FileNotFoundError(f"cache missing at {private_data}"))
        )

        run = await manager.create_run(config)
        with self.assertLogs("fedrbtvis.manager", level="ERROR"):
            await manager.wait(run.run_id)

        manifest = manager.get_run(run.run_id)
        failed = manager.events(run.run_id, 0)[-1]
        self.assertNotIn(str(private_data), manifest["error_message"])
        self.assertNotIn(str(private_data), failed.payload["message"])

    async def test_finalize_failure_replaces_deferred_terminal_event(self) -> None:
        store = FinalizeFailingStore(self.root)
        manager = self.make_manager(one_event_runner(), store)

        run = await manager.create_run(fixture_config(self.root))
        with self.assertLogs("fedrbtvis.manager", level="ERROR"):
            await manager.wait(run.run_id)

        manifest = manager.get_run(run.run_id)
        events = manager.events(run.run_id, 0)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(
            [item.type for item in events],
            ["run.started", "run.failed"],
        )
        self.assertEqual([item.sequence for item in events], [1, 2])

    async def test_stopped_finalize_failure_has_only_failed_terminal(self) -> None:
        store = FinalizeFailingStore(self.root)
        manager = self.make_manager(stoppable_runner(), store)
        run = await manager.create_run(fixture_config(self.root))
        await manager.request_stop(run.run_id)

        with self.assertLogs("fedrbtvis.manager", level="ERROR"):
            await manager.wait(run.run_id)

        manifest = manager.get_run(run.run_id)
        events = manager.events(run.run_id, 0)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(
            [item.type for item in events],
            ["run.started", "run.stop_requested", "run.failed"],
        )

    async def test_terminal_subscription_replays_then_finishes(self) -> None:
        manager = self.make_manager(one_event_runner())
        run = await manager.create_run(fixture_config(self.root))
        await manager.wait(run.run_id)

        async def collect() -> list[TrainingEvent]:
            return [
                item
                async for item in manager.subscribe(run.run_id, after_sequence=0)
            ]

        replayed = await asyncio.wait_for(collect(), timeout=5)

        self.assertEqual(
            [item.type for item in replayed],
            ["run.started", "run.completed"],
        )

    async def test_subscription_drains_events_arriving_after_replay(self) -> None:
        ready = threading.Event()
        gate = threading.Event()
        manager = self.make_manager(replay_race_runner(ready, gate))
        run = await manager.create_run(fixture_config(self.root))
        self.assertTrue(await asyncio.to_thread(ready.wait, 5))
        subscription = manager.subscribe(run.run_id, after_sequence=0)

        first = await asyncio.wait_for(anext(subscription), timeout=5)
        gate.set()
        await manager.wait(run.run_id)
        second = await asyncio.wait_for(anext(subscription), timeout=5)

        self.assertEqual(first.type, "run.started")
        self.assertEqual(second.type, "run.completed")
        await subscription.aclose()


if __name__ == "__main__":
    unittest.main()
