from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import torch

from fedrbtvis.artifacts import (
    ArtifactCorruptError,
    ArtifactInvariantError,
    ArtifactNotFoundError,
    ArtifactStore,
)
from fedrbtvis.config import RunConfig
from fedrbtvis.data import DatasetBundle, build_client_partitions
from fedrbtvis.engine import ExperimentResult
from fedrbtvis.events import TrainingEvent

ACTIVE_STATES = frozenset({"queued", "running", "stopping"})
TERMINAL_STATES = frozenset({"stopped", "completed", "failed"})
TERMINAL_EVENTS = frozenset({"run.stopped", "run.completed", "run.failed"})
logger = logging.getLogger(__name__)


class RunAlreadyActiveError(RuntimeError):
    pass


class RunNotFoundError(KeyError):
    pass


class RunStateError(RuntimeError):
    pass


class RunNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class CreatedRun:
    run_id: str
    status: str


class RunManager:
    def __init__(
        self,
        store: ArtifactStore,
        bundle_loader: Callable[[RunConfig], DatasetBundle],
        runner: Callable[..., ExperimentResult],
        device_resolver: Callable[[RunConfig], torch.device],
    ) -> None:
        self.store = store
        self._bundle_loader = bundle_loader
        self._runner = runner
        self._device_resolver = device_resolver
        self._state_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_flags: dict[str, threading.Event] = {}
        self._pending_terminal_events: dict[str, TrainingEvent] = {}
        self._subscribers: dict[
            str,
            set[asyncio.Queue[TrainingEvent | None]],
        ] = {}

    def _manifest(self, run_id: str) -> dict[str, Any]:
        try:
            return self.store.read_manifest(run_id)
        except (ArtifactInvariantError, ArtifactNotFoundError) as error:
            raise RunNotFoundError(run_id) from error

    async def create_run(self, config: RunConfig) -> CreatedRun:
        async with self._state_lock:
            if any(
                manifest.get("status") in ACTIVE_STATES
                for manifest in self.store.list_manifests()
            ):
                raise RunAlreadyActiveError("another run is already active")

            run_id = str(uuid4())
            self.store.create_run(run_id, config)
            self.store.mark_running(run_id)
            stop_flag = threading.Event()
            self._stop_flags[run_id] = stop_flag
            self._subscribers[run_id] = set()
            loop = asyncio.get_running_loop()
            task = asyncio.create_task(
                self._execute(run_id, config, loop, stop_flag),
                name=f"fedrbtvis-run-{run_id}",
            )
            self._tasks[run_id] = task
            return CreatedRun(run_id=run_id, status="running")

    def _run_sync(
        self,
        run_id: str,
        config: RunConfig,
        loop: asyncio.AbstractEventLoop,
        stop_flag: threading.Event,
    ) -> None:
        bundle = self._bundle_loader(config)
        partitions = build_client_partitions(bundle.train_labels, config)
        self.store.write_partitions(run_id, partitions)
        device = self._device_resolver(config)

        def accept_from_thread(event: TrainingEvent) -> None:
            acknowledgement: Future[None] = Future()
            loop.call_soon_threadsafe(
                self._accept_event,
                event,
                acknowledgement,
                stop_flag,
            )
            acknowledgement.result()

        result = self._runner(
            run_id,
            config,
            bundle,
            device,
            accept_from_thread,
            stop_flag.is_set,
        )
        terminal = self._pending_terminal_events.get(run_id)
        expected_type = (
            "run.completed" if result.status == "completed" else "run.stopped"
        )
        if terminal is None or terminal.type != expected_type:
            raise RunStateError("runner did not emit its matching terminal event")
        self.store.finalize(
            run_id,
            result.status,
            result,
            terminal_event=terminal,
        )

    def _accept_event(
        self,
        event: TrainingEvent,
        acknowledgement: Future[None],
        stop_flag: threading.Event,
    ) -> None:
        try:
            if event.type in TERMINAL_EVENTS:
                if event.run_id in self._pending_terminal_events:
                    raise RunStateError("runner emitted more than one terminal event")
                self._pending_terminal_events[event.run_id] = event
            else:
                accepted = self.store.append_event(event)
                for queue in tuple(self._subscribers.get(event.run_id, ())):
                    queue.put_nowait(accepted)
        except BaseException as error:
            stop_flag.set()
            acknowledgement.set_exception(error)
        else:
            acknowledgement.set_result(None)

    async def _execute(
        self,
        run_id: str,
        config: RunConfig,
        loop: asyncio.AbstractEventLoop,
        stop_flag: threading.Event,
    ) -> None:
        try:
            await asyncio.to_thread(
                self._run_sync,
                run_id,
                config,
                loop,
                stop_flag,
            )
            self._publish_terminal(run_id)
        except Exception:
            logger.exception("run execution failed: %s", run_id)
            self._record_failure(run_id)
        finally:
            for queue in tuple(self._subscribers.get(run_id, ())):
                queue.put_nowait(None)
            self._tasks.pop(run_id, None)
            self._stop_flags.pop(run_id, None)
            self._pending_terminal_events.pop(run_id, None)
            self._subscribers.pop(run_id, None)

    def _publish_terminal(self, run_id: str) -> None:
        terminal = self._pending_terminal_events.pop(run_id, None)
        if terminal is None:
            raise RunStateError("terminal event is missing after finalization")
        for queue in tuple(self._subscribers.get(run_id, ())):
            queue.put_nowait(terminal)

    def _record_failure(self, run_id: str) -> None:
        message = "run execution failed"
        pending = self._pending_terminal_events.pop(run_id, None)
        last_sequence = self.store.last_sequence(run_id)
        sequence = (
            pending.sequence
            if pending is not None and pending.sequence >= last_sequence
            else last_sequence + 1
        )
        failed = TrainingEvent(
            schema_version=1,
            event_id=str(uuid4()),
            run_id=run_id,
            sequence=sequence,
            type="run.failed",
            created_at=datetime.now(timezone.utc).isoformat(),
            payload={
                "error_code": "RUN_EXECUTION_FAILED",
                "message": message,
            },
        )
        try:
            if pending is not None and last_sequence == pending.sequence:
                accepted = self.store.replace_terminal_event(failed)
            else:
                accepted = self.store.append_event(failed)
        except Exception:
            logger.exception("failed to persist run.failed event: %s", run_id)
            accepted = None
        if accepted is not None:
            for queue in tuple(self._subscribers.get(run_id, ())):
                queue.put_nowait(accepted)
        self.store.fail(run_id, "RUN_EXECUTION_FAILED", message)

    async def request_stop(self, run_id: str) -> dict[str, Any]:
        async with self._state_lock:
            manifest = self._manifest(run_id)
            status = manifest.get("status")
            if status in TERMINAL_STATES:
                return manifest
            if status not in ACTIVE_STATES:
                raise RunStateError(f"run cannot stop from state {status}")
            stop_flag = self._stop_flags.get(run_id)
            if stop_flag is None:
                raise RunStateError("active run has no stop controller")
            manifest = self.store.mark_stopping(run_id)
            stop_flag.set()
            return manifest

    async def wait(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.shield(task)
            return
        manifest = self._manifest(run_id)
        if manifest.get("status") not in TERMINAL_STATES:
            raise RunStateError("active run has no execution task")

    async def shutdown(self) -> None:
        async with self._state_lock:
            tasks = tuple(self._tasks.items())
            for run_id, task in tasks:
                if task.done():
                    continue
                stop_flag = self._stop_flags.get(run_id)
                if stop_flag is not None:
                    stop_flag.set()
                manifest = self._manifest(run_id)
                if manifest.get("status") in {"queued", "running"}:
                    self.store.mark_stopping(run_id)
        if tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for _, task in tasks),
                return_exceptions=True,
            )

    async def subscribe(
        self,
        run_id: str,
        after_sequence: int,
    ) -> AsyncIterator[TrainingEvent]:
        self._manifest(run_id)
        queue: asyncio.Queue[TrainingEvent | None] = asyncio.Queue()
        subscribers = self._subscribers.setdefault(run_id, set())
        subscribers.add(queue)
        seen: set[str] = set()
        try:
            replay = self.store.read_events(run_id, after_sequence)
            for item in replay:
                if (
                    item.type in TERMINAL_EVENTS
                    and self._manifest(run_id).get("status") in ACTIVE_STATES
                ):
                    continue
                if item.event_id in seen:
                    continue
                seen.add(item.event_id)
                yield item
                if item.type in TERMINAL_EVENTS:
                    return

            while True:
                if (
                    queue.empty()
                    and self._manifest(run_id).get("status") in TERMINAL_STATES
                ):
                    return
                item = await queue.get()
                if item is None:
                    return
                if item.event_id in seen or item.sequence <= after_sequence:
                    continue
                seen.add(item.event_id)
                yield item
                if item.type in TERMINAL_EVENTS:
                    return
        finally:
            subscribers.discard(queue)

    def list_runs(self) -> list[dict[str, Any]]:
        return self.store.list_manifests()

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._manifest(run_id)

    def events(self, run_id: str, after_sequence: int) -> list[TrainingEvent]:
        manifest = self._manifest(run_id)
        events = self.store.read_events(run_id, after_sequence)
        if manifest.get("status") in ACTIVE_STATES:
            return [item for item in events if item.type not in TERMINAL_EVENTS]
        return events

    def client_metrics(self, run_id: str) -> list[dict[str, Any]]:
        manifest = self._manifest(run_id)
        if manifest.get("status") not in {"completed", "stopped"}:
            raise RunNotReadyError("run metrics are not available")
        try:
            return self.store.read_client_metrics(run_id)
        except ArtifactNotFoundError as error:
            raise ArtifactCorruptError("required client metrics are missing") from error

    def aggregation_metrics(self, run_id: str) -> list[dict[str, Any]]:
        manifest = self._manifest(run_id)
        if manifest.get("status") not in {"completed", "stopped"}:
            raise RunNotReadyError("run metrics are not available")
        try:
            return self.store.read_aggregation_metrics(run_id)
        except ArtifactNotFoundError as error:
            raise ArtifactCorruptError(
                "required aggregation metrics are missing"
            ) from error
