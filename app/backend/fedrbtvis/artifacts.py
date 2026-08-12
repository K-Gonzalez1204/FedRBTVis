from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch

from fedrbtvis.config import RunConfig
from fedrbtvis.data import ClientPartition
from fedrbtvis.engine import ExperimentResult
from fedrbtvis.events import JsonValue, TrainingEvent


class ArtifactInvariantError(RuntimeError):
    pass


class ArtifactNotFoundError(FileNotFoundError):
    pass


class ArtifactCorruptError(RuntimeError):
    pass


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ACTIVE_STATUSES = frozenset({"queued", "running", "stopping"})
_TERMINAL_STATUSES = frozenset({"completed", "stopped", "failed"})
_RUN_STATUSES = _ACTIVE_STATUSES | _TERMINAL_STATUSES
_TERMINAL_EVENTS = frozenset({"run.completed", "run.stopped", "run.failed"})
_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "run_id",
    "preset",
    "source",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "error_code",
    "error_message",
    "files",
})
_CLIENT_HEADERS = (
    "cycle",
    "step",
    "client_id",
    "role",
    "sample_count",
    "target_noise",
    "actual_noise",
    "target_emd",
    "actual_emd",
    "lid_k",
    "train_loss",
    "test_loss",
    "test_accuracy",
    "test_correct",
    "test_samples",
    "lid_mean",
    "lid_std",
    "state_sha256",
)
_AGGREGATION_HEADERS = (
    "cycle",
    "step",
    "client_ids",
    "test_loss",
    "test_accuracy",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def _event_line(event: TrainingEvent) -> str:
    return json.dumps(
        asdict(event),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(path, _json_text(value).encode("utf-8"))


def _atomic_write_csv(
    path: Path,
    headers: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(headers),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def _read_text(path: Path) -> str:
    for attempt in range(10):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.001 * (attempt + 1))
    raise AssertionError("unreachable")


def _replace_file(source: Path, destination: Path) -> None:
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.001 * (attempt + 1))


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4()}.tmp"
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        _replace_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._last_sequences: dict[str, int] = {}
        self._reindex_interrupted_runs()

    def _validate_run_id(self, run_id: str) -> None:
        if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
            raise ArtifactInvariantError("run_id contains unsafe path characters")

    def _run_dir(self, run_id: str, *, must_exist: bool = True) -> Path:
        self._validate_run_id(run_id)
        path = self.root / run_id
        if must_exist and not path.is_dir():
            raise ArtifactNotFoundError(f"run not found: {run_id}")
        return path

    def _lock_for(self, run_id: str) -> threading.Lock:
        return self._locks.setdefault(run_id, threading.Lock())

    def _read_json(self, path: Path) -> object:
        if not path.is_file():
            raise ArtifactCorruptError(f"required artifact is missing: {path.name}")
        try:
            return json.loads(_read_text(path))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ArtifactCorruptError(
                f"artifact is not valid JSON: {path.name}"
            ) from error

    def _file_inventory(self, run_dir: Path) -> list[dict[str, JsonValue]]:
        items: list[dict[str, JsonValue]] = []
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                continue
            items.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return items

    def _write_manifest(self, run_id: str, manifest: dict[str, JsonValue]) -> None:
        run_dir = self._run_dir(run_id)
        manifest["files"] = self._file_inventory(run_dir)
        _atomic_write_json(run_dir / "manifest.json", manifest)

    def _verify_manifest_inventory(
        self,
        run_dir: Path,
        manifest: dict[str, JsonValue],
    ) -> None:
        recorded = manifest.get("files")
        if not isinstance(recorded, list):
            raise ArtifactCorruptError("manifest file inventory is invalid")
        current = self._file_inventory(run_dir)
        if recorded != current:
            raise ArtifactCorruptError("terminal artifact inventory does not match")

    def _validate_manifest(
        self,
        run_id: str,
        manifest: dict[str, JsonValue],
    ) -> None:
        if not _MANIFEST_FIELDS.issubset(manifest):
            raise ArtifactCorruptError("manifest is missing required fields")
        if type(manifest.get("schema_version")) is not int or manifest.get(
            "schema_version"
        ) != 1:
            raise ArtifactCorruptError("manifest schema version is invalid")
        if manifest.get("run_id") != run_id:
            raise ArtifactCorruptError("manifest identity is invalid")
        status = manifest.get("status")
        if not isinstance(status, str) or status not in _RUN_STATUSES:
            raise ArtifactCorruptError("manifest status is invalid")
        for field in ("preset", "source", "created_at"):
            if not isinstance(manifest.get(field), str) or not manifest[field]:
                raise ArtifactCorruptError(f"manifest {field} is invalid")
        for field in ("started_at", "finished_at", "error_code", "error_message"):
            if manifest.get(field) is not None and not isinstance(
                manifest[field],
                str,
            ):
                raise ArtifactCorruptError(f"manifest {field} is invalid")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise ArtifactCorruptError("manifest file inventory is invalid")
        for item in files:
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "bytes", "sha256"}
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or item["path"].startswith("/")
                or "\\" in item["path"]
                or ".." in item["path"].split("/")
                or not isinstance(item.get("bytes"), int)
                or isinstance(item.get("bytes"), bool)
                or item["bytes"] < 0
                or not isinstance(item.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            ):
                raise ArtifactCorruptError("manifest file inventory is invalid")

    def create_run(
        self,
        run_id: str,
        config: RunConfig,
    ) -> dict[str, JsonValue]:
        run_dir = self._run_dir(run_id, must_exist=False)
        if run_dir.exists():
            raise ArtifactInvariantError(f"run already exists: {run_id}")
        run_dir.mkdir(parents=True)
        self._locks[run_id] = threading.Lock()
        self._last_sequences[run_id] = 0
        _atomic_write_json(
            run_dir / "config.json",
            config.model_dump(mode="json"),
        )
        _atomic_write_bytes(run_dir / "events.jsonl", b"")
        manifest: dict[str, JsonValue] = {
            "schema_version": 1,
            "run_id": run_id,
            "preset": config.preset,
            "source": config.source,
            "status": "queued",
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "error_code": None,
            "error_message": None,
            "files": [],
        }
        self._write_manifest(run_id, manifest)
        return dict(manifest)

    def mark_running(self, run_id: str) -> None:
        with self._lock_for(run_id):
            manifest = self.read_manifest(run_id)
            if manifest["status"] != "queued":
                raise ArtifactInvariantError("only a queued run can start")
            manifest["status"] = "running"
            manifest["started_at"] = _utc_now()
            self._write_manifest(run_id, manifest)

    def mark_stopping(self, run_id: str) -> dict[str, JsonValue]:
        with self._lock_for(run_id):
            manifest = self.read_manifest(run_id)
            if manifest["status"] == "running":
                manifest["status"] = "stopping"
                self._write_manifest(run_id, manifest)
            return manifest

    def _append_event_unlocked(
        self,
        run_dir: Path,
        event: TrainingEvent,
    ) -> TrainingEvent:
        last = self._last_sequences.get(event.run_id)
        if last is None:
            last = self._read_last_sequence(run_dir / "events.jsonl")
        expected = last + 1
        if event.sequence != expected:
            raise ArtifactInvariantError(
                f"event sequence must be {expected}, got {event.sequence}"
            )
        with (run_dir / "events.jsonl").open(
            "a",
            encoding="utf-8",
            newline="",
        ) as handle:
            handle.write(_event_line(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._last_sequences[event.run_id] = event.sequence
        return event

    def append_event(self, event: TrainingEvent) -> TrainingEvent:
        run_dir = self._run_dir(event.run_id)
        with self._lock_for(event.run_id):
            return self._append_event_unlocked(run_dir, event)

    def replace_terminal_event(self, event: TrainingEvent) -> TrainingEvent:
        if event.type != "run.failed":
            raise ArtifactInvariantError("replacement event must be run.failed")
        run_dir = self._run_dir(event.run_id)
        with self._lock_for(event.run_id):
            existing = self.read_events(event.run_id, 0)
            if not existing or existing[-1].type not in _TERMINAL_EVENTS:
                raise ArtifactInvariantError("last event is not terminal")
            if existing[-1].sequence != event.sequence:
                raise ArtifactInvariantError("replacement sequence does not match")
            lines = [_event_line(item) for item in (*existing[:-1], event)]
            _atomic_write_bytes(
                run_dir / "events.jsonl",
                ("\n".join(lines) + "\n").encode("utf-8"),
            )
            self._last_sequences[event.run_id] = event.sequence
            return event

    def write_partitions(
        self,
        run_id: str,
        partitions: Sequence[ClientPartition],
    ) -> None:
        run_dir = self._run_dir(run_id)
        with self._lock_for(run_id):
            _atomic_write_json(
                run_dir / "partitions.json",
                [asdict(partition) for partition in partitions],
            )

    def _client_rows(
        self,
        run_id: str,
        result: ExperimentResult,
    ) -> list[dict[str, object]]:
        completed = [
            event
            for event in self.read_events(run_id, 0)
            if event.type == "client.completed"
        ]
        rows: list[dict[str, object]] = []
        for index, update in enumerate(result.client_updates):
            payload = completed[index].payload if index < len(completed) else {}
            rows.append(
                {
                    "cycle": payload.get("cycle", ""),
                    "step": payload.get("step", ""),
                    "client_id": update.client_id,
                    "role": payload.get("role", ""),
                    "sample_count": update.sample_count,
                    "target_noise": payload.get("target_noise", ""),
                    "actual_noise": update.actual_noise,
                    "target_emd": payload.get("target_categorical_emd_01", ""),
                    "actual_emd": update.actual_emd,
                    "lid_k": payload.get("lid_k", ""),
                    "train_loss": update.train_loss,
                    "test_loss": update.test.loss,
                    "test_accuracy": update.test.accuracy,
                    "test_correct": update.test.correct,
                    "test_samples": update.test.samples,
                    "lid_mean": update.lid_mean,
                    "lid_std": update.lid_std,
                    "state_sha256": update.state_sha256,
                }
            )
        return rows

    def _aggregation_rows(
        self,
        result: ExperimentResult,
    ) -> list[dict[str, object]]:
        return [
            {
                "cycle": row.cycle,
                "step": row.step,
                "client_ids": json.dumps(list(row.client_ids), separators=(",", ":")),
                "test_loss": row.test_loss,
                "test_accuracy": row.test_accuracy,
            }
            for row in result.aggregations
        ]

    def _write_checkpoints(
        self,
        run_id: str,
        result: ExperimentResult,
    ) -> None:
        run_dir = self._run_dir(run_id)
        config = self._read_json(run_dir / "config.json")
        if not isinstance(config, dict):
            raise ArtifactCorruptError("config.json must contain an object")
        policy = config.get("checkpoint_policy")
        if policy == "none":
            return
        if policy not in {"server-only", "probe-clients"}:
            raise ArtifactCorruptError("checkpoint policy is invalid")
        checkpoint_dir = run_dir / "checkpoints"
        _atomic_torch_save(
            checkpoint_dir / "server-final.pt",
            result.final_server_state,
        )
        if policy != "probe-clients":
            return
        partitions = self._read_json(run_dir / "partitions.json")
        if not isinstance(partitions, list):
            raise ArtifactCorruptError("partitions.json must contain a list")
        probe_ids = {
            int(item["client_id"])
            for item in partitions
            if isinstance(item, dict) and item.get("role") == "probe"
        }
        latest = {update.client_id: update for update in result.client_updates}
        for client_id in sorted(probe_ids):
            if client_id not in latest:
                if result.status == "stopped":
                    continue
                raise ArtifactInvariantError(
                    f"probe client {client_id} has no completed state"
                )
            _atomic_torch_save(
                checkpoint_dir / "probes" / f"client-{client_id}.pt",
                latest[client_id].state_dict,
            )

    def finalize(
        self,
        run_id: str,
        status: Literal["completed", "stopped"],
        result: ExperimentResult,
        terminal_event: TrainingEvent | None = None,
    ) -> None:
        if result.run_id != run_id or result.status != status:
            raise ArtifactInvariantError("result identity or status does not match run")
        run_dir = self._run_dir(run_id)
        with self._lock_for(run_id):
            manifest = self.read_manifest(run_id)
            if manifest["status"] not in {"running", "stopping", "queued"}:
                raise ArtifactInvariantError("run is already terminal")
            _atomic_write_csv(
                run_dir / "metrics" / "client_updates.csv",
                _CLIENT_HEADERS,
                self._client_rows(run_id, result),
            )
            _atomic_write_csv(
                run_dir / "metrics" / "aggregations.csv",
                _AGGREGATION_HEADERS,
                self._aggregation_rows(result),
            )
            self._write_checkpoints(run_id, result)
            if terminal_event is not None:
                expected_type = (
                    "run.completed" if status == "completed" else "run.stopped"
                )
                if (
                    terminal_event.run_id != run_id
                    or terminal_event.type != expected_type
                ):
                    raise ArtifactInvariantError(
                        "terminal event does not match final status"
                    )
                self._append_event_unlocked(run_dir, terminal_event)
            manifest["status"] = status
            manifest["finished_at"] = _utc_now()
            manifest["error_code"] = None
            manifest["error_message"] = None
            self._write_manifest(run_id, manifest)

    def fail(self, run_id: str, error_code: str, message: str) -> None:
        with self._lock_for(run_id):
            manifest = self.read_manifest(run_id)
            if manifest["status"] in {"completed", "stopped"}:
                raise ArtifactInvariantError("successful terminal run cannot fail")
            manifest["status"] = "failed"
            manifest["finished_at"] = _utc_now()
            manifest["error_code"] = error_code
            manifest["error_message"] = message
            self._write_manifest(run_id, manifest)

    def read_manifest(self, run_id: str) -> dict[str, JsonValue]:
        run_dir = self._run_dir(run_id)
        value = self._read_json(run_dir / "manifest.json")
        if not isinstance(value, dict):
            raise ArtifactCorruptError("manifest must be a JSON object")
        self._validate_manifest(run_id, value)
        if value.get("status") in _TERMINAL_STATUSES:
            self._verify_manifest_inventory(run_dir, value)
        return value

    def list_manifests(self) -> list[dict[str, JsonValue]]:
        return [
            self.read_manifest(path.name)
            for path in sorted(self.root.iterdir(), key=lambda item: item.name)
            if path.is_dir()
        ]

    def read_events(
        self,
        run_id: str,
        after_sequence: int,
    ) -> list[TrainingEvent]:
        if after_sequence < 0:
            raise ArtifactInvariantError("after_sequence must be non-negative")
        path = self._run_dir(run_id) / "events.jsonl"
        if not path.is_file():
            raise ArtifactCorruptError("events.jsonl is missing")
        events: list[TrainingEvent] = []
        try:
            lines = _read_text(path).splitlines()
            for line in lines:
                value = json.loads(line)
                event = TrainingEvent(**value)
                if event.sequence > after_sequence:
                    events.append(event)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
            raise ArtifactCorruptError("events.jsonl is corrupt") from error
        return events

    def _read_last_sequence(self, path: Path) -> int:
        run_id = path.parent.name
        events = self.read_events(run_id, 0)
        if not events:
            return 0
        expected = list(range(1, len(events) + 1))
        actual = [event.sequence for event in events]
        if actual != expected:
            raise ArtifactCorruptError("persisted event sequence is not contiguous")
        return events[-1].sequence

    def last_sequence(self, run_id: str) -> int:
        if run_id not in self._last_sequences:
            self._last_sequences[run_id] = self._read_last_sequence(
                self._run_dir(run_id) / "events.jsonl"
            )
        return self._last_sequences[run_id]

    def _read_csv(
        self,
        path: Path,
        expected_headers: tuple[str, ...],
        integer_fields: frozenset[str],
        float_fields: frozenset[str],
    ) -> list[dict[str, JsonValue]]:
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact not found: {path.name}")
        try:
            reader = csv.DictReader(StringIO(_read_text(path), newline=""))
            if tuple(reader.fieldnames or ()) != expected_headers:
                raise ArtifactCorruptError(f"CSV schema is invalid: {path.name}")
            rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as error:
            raise ArtifactCorruptError(f"CSV is corrupt: {path.name}") from error
        converted: list[dict[str, JsonValue]] = []
        try:
            for row in rows:
                item: dict[str, JsonValue] = {}
                for key, value in row.items():
                    if key == "client_ids":
                        item[key] = json.loads(value)
                    elif value == "":
                        item[key] = None
                    elif key in integer_fields:
                        item[key] = int(value)
                    elif key in float_fields:
                        item[key] = float(value)
                    else:
                        item[key] = value
                converted.append(item)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ArtifactCorruptError(f"CSV values are corrupt: {path.name}") from error
        return converted

    def read_client_metrics(self, run_id: str) -> list[dict[str, JsonValue]]:
        return self._read_csv(
            self._run_dir(run_id) / "metrics" / "client_updates.csv",
            _CLIENT_HEADERS,
            frozenset({
                "cycle",
                "step",
                "client_id",
                "sample_count",
                "lid_k",
                "test_correct",
                "test_samples",
            }),
            frozenset({
                "target_noise",
                "actual_noise",
                "target_emd",
                "actual_emd",
                "train_loss",
                "test_loss",
                "test_accuracy",
                "lid_mean",
                "lid_std",
            }),
        )

    def read_aggregation_metrics(
        self,
        run_id: str,
    ) -> list[dict[str, JsonValue]]:
        return self._read_csv(
            self._run_dir(run_id) / "metrics" / "aggregations.csv",
            _AGGREGATION_HEADERS,
            frozenset({"cycle", "step"}),
            frozenset({"test_loss", "test_accuracy"}),
        )

    def _reindex_interrupted_runs(self) -> None:
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not path.is_dir():
                continue
            self._validate_run_id(path.name)
            if not (path / "config.json").is_file():
                raise ArtifactCorruptError(
                    f"run {path.name} is missing config.json"
                )
            manifest = self.read_manifest(path.name)
            self._locks[path.name] = threading.Lock()
            self._last_sequences[path.name] = self._read_last_sequence(
                path / "events.jsonl"
            )
            if manifest.get("status") in _ACTIVE_STATUSES:
                events = self.read_events(path.name, 0)
                last = events[-1] if events else None
                sequence = (
                    last.sequence
                    if last is not None and last.type in _TERMINAL_EVENTS
                    else self._last_sequences[path.name] + 1
                )
                interrupted = TrainingEvent(
                    schema_version=1,
                    event_id=str(uuid4()),
                    run_id=path.name,
                    sequence=sequence,
                    type="run.failed",
                    created_at=_utc_now(),
                    payload={
                        "error_code": "PROCESS_INTERRUPTED",
                        "message": "process interrupted before terminal state",
                    },
                )
                if last is not None and last.type in _TERMINAL_EVENTS:
                    self.replace_terminal_event(interrupted)
                else:
                    self.append_event(interrupted)
                manifest["status"] = "failed"
                manifest["finished_at"] = _utc_now()
                manifest["error_code"] = "PROCESS_INTERRUPTED"
                manifest["error_message"] = "process interrupted before terminal state"
                self._write_manifest(path.name, manifest)
