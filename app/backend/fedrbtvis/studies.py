from __future__ import annotations

import asyncio
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from itertools import product
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from fedrbtvis.artifacts import _atomic_write_csv, _atomic_write_json, _read_text
from fedrbtvis.config import ProbeSpec, RunConfig
from fedrbtvis.events import JsonValue
from fedrbtvis.manager import RunManager
from fedrbtvis.presets import build_preset

FactorName = Literal["target_noise", "target_emd", "sample_count", "lid_k"]
StudyStatus = Literal["queued", "running", "completed", "failed", "stopped"]

_STUDY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_OBSERVATION_HEADERS = (
    "source",
    "study_id",
    "run_id",
    "client_id",
    "seed",
    "cycle",
    "step",
    "role",
    "target_noise",
    "actual_noise",
    "target_emd",
    "actual_emd",
    "sample_count",
    "lid_k",
    "lid_mean",
    "lid_std",
    "train_loss",
    "test_loss",
    "test_accuracy",
)
_OBSERVATION_INTEGER_FIELDS = frozenset(
    {"client_id", "seed", "cycle", "step", "sample_count", "lid_k"}
)
_OBSERVATION_FLOAT_FIELDS = frozenset(
    {
        "target_noise",
        "actual_noise",
        "target_emd",
        "actual_emd",
        "lid_mean",
        "lid_std",
        "train_loss",
        "test_loss",
        "test_accuracy",
    }
)


class StudyNotFoundError(FileNotFoundError):
    pass


class StudyCorruptError(RuntimeError):
    pass


class StudyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    preset: Literal["research-lite", "historical-compatible"]
    factors: dict[FactorName, tuple[int | float, ...]]
    seeds: tuple[int, ...]

    @field_validator("factors")
    @classmethod
    def validate_factors(
        cls,
        factors: dict[FactorName, tuple[int | float, ...]],
    ) -> dict[FactorName, tuple[int | float, ...]]:
        for name, values in factors.items():
            if not values:
                raise ValueError(f"factor {name} must contain at least one value")
            for value in values:
                if isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(f"factor {name} contains a non-finite value")
                if name in {"target_noise", "target_emd"}:
                    if not 0.0 <= float(value) <= 0.9:
                        raise ValueError(f"factor {name} must be between 0 and 0.9")
                elif name == "sample_count":
                    if not isinstance(value, int) or value <= 1:
                        raise ValueError("sample_count values must be integers above one")
                elif not isinstance(value, int) or value < 2:
                    raise ValueError("lid_k values must be integers of at least two")
        return factors

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        if not seeds:
            raise ValueError("seeds must not be empty")
        if any(isinstance(seed, bool) or seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative integers")
        if len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be unique")
        return seeds

    @model_validator(mode="after")
    def validate_size(self) -> "StudyConfig":
        combinations = len(self.seeds)
        for values in self.factors.values():
            combinations *= len(values)
        if combinations > 100:
            raise ValueError("study expands beyond the 100-run limit")
        return self


@dataclass(frozen=True)
class StudyResult:
    study_id: str
    status: StudyStatus
    run_ids: tuple[str, ...]
    manifest_path: Path
    observations_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _study_dir(root: Path, study_id: str, *, must_exist: bool = True) -> Path:
    if not _STUDY_ID.fullmatch(study_id) or study_id in {".", ".."}:
        raise StudyNotFoundError("study was not found")
    path = Path(root).resolve() / study_id
    if must_exist and not path.is_dir():
        raise StudyNotFoundError("study was not found")
    return path


def expand_study(
    spec: StudyConfig,
    data_dir: Path,
    artifact_root: Path,
) -> tuple[RunConfig, ...]:
    names = tuple(sorted(spec.factors))
    value_sets = tuple(spec.factors[name] for name in names)
    combinations = product(*value_sets) if value_sets else [()]
    factor_combinations = tuple(combinations)
    runs: list[RunConfig] = []
    for seed in spec.seeds:
        for values in factor_combinations:
            factor_values = dict(zip(names, values, strict=True))
            base = build_preset(spec.preset, data_dir, artifact_root)
            probe_updates: dict[str, int | float] = {}
            for name, value in factor_values.items():
                probe_updates[name] = (
                    int(value)
                    if name in {"sample_count", "lid_k"}
                    else float(value)
                )
            probes = tuple(
                ProbeSpec.model_validate(
                    {**probe.model_dump(), **probe_updates}
                )
                for probe in base.probes
            )
            runs.append(
                RunConfig.model_validate(
                    {
                        **base.model_dump(),
                        "seed": seed,
                        "probes": probes,
                    }
                )
            )
    return tuple(runs)


def prepare_study(
    study_id: str,
    configs: tuple[RunConfig, ...],
    study_root: Path,
) -> dict[str, JsonValue]:
    if not configs:
        raise ValueError("study must contain at least one run")
    study_dir = _study_dir(study_root, study_id, must_exist=False)
    try:
        study_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("study already exists") from error
    manifest: dict[str, JsonValue] = {
        "schema_version": 1,
        "study_id": study_id,
        "status": "queued",
        "preset": configs[0].preset,
        "total_runs": len(configs),
        "run_ids": [],
        "active_run_id": None,
        "created_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "error_code": None,
        "error_message": None,
    }
    _atomic_write_csv(
        study_dir / "observations.csv",
        _OBSERVATION_HEADERS,
        (),
    )
    _atomic_write_json(study_dir / "manifest.json", manifest)
    return manifest


def read_study(study_root: Path, study_id: str) -> dict[str, JsonValue]:
    path = _study_dir(study_root, study_id) / "manifest.json"
    if not path.is_file():
        raise StudyCorruptError("study manifest is missing")
    try:
        value = json.loads(_read_text(path))
    except (OSError, UnicodeError, ValueError) as error:
        raise StudyCorruptError("study manifest is corrupt") from error
    if not isinstance(value, dict) or value.get("study_id") != study_id:
        raise StudyCorruptError("study manifest identity is invalid")
    return value


def read_observations(
    study_root: Path,
    study_id: str,
) -> list[dict[str, JsonValue]]:
    path = _study_dir(study_root, study_id) / "observations.csv"
    if not path.is_file():
        raise StudyCorruptError("study observations are missing")
    try:
        reader = csv.DictReader(StringIO(_read_text(path), newline=""))
        if tuple(reader.fieldnames or ()) != _OBSERVATION_HEADERS:
            raise StudyCorruptError("study observations schema is invalid")
        rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise StudyCorruptError("study observations are corrupt") from error

    converted: list[dict[str, JsonValue]] = []
    try:
        for row in rows:
            item: dict[str, JsonValue] = {}
            for key, value in row.items():
                if key in _OBSERVATION_INTEGER_FIELDS:
                    item[key] = int(value)
                elif key in _OBSERVATION_FLOAT_FIELDS:
                    item[key] = float(value)
                else:
                    item[key] = value
            converted.append(item)
    except (TypeError, ValueError) as error:
        raise StudyCorruptError("study observation values are corrupt") from error
    return converted


def reindex_interrupted_studies(study_root: Path) -> None:
    root = Path(study_root).resolve()
    if not root.is_dir():
        return
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        manifest = read_study(root, path.name)
        if manifest.get("status") not in {"queued", "running"}:
            continue
        manifest["status"] = "failed"
        manifest["active_run_id"] = None
        manifest["finished_at"] = _utc_now()
        manifest["error_code"] = "PROCESS_INTERRUPTED"
        manifest["error_message"] = "process interrupted before terminal state"
        _atomic_write_json(path / "manifest.json", manifest)


def _observation_rows(
    study_id: str,
    run_id: str,
    config: RunConfig,
    manager: RunManager,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in manager.client_metrics(run_id):
        rows.append(
            {
                "source": "fresh",
                "study_id": study_id,
                "run_id": run_id,
                "client_id": metric["client_id"],
                "seed": config.seed,
                "cycle": metric["cycle"],
                "step": metric["step"],
                "role": metric["role"],
                "target_noise": metric["target_noise"],
                "actual_noise": metric["actual_noise"],
                "target_emd": metric["target_emd"],
                "actual_emd": metric["actual_emd"],
                "sample_count": metric["sample_count"],
                "lid_k": metric["lid_k"],
                "lid_mean": metric["lid_mean"],
                "lid_std": metric["lid_std"],
                "train_loss": metric["train_loss"],
                "test_loss": metric["test_loss"],
                "test_accuracy": metric["test_accuracy"],
            }
        )
    return rows


async def run_study(
    study_id: str,
    configs: tuple[RunConfig, ...],
    manager: RunManager,
    study_root: Path | None = None,
) -> StudyResult:
    configs = tuple(configs)
    root = (
        Path(study_root)
        if study_root is not None
        else Path(manager.store.root).parent / "studies"
    )
    study_dir = _study_dir(root, study_id, must_exist=False)
    if not study_dir.is_dir():
        manifest = prepare_study(study_id, configs, root)
    else:
        manifest = read_study(root, study_id)
        if manifest.get("status") != "queued":
            raise ValueError("study is not queued")

    manifest["status"] = "running"
    manifest["started_at"] = _utc_now()
    _atomic_write_json(study_dir / "manifest.json", manifest)
    run_ids: list[str] = []
    observations: list[dict[str, object]] = []
    status_value: StudyStatus = "completed"

    try:
        for config in configs:
            created = await manager.create_run(config)
            run_ids.append(created.run_id)
            manifest["run_ids"] = list(run_ids)
            manifest["active_run_id"] = created.run_id
            _atomic_write_json(study_dir / "manifest.json", manifest)
            await manager.wait(created.run_id)
            run_manifest = manager.get_run(created.run_id)
            run_status = run_manifest.get("status")
            if run_status != "completed":
                status_value = "stopped" if run_status == "stopped" else "failed"
                manifest["error_code"] = (
                    "RUN_STOPPED" if run_status == "stopped" else "RUN_FAILED"
                )
                manifest["error_message"] = (
                    f"run ended with terminal status {run_status}"
                )
                break
            observations.extend(
                _observation_rows(study_id, created.run_id, config, manager)
            )
            _atomic_write_csv(
                study_dir / "observations.csv",
                _OBSERVATION_HEADERS,
                observations,
            )
    except asyncio.CancelledError:
        status_value = "failed"
        manifest["error_code"] = "PROCESS_INTERRUPTED"
        manifest["error_message"] = "process interrupted before terminal state"
        manifest["status"] = status_value
        manifest["run_ids"] = list(run_ids)
        manifest["active_run_id"] = None
        manifest["finished_at"] = _utc_now()
        _atomic_write_csv(
            study_dir / "observations.csv",
            _OBSERVATION_HEADERS,
            observations,
        )
        _atomic_write_json(study_dir / "manifest.json", manifest)
        raise
    except Exception as error:
        status_value = "failed"
        manifest["error_code"] = "STUDY_EXECUTION_FAILED"
        manifest["error_message"] = f"{error.__class__.__name__}: study execution failed"

    manifest["status"] = status_value
    manifest["run_ids"] = list(run_ids)
    manifest["active_run_id"] = None
    manifest["finished_at"] = _utc_now()
    _atomic_write_csv(
        study_dir / "observations.csv",
        _OBSERVATION_HEADERS,
        observations,
    )
    _atomic_write_json(study_dir / "manifest.json", manifest)
    return StudyResult(
        study_id=study_id,
        status=status_value,
        run_ids=tuple(run_ids),
        manifest_path=study_dir / "manifest.json",
        observations_path=study_dir / "observations.csv",
    )
