from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Literal

from fedrbtvis.artifacts import ArtifactCorruptError, sha256_file


class LegacyHashError(ValueError):
    pass


class LegacyFormatError(ValueError):
    pass


ROW_RE = re.compile(
    r"^i=(?P<client>\d+) (?P<noise>\d+(?:\.\d+)?) (?P<emd>\d+(?:\.\d+)?) "
    r"(?P<lid>\d+(?:\.\d+)?) (?P<size>\d+)$"
)
BLOCK_RE = re.compile(r"^-+k=(?P<k>\d+) ROUND (?P<round>\d+)-+")
TOTAL_RE = re.compile(
    r"^k=\d+ ROUND=\d+ --- total time cost: .* (?P<seconds>\d+(?:\.\d+)?)s$"
)
BEGIN_RE = re.compile(r"^----------begin----------$")
STAGE_RE = re.compile(r"^.*stage time cost:.*$")
CSV_HEADERS = (
    "source",
    "block_index",
    "k_index",
    "round_index",
    "client_id",
    "actual_noise",
    "categorical_emd_01",
    "lid",
    "sample_count",
    "inferred_target_noise",
    "inferred_emd_base",
)
_INTEGER_FIELDS = frozenset(
    {"block_index", "k_index", "round_index", "client_id", "sample_count"}
)
_FLOAT_FIELDS = frozenset(
    {
        "actual_noise",
        "categorical_emd_01",
        "lid",
        "inferred_target_noise",
        "inferred_emd_base",
    }
)
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "source_basename",
        "source_sha256",
        "importer_version",
        "rows",
        "blocks",
        "clients_per_block",
        "client_min",
        "client_max",
        "actual_noise_min",
        "actual_noise_max",
        "categorical_emd_01_min",
        "categorical_emd_01_max",
        "lid_min",
        "lid_max",
        "sample_count_min",
        "sample_count_max",
        "inferred_target_noise_min",
        "inferred_target_noise_max",
        "inferred_emd_base_min",
        "inferred_emd_base_max",
        "total_runtime_seconds",
        "csv_sha256",
        "field_semantics",
        "inferred_formulas",
    }
)


@dataclass(frozen=True)
class LegacyObservation:
    source: Literal["legacy"]
    block_index: int
    k_index: int
    round_index: int
    client_id: int
    actual_noise: float
    categorical_emd_01: float
    lid: float
    sample_count: int
    inferred_target_noise: float
    inferred_emd_base: float


@dataclass(frozen=True)
class LegacyImportResult:
    source_sha256: str
    observations: tuple[LegacyObservation, ...]
    blocks: int
    clients_per_block: int
    total_runtime_seconds: float


def _accept_block(
    pending: list[tuple[int, float, float, float, int]],
    expected_clients: tuple[int, ...],
    k_index: int,
    round_index: int,
    block_index: int,
    observations: list[LegacyObservation],
) -> None:
    if len(pending) != len(expected_clients) or sorted(
        client for client, _, _, _, _ in pending
    ) != sorted(expected_clients):
        raise LegacyFormatError("block client set does not match expected clients")

    for client_id, noise, emd, lid, sample_count in pending:
        if not all(
            math.isfinite(value)
            for value in (noise, emd, lid)
        ):
            raise LegacyFormatError("non-finite numeric value in data row")
        if not 0.0 <= noise <= 1.0:
            raise LegacyFormatError("actual noise must be in [0, 1]")
        if not 0.0 <= emd <= 1.0:
            raise LegacyFormatError("categorical distance must be in [0, 1]")
        if lid <= 0.0:
            raise LegacyFormatError("LID must be positive")
        if sample_count <= 1:
            raise LegacyFormatError("sample count must be greater than one")
        observations.append(
            LegacyObservation(
                source="legacy",
                block_index=block_index,
                k_index=k_index,
                round_index=round_index,
                client_id=client_id,
                actual_noise=noise,
                categorical_emd_01=emd,
                lid=lid,
                sample_count=sample_count,
                inferred_target_noise=k_index * 0.05,
                inferred_emd_base=round_index * 0.05,
            )
        )


def parse_predict_file(
    path: Path,
    expected_sha256: str,
    expected_clients: Sequence[int],
) -> LegacyImportResult:
    path = Path(path)
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise LegacyHashError(
            f"source hash mismatch: expected {expected_sha256.lower()}, "
            f"got {actual_sha256}"
        )

    clients = tuple(int(value) for value in expected_clients)
    observations: list[LegacyObservation] = []
    pending: list[tuple[int, float, float, float, int]] = []
    block_index = 0
    total_runtime_seconds = 0.0

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise LegacyFormatError("source is not valid UTF-8") from error

    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue

        row = ROW_RE.fullmatch(line)
        if row is not None:
            pending.append(
                (
                    int(row.group("client")),
                    float(row.group("noise")),
                    float(row.group("emd")),
                    float(row.group("lid")),
                    int(row.group("size")),
                )
            )
            continue

        block = BLOCK_RE.match(line)
        if block is not None:
            if not pending:
                raise LegacyFormatError(
                    f"block header at line {line_number} has no data rows"
                )
            k_index = int(block.group("k"))
            round_index = int(block.group("round"))
            _accept_block(
                pending,
                clients,
                k_index,
                round_index,
                block_index,
                observations,
            )
            block_index += 1
            pending = []
            continue

        total = TOTAL_RE.match(line)
        if total is not None:
            total_runtime_seconds = float(total.group("seconds"))
            continue

        if BEGIN_RE.fullmatch(line) is not None or STAGE_RE.fullmatch(line) is not None:
            continue

        raise LegacyFormatError(f"unexpected line {line_number}: {line}")

    if pending:
        raise LegacyFormatError("file ended with an incomplete block")

    return LegacyImportResult(
        source_sha256=actual_sha256,
        observations=tuple(observations),
        blocks=block_index,
        clients_per_block=len(clients),
        total_runtime_seconds=total_runtime_seconds,
    )


class LegacyRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._manifest = self._load_manifest()
        self._observations = self._load_observations()

    @classmethod
    def from_directory(cls, root: Path) -> "LegacyRepository":
        return cls(root)

    def _load_manifest(self) -> dict[str, object]:
        path = self.root / "manifest.json"
        if not path.is_file():
            raise ArtifactCorruptError("legacy manifest.json is missing")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ArtifactCorruptError(
                "legacy manifest.json is invalid"
            ) from error
        if not isinstance(value, dict):
            raise ArtifactCorruptError("legacy manifest.json must be an object")
        missing = _REQUIRED_MANIFEST_FIELDS.difference(value)
        if missing:
            raise ArtifactCorruptError(
                "legacy manifest is missing fields: "
                + ", ".join(sorted(missing))
            )
        if value.get("schema_version") != 1:
            raise ArtifactCorruptError("legacy manifest schema version is invalid")
        for key in ("rows", "blocks", "clients_per_block"):
            if type(value.get(key)) is not int:
                raise ArtifactCorruptError(f"legacy manifest {key} is invalid")
        return value

    def _load_observations(self) -> tuple[dict[str, object], ...]:
        path = self.root / "observations.csv"
        if not path.is_file():
            raise ArtifactCorruptError("legacy observations.csv is missing")
        actual_hash = sha256_file(path)
        expected_hash = str(self._manifest.get("csv_sha256", "")).lower()
        if actual_hash.lower() != expected_hash:
            raise ArtifactCorruptError("legacy observations.csv hash mismatch")
        try:
            reader = csv.DictReader(
                StringIO(path.read_text(encoding="utf-8"), newline="")
            )
            if tuple(reader.fieldnames or ()) != CSV_HEADERS:
                raise ArtifactCorruptError("legacy observations.csv schema is invalid")
            raw_rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as error:
            raise ArtifactCorruptError(
                "legacy observations.csv is invalid"
            ) from error

        if len(raw_rows) != 4500:
            raise ArtifactCorruptError(
                f"legacy observations.csv rows={len(raw_rows)}, expected 4500"
            )

        observations: list[dict[str, object]] = []
        for raw in raw_rows:
            item: dict[str, object] = {}
            for key, value in raw.items():
                if key in _INTEGER_FIELDS:
                    item[key] = int(value)
                elif key in _FLOAT_FIELDS:
                    item[key] = float(value)
                else:
                    item[key] = value
            observations.append(item)

        for row in observations:
            if row.get("source") != "legacy":
                raise ArtifactCorruptError("legacy observation source is invalid")
            if not all(
                math.isfinite(float(row[field]))
                for field in _FLOAT_FIELDS
                if field in row
            ):
                raise ArtifactCorruptError("legacy observation has non-finite value")
            if not 0.0 <= float(row["actual_noise"]) <= 1.0:
                raise ArtifactCorruptError("legacy actual_noise is out of range")
            if not 0.0 <= float(row["categorical_emd_01"]) <= 1.0:
                raise ArtifactCorruptError("legacy categorical_emd_01 is out of range")
            if float(row["lid"]) <= 0.0:
                raise ArtifactCorruptError("legacy LID is not positive")
            if int(row["sample_count"]) != 200:
                raise ArtifactCorruptError("legacy sample_count is not 200")

        by_block: dict[int, list[int]] = {}
        for row in observations:
            by_block.setdefault(int(row["block_index"]), []).append(
                int(row["client_id"])
            )
        for block_index in range(180):
            if sorted(by_block.get(block_index, [])) != list(range(100, 125)):
                raise ArtifactCorruptError(
                    "legacy client ids are not 100-124 per block"
                )
        coords = {
            (int(row["k_index"]), int(row["round_index"]))
            for row in observations
        }
        expected_coords = {
            (k_index, round_index)
            for k_index in range(10)
            for round_index in range(18)
        }
        if coords != expected_coords:
            raise ArtifactCorruptError("legacy block coordinates are invalid")

        return tuple(observations)

    def manifest_summary(self) -> dict[str, object]:
        return dict(self._manifest)

    def observations(self) -> tuple[dict[str, object], ...]:
        return self._observations
