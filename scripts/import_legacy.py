from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path

from fedrbtvis.legacy import (
    LegacyFormatError,
    LegacyHashError,
    LegacyImportResult,
    parse_predict_file,
)


EXPECTED_CLIENTS = tuple(range(100, 125))
EXPECTED_RUNTIME = 52257.863918
EXPECTED_ROWS = 4500
EXPECTED_BLOCKS = 180
EXPECTED_CLIENTS_PER_BLOCK = 25
EXPECTED_COORDS = {
    (k_index, round_index)
    for k_index in range(10)
    for round_index in range(18)
}


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


def _validate(result: LegacyImportResult) -> None:
    errors: list[str] = []
    if len(result.observations) != EXPECTED_ROWS:
        errors.append(f"rows={len(result.observations)}, expected {EXPECTED_ROWS}")
    if result.blocks != EXPECTED_BLOCKS:
        errors.append(f"blocks={result.blocks}, expected {EXPECTED_BLOCKS}")
    if result.clients_per_block != EXPECTED_CLIENTS_PER_BLOCK:
        errors.append(
            f"clients_per_block={result.clients_per_block}, "
            f"expected {EXPECTED_CLIENTS_PER_BLOCK}"
        )
    coords = {(item.k_index, item.round_index) for item in result.observations}
    if coords != EXPECTED_COORDS:
        errors.append("block coordinates are not exactly k 0..9 x round 0..17")
    if abs(result.total_runtime_seconds - EXPECTED_RUNTIME) > 1e-6:
        errors.append(
            f"runtime={result.total_runtime_seconds}, expected {EXPECTED_RUNTIME}"
        )
    if errors:
        raise LegacyFormatError("; ".join(errors))


def _csv_text(result: LegacyImportResult) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(CSV_HEADERS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for item in result.observations:
        writer.writerow(
            {
                "source": item.source,
                "block_index": item.block_index,
                "k_index": item.k_index,
                "round_index": item.round_index,
                "client_id": item.client_id,
                "actual_noise": item.actual_noise,
                "categorical_emd_01": item.categorical_emd_01,
                "lid": item.lid,
                "sample_count": item.sample_count,
                "inferred_target_noise": item.inferred_target_noise,
                "inferred_emd_base": item.inferred_emd_base,
            }
        )
    return stream.getvalue()


def _min_max(values: list[float]) -> tuple[float, float]:
    return min(values), max(values)


def _manifest_text(result: LegacyImportResult, csv_sha256: str) -> str:
    noise = [item.actual_noise for item in result.observations]
    emd = [item.categorical_emd_01 for item in result.observations]
    lid = [item.lid for item in result.observations]
    size = [item.sample_count for item in result.observations]
    inferred_noise = [item.inferred_target_noise for item in result.observations]
    inferred_emd = [item.inferred_emd_base for item in result.observations]
    manifest = {
        "schema_version": 1,
        "source_basename": "predict.txt",
        "source_sha256": result.source_sha256,
        "importer_version": 1,
        "rows": len(result.observations),
        "blocks": result.blocks,
        "clients_per_block": result.clients_per_block,
        "client_min": min(item.client_id for item in result.observations),
        "client_max": max(item.client_id for item in result.observations),
        "actual_noise_min": _min_max(noise)[0],
        "actual_noise_max": _min_max(noise)[1],
        "categorical_emd_01_min": _min_max(emd)[0],
        "categorical_emd_01_max": _min_max(emd)[1],
        "lid_min": _min_max(lid)[0],
        "lid_max": _min_max(lid)[1],
        "sample_count_min": min(size),
        "sample_count_max": max(size),
        "inferred_target_noise_min": min(inferred_noise),
        "inferred_target_noise_max": max(inferred_noise),
        "inferred_emd_base_min": min(inferred_emd),
        "inferred_emd_base_max": max(inferred_emd),
        "total_runtime_seconds": result.total_runtime_seconds,
        "csv_sha256": csv_sha256,
        "field_semantics": {
            "source": "legacy historical observation",
            "block_index": "zero-based block ordinal",
            "k_index": "LID neighbor count index from historical loop",
            "round_index": "historical round index",
            "client_id": "probe client id 100-124",
            "actual_noise": "measured label noise from the log",
            "categorical_emd_01": "0/1 category-cost distance, not pixel EMD",
            "lid": "local intrinsic dimensionality from the log",
            "sample_count": "per-client sample count from the log",
            "inferred_target_noise": "derived target noise from the historical loop",
            "inferred_emd_base": "derived target EMD base from the historical loop",
        },
        "inferred_formulas": {
            "inferred_target_noise": "k_index * 0.05",
            "inferred_emd_base": "round_index * 0.05",
        },
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def _readme_text() -> str:
    return (
        "# Legacy LID Observations\n\n"
        "This directory contains the hash-anchored historical 4,500-row "
        "observation set imported from the course-stage `predict.txt`. "
        "These are course-era historical observations, not 2026 experiments. "
        "The source archive and CIFAR datasets are not included.\n\n"
        "`categorical_emd_01` is a 0/1 category-cost distance, not pixel-space "
        "EMD. `inferred_target_noise` and `inferred_emd_base` are derived from "
        "the historical loop and must never be described as measured values.\n\n"
        "Attribution remains three-layered: the 2023-2024 three-person course "
        "project, the user's personal graduation project extension, and the "
        "2026 owned rebuild. See `docs/provenance.md` for the full source "
        "boundary.\n"
    )


def _atomic_write_assets(
    output_dir: Path,
    csv_text: str,
    manifest_text: str,
    readme_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        temporary_dir = Path(temporary)
        assets = {
            "observations.csv": csv_text,
            "manifest.json": manifest_text,
            "README.md": readme_text,
        }
        for name, content in assets.items():
            (temporary_dir / name).write_bytes(content.encode("utf-8"))
        for name in assets:
            os.replace(temporary_dir / name, output_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import the hash-gated historical predict.txt evidence set"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    try:
        result = parse_predict_file(
            args.source,
            expected_sha256=args.expected_sha256,
            expected_clients=EXPECTED_CLIENTS,
        )
        _validate(result)
    except (LegacyHashError, LegacyFormatError) as error:
        print(f"import rejected: {error}", file=sys.stderr)
        return 1

    csv_text = _csv_text(result)
    csv_sha256 = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    manifest_text = _manifest_text(result, csv_sha256)
    readme_text = _readme_text()
    _atomic_write_assets(
        Path(args.output_dir).resolve(),
        csv_text,
        manifest_text,
        readme_text,
    )
    print(f"imported {len(result.observations)} observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
