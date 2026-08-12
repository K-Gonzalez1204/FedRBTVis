from __future__ import annotations

import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI

from fedrbtvis.api import create_app
from fedrbtvis.artifacts import ArtifactCorruptError, ArtifactStore
from fedrbtvis.config import RunConfig
from fedrbtvis.data import build_synthetic_bundle, load_cifar10
from fedrbtvis.engine import run_experiment
from fedrbtvis.legacy import LegacyRepository
from fedrbtvis.manager import RunManager


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _environment_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _device(setting: str) -> torch.device:
    normalized = setting.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("FEDRBTVIS_DEVICE requests unavailable CUDA")
        return torch.device("cuda")
    raise ValueError("FEDRBTVIS_DEVICE must be auto, cpu, or cuda")


def build_application() -> FastAPI:
    root = _project_root()
    data_dir = _environment_path("FEDRBTVIS_DATA_DIR", root / "data")
    run_dir = _environment_path("FEDRBTVIS_RUN_DIR", root / "runs")
    legacy_dir = _environment_path(
        "FEDRBTVIS_LEGACY_DIR",
        root / "evidence" / "legacy",
    )
    selected_device = _device(os.environ.get("FEDRBTVIS_DEVICE", "auto"))

    def bundle_loader(config: RunConfig):
        if config.dataset == "synthetic-cifar":
            return build_synthetic_bundle(
                seed=config.seed,
                num_classes=config.num_classes,
                train_size=300,
                test_size=100,
            )
        return load_cifar10(config.data_dir, download=False)

    manager = RunManager(
        store=ArtifactStore(run_dir),
        bundle_loader=bundle_loader,
        runner=run_experiment,
        device_resolver=lambda config: selected_device,
    )
    legacy_repository = None
    legacy_error = None
    if legacy_dir.is_dir():
        try:
            legacy_repository = LegacyRepository.from_directory(legacy_dir)
        except ArtifactCorruptError:
            legacy_error = "ARTIFACT_CORRUPT"

    application = create_app(
        manager,
        legacy_repository=legacy_repository,
        legacy_error=legacy_error,
    )
    application.state.data_dir = data_dir
    return application


app = build_application()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
