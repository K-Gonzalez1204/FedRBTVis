from pathlib import Path
from typing import Literal

from fedrbtvis.config import ProbeSpec, RunConfig

PresetName = Literal["test-fixture", "research-lite", "historical-compatible"]


def _probe(
    client_id: int,
    target_noise: float,
    target_emd: float,
    sample_count: int,
    lid_k: int,
) -> ProbeSpec:
    return ProbeSpec(
        client_id=client_id,
        target_noise=target_noise,
        target_emd=target_emd,
        sample_count=sample_count,
        lid_k=lid_k,
    )


def build_preset(name: PresetName, data_dir: Path, artifact_root: Path) -> RunConfig:
    """Build a new immutable configuration for a supported experiment preset."""
    if name == "test-fixture":
        probes = (
            _probe(2, 0.0, 0.0, 30, 5),
            _probe(3, 0.25, 0.4, 30, 5),
        )
        return RunConfig(
            preset=name,
            source="fixture",
            dataset="synthetic-cifar",
            model="tiny-cnn",
            seed=7,
            background_clients=2,
            background_samples=30,
            background_lid_k=5,
            background_noise_fraction=0.0,
            background_noise_min=0.0,
            background_noise_max=0.0,
            probes=probes,
            local_epochs=1,
            batch_size=10,
            learning_rate=0.05,
            cycles=1,
            clients_per_step=2,
            checkpoint_policy="none",
            data_dir=data_dir,
            artifact_root=artifact_root,
        )

    if name == "research-lite":
        probes = tuple(
            _probe(client_id, 0.2, target_emd, 200, 20)
            for client_id, target_emd in zip(
                range(10, 15),
                (0.0, 0.2, 0.4, 0.6, 0.8),
                strict=True,
            )
        )
        return RunConfig(
            preset=name,
            source="fresh",
            dataset="cifar10",
            model="cifar-resnet18",
            seed=7,
            background_clients=10,
            background_samples=200,
            background_lid_k=20,
            background_noise_fraction=0.2,
            background_noise_min=0.0,
            background_noise_max=0.5,
            probes=probes,
            local_epochs=1,
            batch_size=32,
            learning_rate=0.01,
            cycles=1,
            clients_per_step=5,
            checkpoint_policy="server-only",
            data_dir=data_dir,
            artifact_root=artifact_root,
        )

    if name == "historical-compatible":
        emd_values = (0.0, 0.2, 0.4, 0.6, 0.8)
        noise_values = (0.0, 0.1, 0.2, 0.3, 0.4)
        probes = tuple(
            _probe(
                client_id=100 + index,
                target_noise=noise_values[index // 5],
                target_emd=emd_values[index % 5],
                sample_count=200,
                lid_k=20,
            )
            for index in range(25)
        )
        return RunConfig(
            preset=name,
            source="fresh",
            dataset="cifar10",
            model="cifar-resnet18",
            seed=7,
            background_clients=100,
            background_samples=200,
            background_lid_k=20,
            background_noise_fraction=0.2,
            background_noise_min=0.0,
            background_noise_max=0.5,
            probes=probes,
            local_epochs=5,
            batch_size=50,
            learning_rate=0.01,
            cycles=1,
            clients_per_step=1,
            checkpoint_policy="server-only",
            data_dir=data_dir,
            artifact_root=artifact_root,
        )

    raise ValueError(f"unknown preset: {name}")
