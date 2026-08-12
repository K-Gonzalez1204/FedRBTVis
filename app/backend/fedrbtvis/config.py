from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DatasetName = Literal["synthetic-cifar", "cifar10"]
ModelName = Literal["tiny-cnn", "cifar-resnet18"]
CheckpointPolicy = Literal["none", "server-only", "probe-clients"]
SourceKind = Literal["fixture", "fresh", "legacy"]


class ProbeSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    client_id: int = Field(ge=0)
    target_noise: float = Field(ge=0.0, le=0.9)
    target_emd: float = Field(ge=0.0, le=0.9)
    sample_count: int = Field(gt=1)
    lid_k: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_k(self) -> "ProbeSpec":
        if self.lid_k >= self.sample_count:
            raise ValueError("lid_k must be smaller than sample_count")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, frozen=True)
    preset: Literal["test-fixture", "research-lite", "historical-compatible"]
    source: SourceKind
    dataset: DatasetName
    model: ModelName
    seed: int = Field(ge=0)
    num_classes: int = Field(default=10, ge=2)
    background_clients: int = Field(gt=0)
    background_samples: int = Field(gt=1)
    background_lid_k: int = Field(ge=2)
    background_noise_fraction: float = Field(ge=0.0, le=1.0)
    background_noise_min: float = Field(ge=0.0, le=0.9)
    background_noise_max: float = Field(ge=0.0, le=0.9)
    probes: tuple[ProbeSpec, ...]
    local_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    cycles: int = Field(gt=0)
    clients_per_step: int = Field(gt=0)
    checkpoint_policy: CheckpointPolicy
    data_dir: Path
    artifact_root: Path

    @model_validator(mode="after")
    def validate_run(self) -> "RunConfig":
        ids = list(range(self.background_clients)) + [
            probe.client_id for probe in self.probes
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("client ids must be unique")
        if self.background_noise_min > self.background_noise_max:
            raise ValueError("background noise bounds are reversed")
        if self.background_lid_k >= self.background_samples:
            raise ValueError("background_lid_k must be smaller than background_samples")

        total_clients = self.background_clients + len(self.probes)
        if self.clients_per_step > total_clients:
            raise ValueError("clients_per_step exceeds cohort size")

        expected = (
            ("synthetic-cifar", "tiny-cnn")
            if self.preset == "test-fixture"
            else ("cifar10", "cifar-resnet18")
        )
        if (self.dataset, self.model) != expected:
            raise ValueError("preset dataset/model combination is invalid")

        expected_source = "fixture" if self.preset == "test-fixture" else "fresh"
        if self.source != expected_source:
            raise ValueError("run source does not match preset")
        return self
