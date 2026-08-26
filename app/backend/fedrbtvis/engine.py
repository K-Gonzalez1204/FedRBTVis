from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedrbtvis.config import RunConfig
from fedrbtvis.data import (
    ClientPartition,
    DatasetBundle,
    build_client_partitions,
)
from fedrbtvis.events import EventEmitter, TrainingEvent
from fedrbtvis.models import build_model
from fedrbtvis.training import (
    ClientUpdate,
    evaluate,
    train_local,
    weighted_fedavg,
)


@dataclass(frozen=True)
class AggregationRow:
    cycle: int
    step: int
    client_ids: tuple[int, ...]
    test_loss: float
    test_accuracy: float


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    status: Literal["completed", "stopped"]
    schedule: tuple[tuple[int, ...], ...]
    client_updates: tuple[ClientUpdate, ...]
    aggregations: tuple[AggregationRow, ...]
    final_server_state: dict[str, torch.Tensor]


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _build_schedule(
    config: RunConfig,
    partitions: tuple[ClientPartition, ...],
) -> tuple[tuple[int, ...], ...]:
    rng = np.random.default_rng(config.seed)
    client_ids = np.array(
        [partition.client_id for partition in partitions],
        dtype=np.int64,
    )
    groups: list[tuple[int, ...]] = []
    for _ in range(config.cycles):
        order = client_ids.copy()
        rng.shuffle(order)
        for start in range(0, len(order), config.clients_per_step):
            groups.append(tuple(int(value) for value in order[start : start + config.clients_per_step]))
    return tuple(groups)


def _validate_bundle(bundle: DatasetBundle, num_classes: int) -> None:
    splits = (
        ("train", bundle.train_images, np.asarray(bundle.train_labels)),
        ("test", bundle.test_images, np.asarray(bundle.test_labels)),
    )
    for name, images, labels in splits:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(f"{name} images must be a four-dimensional tensor")
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{name} labels must be a one-dimensional integer array")
        if len(images) != len(labels):
            raise ValueError(f"{name} images and labels must align")
        if len(labels) == 0:
            raise ValueError(f"{name} split must not be empty")
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError(f"{name} labels fall outside the configured class range")
        if images.shape[1] != 3:
            raise ValueError(f"{name} images must have three channels")
    if bundle.train_images.shape[1:] != bundle.test_images.shape[1:]:
        raise ValueError("train and test image shapes must match")


def run_experiment(
    run_id: str,
    config: RunConfig,
    bundle: DatasetBundle,
    device: torch.device,
    on_event: Callable[[TrainingEvent], None],
    stop_requested: Callable[[], bool],
) -> ExperimentResult:
    """Execute one immutable experiment and synchronously emit accepted events."""
    if not run_id:
        raise ValueError("run_id must not be empty")

    _validate_bundle(bundle, config.num_classes)
    partitions = build_client_partitions(bundle.train_labels, config)
    partition_by_id = {partition.client_id: partition for partition in partitions}
    schedule = _build_schedule(config, partitions)
    emitter = EventEmitter(run_id, on_event)
    emitter.emit(
        "run.started",
        {
            "preset": config.preset,
            "source": config.source,
            "cycles": config.cycles,
            "client_count": len(partitions),
            "device": str(device),
        },
    )

    test_loader = DataLoader(
        TensorDataset(
            bundle.test_images,
            torch.from_numpy(np.asarray(bundle.test_labels, dtype=np.int64)),
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    server_model = build_model(config.model, config.num_classes, config.seed)
    server_state = _clone_state(server_model.state_dict())
    client_updates: list[ClientUpdate] = []
    aggregations: list[AggregationRow] = []

    group_index = 0
    for cycle in range(1, config.cycles + 1):
        steps_in_cycle = (
            len(partitions) + config.clients_per_step - 1
        ) // config.clients_per_step
        for step in range(1, steps_in_cycle + 1):
            client_ids = schedule[group_index]
            group_index += 1
            group_updates: list[ClientUpdate] = []
            for client_id in client_ids:
                if stop_requested():
                    emitter.emit(
                        "run.stop_requested",
                        {
                            "cycle": cycle,
                            "step": step,
                            "completed_clients": len(client_updates),
                        },
                    )
                    emitter.emit(
                        "run.stopped",
                        {
                            "cycle": cycle,
                            "step": step,
                            "completed_clients": len(client_updates),
                            "completed_aggregations": len(aggregations),
                        },
                    )
                    return ExperimentResult(
                        run_id=run_id,
                        status="stopped",
                        schedule=schedule,
                        client_updates=tuple(client_updates),
                        aggregations=tuple(aggregations),
                        final_server_state=_clone_state(server_state),
                    )

                client = partition_by_id[client_id]
                emitter.emit(
                    "client.started",
                    {
                        "cycle": cycle,
                        "step": step,
                        "client_id": client.client_id,
                        "role": client.role,
                    },
                )
                client_seed = int(
                    np.random.SeedSequence(
                        (config.seed, client.client_id)
                    ).generate_state(1)[0]
                )
                update = train_local(
                    client=client,
                    server_state=server_state,
                    model_name=config.model,
                    num_classes=config.num_classes,
                    train_images=bundle.train_images,
                    clean_labels=bundle.train_labels,
                    test_loader=test_loader,
                    local_epochs=config.local_epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    seed=client_seed,
                    device=device,
                )
                group_updates.append(update)
                client_updates.append(update)
                emitter.emit(
                    "client.completed",
                    {
                        "cycle": cycle,
                        "step": step,
                        "client_id": client.client_id,
                        "role": client.role,
                        "sample_count": client.sample_count,
                        "target_noise": client.target_noise,
                        "actual_noise": update.actual_noise,
                        "target_categorical_emd_01": client.target_emd,
                        "actual_categorical_emd_01": update.actual_emd,
                        "lid_k": client.lid_k,
                        "train_loss": update.train_loss,
                        "test_loss": update.test.loss,
                        "test_accuracy": update.test.accuracy,
                        "lid_mean": update.lid_mean,
                        "lid_std": update.lid_std,
                        "state_sha256": update.state_sha256,
                    },
                )

            server_state = weighted_fedavg(
                [
                    (update.state_dict, update.sample_count)
                    for update in group_updates
                ]
            )
            aggregate_model = build_model(
                config.model,
                config.num_classes,
                config.seed,
            )
            aggregate_model.load_state_dict(server_state, strict=True)
            aggregate_model.to(device)
            test_metrics = evaluate(aggregate_model, test_loader, device)
            row = AggregationRow(
                cycle=cycle,
                step=step,
                client_ids=client_ids,
                test_loss=test_metrics.loss,
                test_accuracy=test_metrics.accuracy,
            )
            aggregations.append(row)
            emitter.emit(
                "aggregation.completed",
                {
                    "cycle": cycle,
                    "step": step,
                    "client_ids": list(client_ids),
                    "test_loss": row.test_loss,
                    "test_accuracy": row.test_accuracy,
                },
            )

    emitter.emit(
        "run.completed",
        {
            "completed_clients": len(client_updates),
            "completed_aggregations": len(aggregations),
        },
    )
    return ExperimentResult(
        run_id=run_id,
        status="completed",
        schedule=schedule,
        client_updates=tuple(client_updates),
        aggregations=tuple(aggregations),
        final_server_state=_clone_state(server_state),
    )
