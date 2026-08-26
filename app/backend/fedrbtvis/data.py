from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from fedrbtvis.config import RunConfig
from fedrbtvis.metrics import categorical_emd_01


@dataclass(frozen=True)
class DatasetBundle:
    train_images: torch.Tensor
    train_labels: np.ndarray
    test_images: torch.Tensor
    test_labels: np.ndarray


@dataclass(frozen=True)
class ClientPartition:
    client_id: int
    role: Literal["background", "probe"]
    indices: tuple[int, ...]
    clean_histogram: tuple[int, ...]
    target_emd: float
    actual_emd: float
    target_noise: float
    sample_count: int
    lid_k: int


class PartitionError(ValueError):
    pass


def _synthetic_split(
    seed: int,
    num_classes: int,
    size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    labels = np.arange(size, dtype=np.int64) % num_classes
    numpy_rng = np.random.default_rng(seed)
    numpy_rng.shuffle(labels)

    torch_rng = torch.Generator().manual_seed(seed)
    images = torch.randn(size, 3, 32, 32, generator=torch_rng) * 0.05
    for index, label in enumerate(labels):
        channel = int(label % 3)
        row = int(label // 3) * 8
        column = int(label % 3) * 8
        images[index, channel, row : row + 8, column : column + 8] += 1.5
    return images.to(dtype=torch.float32), labels


def build_synthetic_bundle(
    seed: int,
    num_classes: int,
    train_size: int,
    test_size: int,
) -> DatasetBundle:
    """Build deterministic CIFAR-shaped tensors with class-dependent signal."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if train_size < num_classes or test_size < num_classes:
        raise ValueError("each split must contain at least one sample per class")

    train_images, train_labels = _synthetic_split(seed, num_classes, train_size)
    test_images, test_labels = _synthetic_split(seed + 1, num_classes, test_size)
    return DatasetBundle(train_images, train_labels, test_images, test_labels)


def _normalized_cifar_tensor(images: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(images)).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
    return (tensor - mean) / std


def load_cifar10(data_dir: Path, download: bool) -> DatasetBundle:
    """Load normalized CIFAR-10 tensors without downloading unless permitted."""
    data_dir = Path(data_dir)
    cache = data_dir / "cifar-10-batches-py"
    if not download and not cache.is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 cache is missing at {cache}; enable download explicitly"
        )

    from torchvision.datasets import CIFAR10

    train = CIFAR10(root=str(data_dir), train=True, download=download)
    test = CIFAR10(root=str(data_dir), train=False, download=download)
    return DatasetBundle(
        train_images=_normalized_cifar_tensor(train.data),
        train_labels=np.asarray(train.targets, dtype=np.int64),
        test_images=_normalized_cifar_tensor(test.data),
        test_labels=np.asarray(test.targets, dtype=np.int64),
    )


def _balanced_histogram(sample_count: int, num_classes: int) -> list[int]:
    quotient, remainder = divmod(sample_count, num_classes)
    return [quotient + (class_id < remainder) for class_id in range(num_classes)]


def _probe_histogram(
    sample_count: int,
    num_classes: int,
    target_emd: float,
) -> list[int]:
    histogram = _balanced_histogram(sample_count, num_classes)
    transfers = round(target_emd * sample_count)
    available = sum(histogram[1:])
    if transfers > available:
        raise PartitionError("target EMD exceeds the categorical allocation capacity")

    donor = 1
    for _ in range(transfers):
        searched = 0
        while histogram[donor] == 0 and searched < num_classes - 1:
            donor = 1 + donor % (num_classes - 1)
            searched += 1
        if histogram[donor] == 0:
            raise PartitionError("probe histogram has no remaining donor capacity")
        histogram[donor] -= 1
        histogram[0] += 1
        donor = 1 + donor % (num_classes - 1)
    return histogram


def build_client_partitions(
    labels: np.ndarray,
    config: RunConfig,
) -> tuple[ClientPartition, ...]:
    """Allocate deterministic, disjoint background and probe sample indices."""
    values = np.asarray(labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise PartitionError("labels must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values >= config.num_classes):
        raise PartitionError("labels fall outside the configured class range")
    if not config.probes:
        raise PartitionError("at least one probe client is required")

    rng = np.random.default_rng(config.seed)
    pools: list[np.ndarray] = []
    cursors = [0] * config.num_classes
    for class_id in range(config.num_classes):
        pool = np.flatnonzero(values == class_id).astype(np.int64)
        rng.shuffle(pool)
        pools.append(pool)

    def consume(histogram: list[int]) -> tuple[int, ...]:
        selected: list[int] = []
        for class_id, requested in enumerate(histogram):
            start = cursors[class_id]
            end = start + requested
            if end > len(pools[class_id]):
                raise PartitionError(
                    f"class {class_id} needs {requested} more samples but lacks capacity"
                )
            selected.extend(int(index) for index in pools[class_id][start:end])
            cursors[class_id] = end
        rng.shuffle(selected)
        return tuple(selected)

    background_noises = [0.0] * config.background_clients
    noisy_background_count = round(
        config.background_clients * config.background_noise_fraction
    )
    if noisy_background_count:
        selected_backgrounds = rng.choice(
            config.background_clients,
            size=noisy_background_count,
            replace=False,
        )
        for client_id in selected_backgrounds:
            background_noises[int(client_id)] = float(
                rng.uniform(config.background_noise_min, config.background_noise_max)
            )

    partitions: list[ClientPartition] = []
    for client_id in range(config.background_clients):
        histogram = _balanced_histogram(
            config.background_samples,
            config.num_classes,
        )
        indices = consume(histogram)
        actual_histogram = np.bincount(
            values[list(indices)],
            minlength=config.num_classes,
        )
        partitions.append(
            ClientPartition(
                client_id=client_id,
                role="background",
                indices=indices,
                clean_histogram=tuple(int(value) for value in actual_histogram),
                target_emd=0.0,
                actual_emd=categorical_emd_01(actual_histogram),
                target_noise=background_noises[client_id],
                sample_count=config.background_samples,
                lid_k=config.background_lid_k,
            )
        )

    for probe in config.probes:
        histogram = _probe_histogram(
            probe.sample_count,
            config.num_classes,
            probe.target_emd,
        )
        indices = consume(histogram)
        actual_histogram = np.bincount(
            values[list(indices)],
            minlength=config.num_classes,
        )
        partitions.append(
            ClientPartition(
                client_id=probe.client_id,
                role="probe",
                indices=indices,
                clean_histogram=tuple(int(value) for value in actual_histogram),
                target_emd=probe.target_emd,
                actual_emd=categorical_emd_01(actual_histogram),
                target_noise=probe.target_noise,
                sample_count=probe.sample_count,
                lid_k=probe.lid_k,
            )
        )
    return tuple(partitions)
