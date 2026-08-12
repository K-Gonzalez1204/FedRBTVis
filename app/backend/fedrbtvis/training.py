from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedrbtvis.config import ModelName
from fedrbtvis.data import ClientPartition
from fedrbtvis.metrics import lid_mle
from fedrbtvis.models import build_model
from fedrbtvis.noise import inject_symmetric_noise


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    accuracy: float
    correct: int
    samples: int


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    sample_count: int
    state_dict: dict[str, torch.Tensor]
    train_loss: float
    test: EvaluationMetrics
    lid_mean: float
    lid_std: float
    actual_noise: float
    actual_emd: float
    state_sha256: str


def metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> EvaluationMetrics:
    """Compute cross entropy from logits and exact classification accuracy."""
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("logits must be 2D and labels must be 1D")
    if logits.shape[0] != labels.shape[0] or labels.numel() == 0:
        raise ValueError("logits and labels must contain the same non-zero samples")
    labels = labels.to(device=logits.device, dtype=torch.long)
    loss = functional.cross_entropy(logits, labels)
    correct = int((logits.argmax(dim=1) == labels).sum().item())
    samples = int(labels.numel())
    return EvaluationMetrics(
        loss=float(loss.item()),
        accuracy=correct / samples,
        correct=correct,
        samples=samples,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EvaluationMetrics:
    """Evaluate without gradients, accumulating loss by sample count."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long)
            logits = model(images)
            total_loss += float(
                functional.cross_entropy(logits, labels, reduction="sum").item()
            )
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += int(labels.numel())
    if was_training:
        model.train()
    if total_samples == 0:
        raise ValueError("evaluation loader is empty")
    return EvaluationMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        correct=total_correct,
        samples=total_samples,
    )


def _owned_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def train_local(
    client: ClientPartition,
    server_state: Mapping[str, torch.Tensor],
    model_name: ModelName,
    num_classes: int,
    train_images: torch.Tensor,
    clean_labels: np.ndarray,
    test_loader: DataLoader,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> ClientUpdate:
    """Train one independently owned client model and return computed metrics."""
    if local_epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("training hyperparameters must be positive")
    if len(train_images) != len(clean_labels):
        raise ValueError("training images and labels must align")

    model = build_model(model_name, num_classes, seed)
    model.load_state_dict(
        {key: value.detach().clone() for key, value in server_state.items()},
        strict=True,
    )
    model.to(device)

    indices = np.asarray(client.indices, dtype=np.int64)
    source_labels = np.asarray(clean_labels, dtype=np.int64)[indices]
    noise_rng = np.random.default_rng(
        np.random.SeedSequence((seed, client.client_id))
    )
    noise = inject_symmetric_noise(
        source_labels,
        target_rate=client.target_noise,
        num_classes=num_classes,
        rng=noise_rng,
    )
    image_indices = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(
        train_images.index_select(0, image_indices),
        torch.from_numpy(noise.labels),
    )
    loader_generator = torch.Generator().manual_seed(
        seed * 1_000_003 + client.client_id
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    processed_loss = 0.0
    processed_samples = 0
    model.train()
    for _ in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            batch_samples = int(labels.numel())
            processed_loss += float(loss.item()) * batch_samples
            processed_samples += batch_samples

    test_metrics = evaluate(model, test_loader, device)
    model.eval()
    with torch.no_grad():
        client_logits = model(
            train_images.index_select(0, image_indices).to(device)
        )
        probabilities = torch.softmax(client_logits, dim=1).cpu().numpy()
    lid = lid_mle(probabilities, client.lid_k)
    owned_state = _owned_state(model)
    return ClientUpdate(
        client_id=client.client_id,
        sample_count=client.sample_count,
        state_dict=owned_state,
        train_loss=processed_loss / processed_samples,
        test=test_metrics,
        lid_mean=float(lid.mean()),
        lid_std=float(lid.std()),
        actual_noise=noise.actual_rate,
        actual_emd=client.actual_emd,
        state_sha256=_state_sha256(owned_state),
    )


def weighted_fedavg(
    updates: Sequence[tuple[Mapping[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """Weight floating tensors by samples and max-reduce integer buffers."""
    if not updates:
        raise ValueError("at least one client update is required")
    if any(sample_count <= 0 for _, sample_count in updates):
        raise ValueError("client sample counts must be positive")

    expected_keys = set(updates[0][0])
    if any(set(state) != expected_keys for state, _ in updates):
        raise ValueError("client state dictionaries must have identical keys")
    total_samples = sum(sample_count for _, sample_count in updates)
    result: dict[str, torch.Tensor] = {}
    for key in sorted(expected_keys):
        tensors = [state[key].detach().cpu() for state, _ in updates]
        first = tensors[0]
        if any(
            tensor.shape != first.shape or tensor.dtype != first.dtype
            for tensor in tensors[1:]
        ):
            raise ValueError(f"state tensor mismatch for {key}")

        if first.is_floating_point() or first.is_complex():
            aggregate = torch.zeros_like(first)
            for tensor, (_, sample_count) in zip(tensors, updates, strict=True):
                aggregate.add_(tensor, alpha=sample_count / total_samples)
            result[key] = aggregate.clone()
        elif first.dtype == torch.bool:
            result[key] = torch.stack(tensors).any(dim=0).clone()
        else:
            result[key] = torch.stack(tensors).amax(dim=0).clone()
    return result
