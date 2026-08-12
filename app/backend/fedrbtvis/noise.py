from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class NoiseResult:
    labels: npt.NDArray[np.int64]
    changed_indices: npt.NDArray[np.int64]
    actual_rate: float


def inject_symmetric_noise(
    labels: npt.NDArray[np.int64],
    target_rate: float,
    num_classes: int,
    rng: np.random.Generator,
) -> NoiseResult:
    """Return a noisy copy; every selected label changes to another class."""
    source = np.asarray(labels)
    if source.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("labels must contain integers")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if not np.isfinite(target_rate) or not 0.0 <= target_rate <= 1.0:
        raise ValueError("target_rate must be between zero and one")
    if np.any(source < 0) or np.any(source >= num_classes):
        raise ValueError("labels fall outside the configured class range")

    noisy = source.astype(np.int64, copy=True)
    changed_count = round(len(noisy) * target_rate)
    if changed_count == 0:
        changed_indices = np.empty(0, dtype=np.int64)
    else:
        changed_indices = np.asarray(
            rng.choice(len(noisy), size=changed_count, replace=False),
            dtype=np.int64,
        )
        alternatives = rng.integers(
            0,
            num_classes - 1,
            size=changed_count,
            dtype=np.int64,
        )
        original = noisy[changed_indices]
        noisy[changed_indices] = alternatives + (alternatives >= original)

    actual_rate = changed_count / len(noisy) if len(noisy) else 0.0
    return NoiseResult(noisy, changed_indices, actual_rate)
