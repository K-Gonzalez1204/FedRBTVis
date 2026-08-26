import numpy as np


def _normalized_distribution(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional categorical vector")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return array / total


def categorical_emd_01(
    histogram: np.ndarray,
    reference: np.ndarray | None = None,
) -> float:
    """Return 0.5 * L1 distance between categorical distributions."""
    distribution = _normalized_distribution(histogram, "histogram")
    if reference is None:
        comparison = np.full(distribution.shape, 1.0 / distribution.size)
    else:
        comparison = _normalized_distribution(reference, "reference")
        if comparison.shape != distribution.shape:
            raise ValueError("histogram and reference must have the same shape")
    return float(0.5 * np.abs(distribution - comparison).sum())


def lid_mle(probabilities: np.ndarray, k: int, eps: float = 1e-6) -> np.ndarray:
    """Compute per-row LID after excluding each row's self-distance."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    sample_count = values.shape[0]
    if sample_count < 2:
        raise ValueError("at least two samples are required")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if not isinstance(k, (int, np.integer)) or not 2 <= int(k) < sample_count:
        raise ValueError("k must satisfy 2 <= k < sample count")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")

    deltas = values[:, None, :] - values[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.sort(distances, axis=1)[:, : int(k)]
    radius = nearest[:, -1:]

    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.log((nearest + eps) / (radius + eps)).sum(axis=1)
        result = -float(k) / denominator
    if not np.isfinite(nearest).all() or not np.isfinite(result).all():
        raise ValueError("LID is undefined for the supplied distances")
    return result
