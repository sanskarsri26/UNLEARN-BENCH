from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("Paired samples must have equal, non-zero length")
    if not 0 < confidence < 1 or samples <= 0:
        raise ValueError("Invalid confidence or sample count")
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(samples, len(differences)))
    estimates = np.apply_along_axis(statistic, 1, differences[indices])
    alpha = 1 - confidence
    return tuple(float(value) for value in np.quantile(estimates, [alpha / 2, 1 - alpha / 2]))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if any(value < 0 or value > 1 for value in p_values):
        raise ValueError("p-values must be in [0, 1]")
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted
