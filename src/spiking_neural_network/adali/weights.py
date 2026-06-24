"""Weight initialization for AdaLi feedforward SNNs."""

from __future__ import annotations

import numpy as np


def init_weights(
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    weight_scale: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Initialize layer weight matrices for a feedforward SNN."""
    dims = [input_dim, *hidden_dims, output_dim]
    return [
        weight_scale * rng.standard_normal((dims[index + 1], dims[index]))
        for index in range(len(dims) - 1)
    ]
