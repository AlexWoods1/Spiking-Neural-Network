from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncodingConfig:
    """Configuration for Poisson spike encoding."""

    t_steps: int = 100
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.t_steps < 1:
            raise ValueError(f"t_steps must be at least 1: {self.t_steps}")

    def make_rng(self) -> np.random.Generator:
        """Return a NumPy random generator seeded from config."""
        return np.random.default_rng(self.seed)
