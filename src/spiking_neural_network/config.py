from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncodingConfig:
    t_steps: int = 100
    seed: int | None = None

    def make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)
