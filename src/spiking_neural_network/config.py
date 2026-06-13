from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for the image preprocessing and encoding pipeline."""

    image_path: Path
    resize_shape: tuple[int, int] | None = (32, 32)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    show_plot: bool = True
    save_plot: Path | None = None

    def __post_init__(self) -> None:
        if self.resize_shape is not None:
            width, height = self.resize_shape
            if width < 1 or height < 1:
                raise ValueError(
                    f"resize_shape dimensions must be positive: {self.resize_shape}"
                )

    @classmethod
    def default(cls, project_root: Path) -> "PreprocessConfig":
        """Return default preprocess settings for the demo pipeline."""
        return cls(
            image_path=project_root / "Images" / "Screenshot 2025-05-02 185435.png",
            encoding=EncodingConfig(t_steps=100, seed=42),
        )
