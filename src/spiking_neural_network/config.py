from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EncodingConfig:
    """Configuration for Poisson spike encoding."""

    t_steps: int = 500
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.t_steps < 1:
            raise ValueError(f"t_steps must be at least 1: {self.t_steps}")

    def make_rng(self) -> np.random.Generator:
        """Return a NumPy random generator seeded from config."""
        return np.random.default_rng(self.seed)


@dataclass(frozen=True)
class LIFConfig:
    """Configuration for the leaky integrate-and-fire model."""

    threshold: float = 1.0
    input_weight: float = 0.3
    dt: float = 1.0
    R: float = 1.0
    C: float = 5.0

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive: {self.threshold}")
        if self.input_weight <= 0:
            raise ValueError(f"input_weight must be positive: {self.input_weight}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive: {self.dt}")
        if self.R <= 0:
            raise ValueError(f"R must be positive: {self.R}")
        if self.C <= 0:
            raise ValueError(f"C must be positive: {self.C}")

    @property
    def tau(self) -> float:
        """Membrane time constant ``R * C``."""
        return self.R * self.C

    @property
    def beta(self) -> float:
        """Leak factor ``exp(-dt / tau)``."""
        return float(np.exp(-self.dt / self.tau))


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for the image preprocessing and encoding pipeline."""

    image_path: Path
    resize_shape: tuple[int, int] | None = (32, 32)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    lif: LIFConfig = field(default_factory=LIFConfig)
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
            lif=LIFConfig(threshold=1.0, input_weight=0.3, C=5.0),
        )


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for one fully connected LIF layer."""

    n_neurons: int
    lif: LIFConfig = field(default_factory=LIFConfig)

    def __post_init__(self) -> None:
        if self.n_neurons < 1:
            raise ValueError(f"n_neurons must be at least 1: {self.n_neurons}")


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for a feedforward spiking network."""

    hidden: LayerConfig
    weight_seed: int | None = 42
    weight_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.weight_scale <= 0:
            raise ValueError(f"weight_scale must be positive: {self.weight_scale}")

    def make_rng(self) -> np.random.Generator:
        """Return a NumPy random generator for weight initialization."""
        return np.random.default_rng(self.weight_seed)

    @classmethod
    def default(cls) -> "NetworkConfig":
        """Return default network settings for the demo pipeline."""
        return cls(
            hidden=LayerConfig(
                n_neurons=64,
                lif=LIFConfig(threshold=1.0, input_weight=0.3, C=5.0),
            ),
            weight_seed=42,
        )