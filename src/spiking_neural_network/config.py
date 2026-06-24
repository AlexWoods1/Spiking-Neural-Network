from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spiking_neural_network.exceptions import ParameterError
from spiking_neural_network.schedules import (
    DEFAULT_LEARNING_RATE,
    LearningRateSchedule,
    _validate_learning_rate,
)


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


### for LIF Model.


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


@dataclass(frozen=True)
class TrainingConfig:
    """Training schedule configuration."""

    train_name: str
    total_epochs: int

    def __post_init__(self) -> None:
        if self.train_name is None:
            raise ParameterError("train_name must be provided")
        if self.train_name is not None and not isinstance(self.train_name, str):
            raise ParameterError("train_name must be a string")
        if self.total_epochs is not None and not isinstance(self.total_epochs, int):
            raise ParameterError("epochs must be an integer")
        if self.total_epochs < 1:
            raise ParameterError("total_epochs must be at least 1")


@dataclass(frozen=True)
class DataModuleConfig:
    """Dataset-agnostic batching configuration."""

    batch_size: int = 32
    shuffle: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ParameterError("batch_size must be at least 1")
        if self.seed <= 0:
            raise ParameterError("seed must be positive")


@dataclass(frozen=True)
class BaseModelConfig:
    """Base metadata shared by model configuration dataclasses."""

    model_name: str
    input_dim: int | None
    output_dim: int | None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.model_name is None:
            raise ParameterError("model_name must be provided")
        if self.model_name is not None and not isinstance(self.model_name, str):
            raise ParameterError("model_name must be a string")
        if self.input_dim is not None and not isinstance(self.input_dim, int):
            raise ParameterError("input_dim must be an integer")
        if self.output_dim is not None and not isinstance(self.output_dim, int):
            raise ParameterError("output_dim must be an integer")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ParameterError("seed must be an integer")

        if self.seed <= 0:
            raise ParameterError("seed must be positive")

        if self.input_dim is not None and self.input_dim <= 0:
            raise ParameterError("input_dim must be positive")

        if self.output_dim is not None and self.output_dim <= 0:
            raise ParameterError("output_dim must be positive")


@dataclass(frozen=True)
class SNN_Config(BaseModelConfig):
    """Shared spiking-network hyperparameters."""

    model_name: str = "SNN"
    input_dim: int = 784
    hidden_dims: tuple[int, ...] = (128,)
    output_dim: int = 10
    weight_scale: float = 0.5
    dt: float = 1.0
    tau: float = 2.0
    v_th: float = 1.0
    decay: float = 0.9
    learning_rate: float | LearningRateSchedule = DEFAULT_LEARNING_RATE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dt <= 0:
            raise ParameterError("dt must be positive")
        if self.tau <= 0:
            raise ParameterError("tau must be positive")
        if self.input_dim <= 0:
            raise ParameterError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ParameterError("output_dim must be positive")
        if self.weight_scale <= 0:
            raise ParameterError("weight_scale must be positive")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ParameterError("hidden_dim must be positive")
        if not self.hidden_dims:
            raise ParameterError("hidden_dim must be provided")
        if self.decay <= 0 or self.decay >= 1:
            raise ParameterError("decay must be between 0 and 1")
        _validate_learning_rate(self.learning_rate)


@dataclass(frozen=True)
class AdaLiConfig(SNN_Config):
    """AdaLi-specific surrogate and boundary schedule parameters."""

    model_name: str = "adaLi"
    alpha: float = 0.5
    beta: float = 0.5
    p: float = 0.2
    left_initial: float = 0.5
    right_initial: float = 1.5
    focal_gamma: float = 2.0
    focal_alpha: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.v_th <= 0:
            raise ParameterError("v_th must be positive")
        if self.alpha <= 0:
            raise ParameterError("alpha must be positive")
        if self.beta <= 0:
            raise ParameterError("beta must be positive")
        if not (0.0 < self.p < 1.0):
            raise ParameterError("p must be between 0 and 1")
        if self.left_initial <= 0:
            raise ParameterError("left_initial must be positive")
        if self.right_initial <= 0:
            raise ParameterError("right_initial must be positive")
        if self.left_initial >= self.right_initial:
            raise ParameterError("left_initial must be less than right_initial")
        if self.left_initial > self.v_th:
            raise ParameterError("left_initial must be less than v_th")
        if self.right_initial < self.v_th:
            raise ParameterError("right_initial must be greater than v_th")
        if self.focal_gamma < 0:
            raise ParameterError("focal_gamma must be non-negative")
        if self.focal_alpha is not None and not (0.0 < self.focal_alpha <= 1.0):
            raise ParameterError("focal_alpha must be in (0, 1] when set")
