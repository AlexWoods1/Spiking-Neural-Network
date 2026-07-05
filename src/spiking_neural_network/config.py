from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spiking_neural_network.exceptions import ParameterError
from spiking_neural_network.schedules import (
    DEFAULT_LEARNING_RATE,
    LearningRateSchedule,
    _validate_learning_rate,
)
from spiking_neural_network.validation import (
    require_at_least,
    require_in_range,
    require_non_negative,
    require_positive,
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
        require_at_least("total_epochs", self.total_epochs)


@dataclass(frozen=True)
class DataModuleConfig:
    """Dataset-agnostic batching configuration."""

    batch_size: int = 32
    shuffle: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        require_at_least("batch_size", self.batch_size)
        require_positive("seed", self.seed)


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
        require_positive("seed", self.seed)
        if self.input_dim is not None:
            require_positive("input_dim", self.input_dim)
        if self.output_dim is not None:
            require_positive("output_dim", self.output_dim)


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
        require_positive("dt", self.dt)
        require_positive("tau", self.tau)
        require_positive("input_dim", self.input_dim)
        require_positive("output_dim", self.output_dim)
        require_positive("weight_scale", self.weight_scale)
        if not self.hidden_dims:
            raise ParameterError("hidden_dim must be provided")
        for dim in self.hidden_dims:
            require_positive("hidden_dim", dim)
        require_in_range("decay", self.decay, 0.0, 1.0)
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
        require_positive("v_th", self.v_th)
        require_positive("alpha", self.alpha)
        require_positive("beta", self.beta)
        require_in_range("p", self.p, 0.0, 1.0)
        require_positive("left_initial", self.left_initial)
        require_positive("right_initial", self.right_initial)
        if self.left_initial >= self.right_initial:
            raise ParameterError("left_initial must be less than right_initial")
        if self.left_initial > self.v_th:
            raise ParameterError("left_initial must be less than v_th")
        if self.right_initial < self.v_th:
            raise ParameterError("right_initial must be greater than v_th")
        require_non_negative("focal_gamma", self.focal_gamma)
        if self.focal_alpha is not None:
            require_in_range(
                "focal_alpha",
                self.focal_alpha,
                0.0,
                1.0,
                high_inclusive=True,
                message_suffix=" when set",
            )
