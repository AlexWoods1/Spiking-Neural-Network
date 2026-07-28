"""Configuration dataclasses and YAML loading for SpikedLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def _require_int_ge(name: str, value: int, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _require_float_gt(name: str, value: float, minimum: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if float(value) <= minimum:
        raise ValueError(f"{name} must be > {minimum}, got {value}")


def _require_float_ge(name: str, value: float, minimum: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _require_in_range(
    name: str, value: float, low: float, high: float, *, inclusive: bool = True
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    v = float(value)
    ok = (low <= v <= high) if inclusive else (low < v < high)
    if not ok:
        bounds = f"[{low}, {high}]" if inclusive else f"({low}, {high})"
        raise ValueError(f"{name} must be in {bounds}, got {value}")


def _require_divisible(name_a: str, a: int, name_b: str, b: int) -> None:
    if b == 0 or a % b != 0:
        raise ValueError(f"{name_a} ({a}) must be divisible by {name_b} ({b})")


@dataclass
class ModelConfig:
    """Decoder size and Spiking LSTM / AdaLi spike hyperparameters."""

    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout: float = 0.0
    bias: bool = True
    v_th: float = 0.5
    leak: float = 0.5
    v_minus: float = -1.0
    v_plus: float = 2.0
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        _require_int_ge("n_layer", self.n_layer, 1)
        _require_int_ge("n_head", self.n_head, 1)
        _require_int_ge("n_embd", self.n_embd, 1)
        _require_int_ge("block_size", self.block_size, 1)
        _require_int_ge("vocab_size", self.vocab_size, 2)
        _require_in_range("dropout", self.dropout, 0.0, 1.0)
        _require_divisible("n_embd", self.n_embd, "n_head", self.n_head)
        _require_float_gt("v_th", self.v_th, 0.0)
        _require_in_range("leak", self.leak, 0.0, 1.0)
        _require_float_ge("alpha", self.alpha, 0.0)
        _require_float_ge("beta", self.beta, 0.0)
        if self.v_minus >= self.v_th:
            raise ValueError("v_minus must be < v_th")
        if self.v_plus <= self.v_th:
            raise ValueError("v_plus must be > v_th")


@dataclass
class TrainConfig:
    """Optax training hyperparameters."""

    batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    grad_clip: float
    eval_interval: int
    eval_batches: int
    sample_interval: int
    checkpoint_interval: int
    seed: int

    def __post_init__(self) -> None:
        _require_int_ge("batch_size", self.batch_size, 1)
        _require_int_ge("max_steps", self.max_steps, 1)
        _require_float_gt("learning_rate", self.learning_rate, 0.0)
        _require_float_ge("weight_decay", self.weight_decay, 0.0)
        _require_in_range("beta1", self.beta1, 0.0, 1.0)
        _require_in_range("beta2", self.beta2, 0.0, 1.0)
        _require_int_ge("warmup_steps", self.warmup_steps, 0)
        _require_float_ge("grad_clip", self.grad_clip, 0.0)
        _require_int_ge("eval_interval", self.eval_interval, 1)
        _require_int_ge("eval_batches", self.eval_batches, 1)
        _require_int_ge("sample_interval", self.sample_interval, 1)
        _require_int_ge("checkpoint_interval", self.checkpoint_interval, 1)
        _require_int_ge("seed", self.seed, 0)


@dataclass
class DataConfig:
    """Shakespeare data locations."""

    dataset: str
    data_dir: Path
    train_frac: float

    def __post_init__(self) -> None:
        _require_in_range("train_frac", self.train_frac, 0.0, 1.0, inclusive=False)
        self.data_dir = Path(self.data_dir)
        if not self.dataset:
            raise ValueError("dataset must be a non-empty string")


@dataclass
class PathsConfig:
    """Checkpoint and tokenizer output paths."""

    out_dir: Path
    tokenizer_path: Path

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.tokenizer_path = Path(self.tokenizer_path)


@dataclass
class Config:
    """Top-level SpikedLM config."""

    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    paths: PathsConfig


def load_config(path: Path | str) -> Config:
    """Load a config from a YAML file.

    Args:
        path: Path to a YAML config file.

    Returns:
        Parsed ``Config`` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML root is not a mapping or required keys are missing.
        TypeError: If field types are invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config file must be a dictionary")

    try:
        return Config(
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw["train"]),
            data=DataConfig(**raw["data"]),
            paths=PathsConfig(**raw["paths"]),
        )
    except KeyError as e:
        raise ValueError(f"Missing required key: {e} in {path}") from e
    except TypeError as e:
        raise TypeError(f"Invalid config fields in {path}: {e}") from e
