"""Factories for MNIST AdaLi training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.config import AdaLiConfig, DataModuleConfig
from spiking_neural_network.data_module import (
    DataModule,
    MNISTDataConfig,
)
from spiking_neural_network.schedules import cosine_learning_rate

MNIST_OFFICIAL_TRAIN = 60_000
MNIST_OFFICIAL_TEST = 10_000
MNIST_DEFAULT_VAL_SIZE = 10_000


def default_split_limits() -> tuple[None, None, None]:
    """Return default limits that use all available MNIST train, val, and test samples."""
    return None, None, None


def full_mnist_split_sizes(
    val_size: int = MNIST_DEFAULT_VAL_SIZE,
) -> tuple[int, int, int]:
    """Return sample counts when every MNIST split is used without limits."""
    return (
        MNIST_OFFICIAL_TRAIN - val_size,
        val_size,
        MNIST_OFFICIAL_TEST,
    )


def build_mnist_data_module(
    *,
    data_dir: Path,
    t_steps: int,
    seed: int,
    batch_size: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
    preencode: bool | None = None,
) -> DataModule:
    """Build a ``DataModule`` for MNIST spike training."""
    return DataModule.from_mnist(
        MNISTDataConfig(
            data_dir=data_dir,
            t_steps=t_steps,
            seed=seed,
            train_limit=train_limit,
            val_limit=val_limit,
            test_limit=test_limit,
            preencode=preencode,
        ),
        DataModuleConfig(batch_size=batch_size, shuffle=True, seed=seed),
    )


def build_adali_model(
    *,
    hidden: int,
    learning_rate: float,
    lr_final: float,
    weight_scale: float,
    seed: int,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    backend: Literal["numpy", "jax"] = "jax",
) -> AdaLi:
    """Build an AdaLi classifier for MNIST training."""
    model = AdaLi(
        AdaLiConfig(
            hidden_dims=(hidden,),
            learning_rate=cosine_learning_rate(learning_rate, lr_final),
            weight_scale=weight_scale,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            seed=seed,
        ),
        backend=backend,
    )
    return model
