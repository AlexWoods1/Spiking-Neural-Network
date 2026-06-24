"""Factories and public API for MNIST AdaLi training runs."""

from __future__ import annotations

from pathlib import Path

from spiking_neural_network.adali.model import AdaLi, ForwardCache, SNN_BaseModel
from spiking_neural_network.config import (
    AdaLiConfig,
    BaseModelConfig,
    DataModuleConfig,
    SNN_Config,
    TrainingConfig,
)
from spiking_neural_network.data_module import (
    ArraySampleSource,
    DataModule,
    DataProvider,
    MNISTDataConfig,
    MNISTDataProvider,
    MNISTSampleSource,
    SampleBatch,
    SampleSource,
    preencode_mnist_source,
)
from spiking_neural_network.evaluation import (
    build_confusion_matrix,
    classify_image,
    collect_predictions,
    predict_with_proba,
    print_prediction_summary,
    softmax,
)
from spiking_neural_network.exceptions import ParameterError
from spiking_neural_network.schedules import (
    BoundaryState,
    EpochContext,
    EpochTrainingState,
    cosine_learning_rate,
    linear_learning_rate,
)
from spiking_neural_network.seeds import (
    ENCODING_SEED_TEST_OFFSET,
    ENCODING_SEED_TRAIN_OFFSET,
    ENCODING_SEED_VAL_OFFSET,
    MODEL_SEED_OFFSET,
    SHUFFLE_SEED_OFFSET,
    derived_seed,
)
from spiking_neural_network.trainer import BaseModel, Trainer

MNIST_OFFICIAL_TRAIN = 60_000
MNIST_OFFICIAL_TEST = 10_000
MNIST_DEFAULT_VAL_SIZE = 10_000

__all__ = [
    "AdaLi",
    "AdaLiConfig",
    "ArraySampleSource",
    "BaseModel",
    "BaseModelConfig",
    "BoundaryState",
    "build_adali_model",
    "build_confusion_matrix",
    "build_mnist_data_module",
    "classify_image",
    "collect_predictions",
    "cosine_learning_rate",
    "DataModule",
    "DataModuleConfig",
    "DataProvider",
    "ENCODING_SEED_TEST_OFFSET",
    "ENCODING_SEED_TRAIN_OFFSET",
    "ENCODING_SEED_VAL_OFFSET",
    "EpochContext",
    "EpochTrainingState",
    "ForwardCache",
    "full_mnist_split_sizes",
    "linear_learning_rate",
    "MNISTDataConfig",
    "MNISTDataProvider",
    "MNIST_DEFAULT_VAL_SIZE",
    "MNIST_OFFICIAL_TEST",
    "MNIST_OFFICIAL_TRAIN",
    "MNISTSampleSource",
    "MODEL_SEED_OFFSET",
    "ParameterError",
    "predict_with_proba",
    "preencode_mnist_source",
    "print_prediction_summary",
    "SHUFFLE_SEED_OFFSET",
    "SNN_BaseModel",
    "SNN_Config",
    "SampleBatch",
    "SampleSource",
    "softmax",
    "Trainer",
    "TrainingConfig",
    "derived_seed",
]


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
) -> AdaLi:
    """Build an AdaLi classifier for MNIST training."""
    return AdaLi(
        AdaLiConfig(
            hidden_dims=(hidden,),
            learning_rate=cosine_learning_rate(learning_rate, lr_final),
            weight_scale=weight_scale,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            seed=seed,
        ),
    )
