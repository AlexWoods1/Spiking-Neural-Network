"""Backward-compatible facade for the AdaLi training pipeline."""

from __future__ import annotations

import numpy as np
from typing import Literal
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

from spiking_neural_network.evaluation import (
    build_confusion_matrix,
    classify_image,
    collect_predictions,
    predict_with_proba,
    print_prediction_summary,
    softmax,
)
from spiking_neural_network.pipeline import (
    build_adali_model,
    build_mnist_data_module,
    default_split_limits,
    full_mnist_split_sizes,
)

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
    "default_split_limits",
    "full_mnist_split_sizes",
    "ENCODING_SEED_TEST_OFFSET",
    "ENCODING_SEED_TRAIN_OFFSET",
    "ENCODING_SEED_VAL_OFFSET",
    "EpochContext",
    "EpochTrainingState",
    "ForwardCache",
    "MNISTDataConfig",
    "MNISTDataProvider",
    "MNISTSampleSource",
    "MODEL_SEED_OFFSET",
    "ModelBuilder",
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
    "cosine_learning_rate",
    "derived_seed",
    "linear_learning_rate",
]


class ModelBuilder:
    """Factory for concrete ``BaseModel`` implementations."""

    @staticmethod
    def build(
        name: str,
        config: SNN_Config,
        *,
        rng: np.random.Generator | None = None,
        backend: Literal["numpy", "jax"] = "jax",
    ) -> BaseModel:
        if name.lower() == "adali":
            if not isinstance(config, AdaLiConfig):
                raise ValueError(f"Invalid configuration for AdaLi: {config}")
            return AdaLi(config, rng=rng, backend=backend)
        raise ValueError(f"Invalid model name: {name}")
