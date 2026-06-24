"""Tests for evaluation helpers and training pipeline factories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.builder import ModelBuilder
from spiking_neural_network.config import AdaLiConfig
from spiking_neural_network.data_module import (
    ArraySampleSource,
    DataModule,
    DataModuleConfig,
)
from spiking_neural_network.config import BaseModelConfig
from spiking_neural_network.evaluation import (
    build_confusion_matrix,
    classify_image,
    collect_predictions,
    predict_with_proba,
    print_prediction_summary,
    softmax,
)
from spiking_neural_network.schedules import EpochContext, EpochTrainingState
from spiking_neural_network.trainer import BaseModel
from spiking_neural_network.pipeline import (
    build_adali_model,
    build_mnist_data_module,
    default_split_limits,
    full_mnist_split_sizes,
)
from spiking_neural_network.plotting import (
    plot_classified_image,
    plot_classified_sample_grid,
)
from spiking_neural_network.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import build_parser


class _PredictOnlyModel(BaseModel):
    """Minimal model without ``predict_proba`` for fallback coverage."""

    def __init__(self, predicted: int = 1, output_dim: int = 3) -> None:
        super().__init__(
            BaseModelConfig(
                model_name="stub",
                input_dim=784,
                output_dim=output_dim,
            )
        )
        self._predicted = predicted

    def predict(self, data: np.ndarray) -> int:
        return self._predicted

    def learning_rate_at(self, ctx: EpochContext) -> float:
        return 0.01

    def train_step(
        self,
        data: np.ndarray,
        label: int,
        *,
        ctx: EpochContext,
    ) -> float:
        return 0.0

    def train_batch_step(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        *,
        state: EpochTrainingState,
    ) -> float:
        return 0.0


def test_softmax_returns_normalized_probabilities() -> None:
    probabilities = softmax(np.array([1.0, 2.0, 3.0]))

    assert probabilities.shape == (3,)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities.argmax() == 2


def test_predict_with_proba_falls_back_to_one_hot_vector() -> None:
    model = _PredictOnlyModel(predicted=2, output_dim=4)
    sample = np.zeros((4, 784), dtype=np.float64)

    predicted, probabilities = predict_with_proba(model, sample)

    assert predicted == 2
    assert probabilities.shape == (4,)
    assert probabilities[2] == 1.0
    assert probabilities.sum() == pytest.approx(1.0)


def test_print_prediction_summary_writes_per_class_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_prediction_summary(
        np.array([0, 0, 1, 2]),
        np.array([0, 1, 1, 2]),
        num_classes=3,
    )
    captured = capsys.readouterr().out

    assert "Test prediction summary" in captured
    assert "digit 0: 2 / 1" in captured
    assert "digit 1: 1 / 2" in captured


def test_collect_predictions_returns_full_score_matrix() -> None:
    model = ModelBuilder.build("adali", AdaLiConfig(hidden_dims=(8,), output_dim=3))
    module = DataModule(DataModuleConfig(batch_size=2, shuffle=False))
    module.test = ArraySampleSource(
        np.ones((4, 4, 784), dtype=np.float64),
        np.array([0, 1, 2, 1], dtype=int),
    )

    y_true, y_pred, y_score = collect_predictions(model, module.test_dataloader())

    assert y_true.shape == (4,)
    assert y_pred.shape == (4,)
    assert y_score.shape == (4, 3)
    assert np.all(y_score >= 0.0)
    assert np.allclose(y_score.sum(axis=1), 1.0)


def test_build_confusion_matrix_keeps_full_label_grid() -> None:
    y_true = np.array([7, 7, 1, 3])
    y_pred = np.array([1, 1, 1, 3])

    matrix = build_confusion_matrix(y_true, y_pred, num_classes=10)

    assert matrix.shape == (10, 10)
    assert matrix[7, 1] == 2
    assert matrix[:, 7].sum() == 0
    assert matrix[7, 7] == 0


def test_collect_predictions_matches_model_predict() -> None:
    model = ModelBuilder.build("adali", AdaLiConfig(hidden_dims=(8,), output_dim=3))
    module = DataModule(DataModuleConfig(batch_size=2, shuffle=False))
    module.test = ArraySampleSource(
        np.ones((3, 4, 784), dtype=np.float64),
        np.array([0, 1, 2], dtype=int),
    )

    expected_predictions = []
    for batch_x, _batch_y in module.test_dataloader():
        for sample in Trainer.iter_samples(batch_x):
            expected_predictions.append(model.predict(sample))

    _y_true, y_pred, _y_score = collect_predictions(model, module.test_dataloader())

    assert y_pred.tolist() == expected_predictions


def test_collect_predictions_uses_batch_probabilities_for_jax_model() -> None:
    model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3), backend="jax")
    module = DataModule(DataModuleConfig(batch_size=2, shuffle=False))
    module.test = ArraySampleSource(
        np.ones((4, 4, 784), dtype=np.float64),
        np.array([0, 1, 2, 1], dtype=int),
    )

    y_true, y_pred, y_score = collect_predictions(model, module.test_dataloader())

    assert y_true.shape == (4,)
    assert y_pred.shape == (4,)
    assert y_score.shape == (4, 3)


def test_classify_image_returns_prediction_and_probabilities() -> None:
    model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3))
    image = np.full((28, 28), 200, dtype=np.uint8)
    rng = np.random.default_rng(0)

    predicted, probabilities = classify_image(model, image, t_steps=4, rng=rng)

    assert predicted in {0, 1, 2}
    assert probabilities.shape == (3,)
    assert np.allclose(probabilities.sum(), 1.0)


def test_plot_classified_image_runs_without_display() -> None:
    probabilities = softmax(np.array([0.1, 0.2, 0.7]))

    plot_classified_image(
        np.full((28, 28), 128, dtype=np.uint8),
        true_label=2,
        predicted=2,
        probabilities=probabilities,
        show=False,
    )


def test_plot_classified_sample_grid_runs_without_display() -> None:
    model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3))
    images = np.full((4, 28, 28), 200, dtype=np.uint8)
    labels = np.array([0, 1, 2, 1], dtype=int)

    plot_classified_sample_grid(
        model,
        images,
        labels,
        t_steps=4,
        seed=7,
        count=4,
        show=False,
    )


def test_default_split_limits_use_full_mnist() -> None:
    assert default_split_limits() == (None, None, None)
    assert full_mnist_split_sizes() == (50_000, 10_000, 10_000)


def test_build_adali_model_respects_backend() -> None:
    model = build_adali_model(
        hidden=8,
        learning_rate=0.1,
        lr_final=0.01,
        weight_scale=0.2,
        seed=1,
        backend="jax",
    )

    assert isinstance(model, AdaLi)
    assert model.backend == "jax"


def test_build_parser_default_limits_use_pipeline() -> None:
    args = build_parser().parse_args([])
    expected = default_split_limits()

    assert (args.train_limit, args.val_limit, args.test_limit) == expected


def test_build_parser_defaults_to_jax_backend() -> None:
    args = build_parser().parse_args([])
    assert args.backend == "jax"


def test_build_parser_disables_in_loop_test_eval_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.eval_test_every == 0


def test_build_mnist_data_module_accepts_default_cli_args(
    mnist_available: Path,
) -> None:
    args = build_parser().parse_args([])
    module = build_mnist_data_module(
        data_dir=mnist_available,
        t_steps=2,
        seed=1,
        batch_size=4,
        train_limit=4,
        val_limit=4,
        test_limit=4,
    )

    assert module is not None


def test_default_limits_cover_full_mnist(mnist_available: Path) -> None:
    from spiking_neural_network.data_module import MNISTDataConfig, MNISTDataProvider

    provider = MNISTDataProvider(
        MNISTDataConfig(data_dir=mnist_available, t_steps=4, preencode=False)
    )
    train, val, test = provider.build_splits()

    assert (len(train), len(val), len(test)) == full_mnist_split_sizes()
