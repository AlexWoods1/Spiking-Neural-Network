"""Tests for the epoch training loop."""

from __future__ import annotations

import numpy as np
import pytest

from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.config import (
    AdaLiConfig,
    BaseModelConfig,
    DataModuleConfig,
    TrainingConfig,
)
from spiking_neural_network.data_module import DataModule
from spiking_neural_network.schedules import EpochContext, EpochTrainingState
from spiking_neural_network.trainer import BaseModel, Trainer
from tests.helpers import array_source, spike_batch


class _StubModel(BaseModel):
    """Minimal concrete model for ``BaseModel`` API coverage."""

    def __init__(self) -> None:
        super().__init__(
            BaseModelConfig(model_name="stub", input_dim=784, output_dim=3)
        )

    def predict(self, data: np.ndarray) -> int:
        return 0

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


class TestTrainer:
    def test_base_model_name_and_resolve_epoch(self) -> None:
        model = _StubModel()
        state = model.resolve_epoch(EpochContext(1, 3))

        assert model.name == "stub"
        assert state.learning_rate == pytest.approx(0.01)
        assert isinstance(state, EpochTrainingState)

    def test_fit_runs_multiple_epochs(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05))
        config = TrainingConfig(train_name="smoke", total_epochs=2)
        module = DataModule(DataModuleConfig(batch_size=4))
        module.train = array_source(8)
        module.val = array_source(4)
        module.test = array_source(4)
        trainer = Trainer(model, config)

        history, _final_accuracies = trainer.fit(module)

        assert len(history) == 2
        assert "train_loss" in history[0]
        assert "val_acc" in history[0]

    def test_fit_runs_with_jax_backend(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05), backend="jax")
        module = DataModule(DataModuleConfig(batch_size=4))
        module.train = array_source(8)
        module.val = array_source(4)
        module.test = array_source(4)
        trainer = Trainer(model, TrainingConfig(train_name="smoke-jax", total_epochs=2))

        history, _final_accuracies = trainer.fit(module)

        assert len(history) == 2
        assert np.isfinite(history[0]["train_loss"])

    def test_fit_can_evaluate_test_split(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05))
        module = DataModule(DataModuleConfig(batch_size=4))
        module.train = array_source(4)
        module.val = array_source(4)
        module.test = array_source(4)
        trainer = Trainer(model, TrainingConfig(train_name="smoke", total_epochs=1))

        history, _final_accuracies = trainer.fit(
            module,
            evaluate_val=False,
            evaluate_test=True,
            eval_test_every=1,
        )

        assert "test_acc" in history[0]
        assert "val_acc" not in history[0]

    def test_fit_skips_intermediate_validation_when_requested(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05))
        module = DataModule(DataModuleConfig(batch_size=4))
        module.train = array_source(8)
        module.val = array_source(4)
        module.test = array_source(4)
        trainer = Trainer(model, TrainingConfig(train_name="smoke", total_epochs=3))

        history, _final_accuracies = trainer.fit(
            module, eval_val_every=3, eval_test_every=0
        )

        assert "val_acc" not in history[0]
        assert "val_acc" not in history[1]
        assert "val_acc" in history[2]

    def test_train_epoch_rejects_empty_loader(self) -> None:
        model = AdaLi(AdaLiConfig())
        trainer = Trainer(model, TrainingConfig(train_name="x", total_epochs=1))
        state = model.resolve_epoch(EpochContext(1, 1))
        with pytest.raises(ValueError, match="no samples"):
            trainer._train_epoch(iter([]), state=state)

    def test_evaluate_rejects_empty_loader(self) -> None:
        model = AdaLi(AdaLiConfig())
        trainer = Trainer(model, TrainingConfig(train_name="x", total_epochs=1))
        with pytest.raises(ValueError, match="no samples"):
            trainer.evaluate(iter([]))

    def test_evaluate_accuracy_on_perfect_labels(self) -> None:
        model = AdaLi(AdaLiConfig())
        trainer = Trainer(model, TrainingConfig(train_name="x", total_epochs=1))
        sample = spike_batch(1)[0]
        label = model.predict(sample)
        loader = iter([(np.expand_dims(sample, axis=0), np.array([label]))])

        accuracy = trainer.evaluate(loader)

        assert accuracy == 1.0

    def test_evaluate_falls_back_without_predict_batch(self) -> None:
        model = _StubModel()
        trainer = Trainer(model, TrainingConfig(train_name="x", total_epochs=1))
        loader = iter([(spike_batch(2), np.array([0, 0]))])

        accuracy = trainer.evaluate(loader)

        assert accuracy == 1.0

    def test_fit_rejects_invalid_eval_schedule(self) -> None:
        model = AdaLi(AdaLiConfig())
        trainer = Trainer(model, TrainingConfig(train_name="x", total_epochs=1))
        module = DataModule(DataModuleConfig(batch_size=4))
        module.train = array_source(4)
        module.val = array_source(4)

        with pytest.raises(ValueError, match="eval_val_every must be at least 1"):
            trainer.fit(module, eval_val_every=0)

    def test_iter_samples_yields_batch_rows(self) -> None:
        batch = spike_batch(3)
        samples = list(Trainer.iter_samples(batch))

        assert len(samples) == 3
        assert samples[0].shape == batch[0].shape

    def test_fit_prints_final_split_accuracies(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05))
        module = DataModule(DataModuleConfig(batch_size=4))
        module.train = array_source(8)
        module.val = array_source(4)
        module.test = array_source(4)
        trainer = Trainer(model, TrainingConfig(train_name="smoke", total_epochs=1))

        trainer.fit(module, eval_val_every=999)

        output = capsys.readouterr().out
        assert "final train_acc=" in output
        assert "val_acc=" in output.split("final")[-1]
        assert "test_acc=" in output.split("final")[-1]

    def test_evaluate_splits_returns_all_splits(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05))
        module = DataModule(DataModuleConfig(batch_size=4))
        module.train = array_source(8)
        module.val = array_source(4)
        module.test = array_source(4)
        trainer = Trainer(model, TrainingConfig(train_name="smoke", total_epochs=1))

        accuracies = trainer.evaluate_splits(module)

        assert set(accuracies) == {"train_acc", "val_acc", "test_acc"}
        assert all(0.0 <= value <= 1.0 for value in accuracies.values())

    def test_fit_runs_with_progress_enabled(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05))
        module = DataModule(DataModuleConfig(batch_size=4))
        module.train = array_source(8)
        module.val = array_source(4)
        module.test = array_source(4)
        trainer = Trainer(model, TrainingConfig(train_name="smoke", total_epochs=1))

        history, _final_accuracies = trainer.fit(
            module, show_progress=True, eval_val_every=999
        )

        assert len(history) == 1
