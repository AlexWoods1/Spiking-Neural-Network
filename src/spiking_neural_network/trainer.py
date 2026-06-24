"""Epoch training loop for ``BaseModel`` implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

import numpy as np
from tqdm import tqdm

from spiking_neural_network.config import BaseModelConfig, TrainingConfig
from spiking_neural_network.data_module import DataModule, SampleBatch
from spiking_neural_network.schedules import EpochContext, EpochTrainingState


class BaseModel(ABC):
    """Abstract classifier API used by ``Trainer``."""

    def __init__(self, config: BaseModelConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.model_name

    @abstractmethod
    def predict(self, data: np.ndarray) -> int:
        """Return the predicted class index for one spike train."""

    @abstractmethod
    def learning_rate_at(self, ctx: EpochContext) -> float:
        """Return the learning rate for the given training epoch."""

    @abstractmethod
    def train_step(
        self,
        data: np.ndarray,
        label: int,
        *,
        ctx: EpochContext,
    ) -> float:
        """Run one labeled training update and return cross-entropy loss."""

    def resolve_epoch(self, ctx: EpochContext) -> EpochTrainingState:
        return EpochTrainingState(ctx=ctx, learning_rate=self.learning_rate_at(ctx))

    @abstractmethod
    def train_batch_step(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        *,
        state: EpochTrainingState,
    ) -> float:
        """Run one mini-batch SGD update and return mean cross-entropy loss."""


class Trainer:
    """Epoch loop over ``DataModule`` batches for any ``BaseModel``."""

    def __init__(self, model: BaseModel, config: TrainingConfig) -> None:
        self.model = model
        self.config = config

    def fit(
        self,
        data_module: DataModule,
        *,
        evaluate_val: bool = True,
        evaluate_test: bool = False,
        eval_val_every: int = 1,
        eval_test_every: int = 0,
        show_progress: bool = False,
    ) -> tuple[list[dict[str, float | int]], dict[str, float]]:
        """Train for ``total_epochs`` and return per-epoch metrics and final split accuracies."""
        if eval_val_every < 1:
            raise ValueError("eval_val_every must be at least 1")
        if eval_test_every < 0:
            raise ValueError("eval_test_every must be non-negative")

        emit = tqdm.write if show_progress else print
        history: list[dict[str, float | int]] = []
        batches_per_epoch = data_module.num_batches(data_module.train)
        progress_bar = tqdm(
            total=batches_per_epoch * self.config.total_epochs,
            desc="Training",
            unit="batch",
            disable=not show_progress,
        )
        try:
            for epoch in range(1, self.config.total_epochs + 1):
                ctx = EpochContext(epoch, self.config.total_epochs)
                state = self.model.resolve_epoch(ctx)
                train_loss = self._train_epoch(
                    data_module.train_dataloader(epoch=ctx.epoch),
                    state=state,
                    progress_bar=progress_bar if show_progress else None,
                    epoch=epoch,
                )
                record: dict[str, float | int] = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": state.learning_rate,
                }
                run_val = evaluate_val and (
                    epoch % eval_val_every == 0 or epoch == self.config.total_epochs
                )
                run_test = (
                    evaluate_test
                    and eval_test_every > 0
                    and (
                        epoch % eval_test_every == 0
                        or epoch == self.config.total_epochs
                    )
                )
                if run_val:
                    val_accuracy = self.evaluate(data_module.val_dataloader())
                    record["val_acc"] = val_accuracy
                if run_test:
                    record["test_acc"] = self.evaluate(data_module.test_dataloader())
                message = f"epoch={epoch} lr={state.learning_rate:.4g} train_loss={train_loss:.4f}"
                if run_val:
                    message += f" val_acc={record['val_acc']:.1%}"
                if run_test:
                    message += f" test_acc={record['test_acc']:.1%}"
                emit(message)
                if show_progress:
                    postfix: dict[str, str] = {
                        "epoch": f"{epoch}/{self.config.total_epochs}",
                        "loss": f"{train_loss:.4f}",
                        "lr": f"{state.learning_rate:.4g}",
                    }
                    if run_val:
                        postfix["val"] = f"{record['val_acc']:.1%}"
                    if run_test:
                        postfix["test"] = f"{record['test_acc']:.1%}"
                    progress_bar.set_postfix(postfix, refresh=False)
                history.append(record)
        finally:
            progress_bar.close()

        final_accuracies = self.evaluate_splits(data_module)
        self._print_final_accuracies(final_accuracies, emit=emit)
        return history, final_accuracies

    def _train_epoch(
        self,
        loader: Iterator[SampleBatch],
        *,
        state: EpochTrainingState,
        progress_bar: tqdm | None = None,
        epoch: int | None = None,
    ) -> float:
        total_loss = 0.0
        sample_count = 0
        for batch_x, batch_y in loader:
            batch_loss = self.model.train_batch_step(batch_x, batch_y, state=state)
            batch_size = int(batch_x.shape[0])
            total_loss += batch_loss * batch_size
            sample_count += batch_size
            if progress_bar is not None:
                progress_bar.update(1)
                postfix: dict[str, str] = {"loss": f"{batch_loss:.4f}"}
                if epoch is not None:
                    postfix["epoch"] = str(epoch)
                progress_bar.set_postfix(postfix, refresh=False)
        if sample_count == 0:
            raise ValueError("train_loader produced no samples")
        return total_loss / sample_count

    def evaluate_splits(self, data_module: DataModule) -> dict[str, float]:
        """Return accuracy on train, validation, and test splits."""
        return {
            "train_acc": self.evaluate(data_module.train_dataloader()),
            "val_acc": self.evaluate(data_module.val_dataloader()),
            "test_acc": self.evaluate(data_module.test_dataloader()),
        }

    @staticmethod
    def _print_final_accuracies(
        accuracies: dict[str, float],
        *,
        emit: Callable[[str], None] = print,
    ) -> None:
        emit(
            "final "
            f"train_acc={accuracies['train_acc']:.1%} "
            f"val_acc={accuracies['val_acc']:.1%} "
            f"test_acc={accuracies['test_acc']:.1%}"
        )

    def evaluate(self, data_loader: Iterator[SampleBatch]) -> float:
        correct = 0
        total = 0
        predict_batch = getattr(self.model, "predict_batch", None)
        for batch_x, batch_y in data_loader:
            if predict_batch is not None:
                predictions = predict_batch(batch_x)
                correct += int(np.sum(predictions == batch_y))
                total += int(batch_y.shape[0])
                continue
            for sample, label in zip(self.iter_samples(batch_x), batch_y, strict=True):
                correct += self.model.predict(sample) == int(label)
                total += 1
        if total == 0:
            raise ValueError("data_loader produced no samples")
        return float(correct) / total

    @staticmethod
    def iter_samples(batch: np.ndarray) -> Iterator[np.ndarray]:
        for sample in batch:
            yield sample
