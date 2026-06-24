"""Inference helpers and classification metrics for trained models."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from spiking_neural_network.data_module import SampleBatch
from spiking_neural_network.datasets import encode_image_spikes
from spiking_neural_network.trainer import BaseModel, Trainer


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return numerically stable softmax probabilities for one logit vector."""
    shifted = logits - logits.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def predict_with_proba(
    model: BaseModel,
    sample: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Return ``(predicted_label, class_probabilities)`` for one spike train."""
    predict_proba = getattr(model, "predict_proba", None)
    if predict_proba is not None:
        probabilities = np.asarray(predict_proba(sample))
        return int(np.argmax(probabilities)), probabilities

    predicted = model.predict(sample)
    output_dim = getattr(model.config, "output_dim", predicted + 1)
    probabilities = np.zeros(output_dim, dtype=np.float64)
    probabilities[predicted] = 1.0
    return predicted, probabilities


def classify_image(
    model: BaseModel,
    image: np.ndarray,
    *,
    t_steps: int,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    """Encode one MNIST image and return ``(predicted_label, class_probabilities)``."""
    spikes = encode_image_spikes(image, t_steps, rng)
    return predict_with_proba(model, spikes)


def collect_predictions(
    model: BaseModel,
    loader: Iterator[SampleBatch],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred, y_score)`` for samples from a dataloader."""
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[np.ndarray] = []
    predict_proba_batch = getattr(model, "predict_proba_batch", None)

    for batch_x, batch_y in loader:
        if predict_proba_batch is not None:
            probabilities = np.asarray(predict_proba_batch(batch_x))
            predictions = np.argmax(probabilities, axis=1)
            y_true.extend(int(label) for label in batch_y)
            y_pred.extend(int(prediction) for prediction in predictions)
            y_score.extend(probabilities)
            continue

        for sample, label in zip(Trainer.iter_samples(batch_x), batch_y, strict=True):
            predicted, probabilities = predict_with_proba(model, sample)
            y_true.append(int(label))
            y_pred.append(predicted)
            y_score.append(probabilities)

    return np.array(y_true), np.array(y_pred), np.stack(y_score)


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return a ``(num_classes, num_classes)`` confusion matrix with fixed label order."""
    from sklearn.metrics import confusion_matrix

    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def print_prediction_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> None:
    """Print per-class true and predicted counts."""
    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    print("Test prediction summary (true_count / pred_count):")
    for digit in range(num_classes):
        print(f"  digit {digit}: {true_counts[digit]} / {pred_counts[digit]}")
