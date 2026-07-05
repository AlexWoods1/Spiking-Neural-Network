"""Shared typing aliases and lightweight protocols."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np

SampleBatch = tuple[np.ndarray, np.ndarray]


def iter_batch_samples(batch: np.ndarray) -> Iterator[np.ndarray]:
    """Yield individual samples from a batch array with a leading batch dimension."""
    for sample in batch:
        yield sample


class ClassifierModel(Protocol):
    """Minimal classifier API consumed by evaluation helpers."""

    config: Any

    def predict(self, data: np.ndarray) -> int:
        """Return the predicted class index for one input sample."""
