"""Shared helpers for training pipeline tests."""

from __future__ import annotations

import numpy as np

from spiking_neural_network.data_module import ArraySampleSource


def spike_batch(count: int, t_steps: int = 4, features: int = 784) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((count, t_steps, features)) < 0.2).astype(np.float64)


def array_source(count: int) -> ArraySampleSource:
    return ArraySampleSource(spike_batch(count), np.arange(count, dtype=int) % 10)
