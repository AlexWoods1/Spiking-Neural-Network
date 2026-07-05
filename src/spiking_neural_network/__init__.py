"""Image-to-spike encoding for spiking neural network experiments."""

from __future__ import annotations

import importlib
from typing import Any

from spiking_neural_network.config import (
    EncodingConfig,
    LayerConfig,
    LIFConfig,
    NetworkConfig,
    PreprocessConfig,
)
from spiking_neural_network.encoding import SpikeEncoding
from spiking_neural_network.exceptions import DatasetError, EncodingError, ImageError
from spiking_neural_network.validation import data_partitions, relative_error

__all__ = [
    "DataLoaderConfig",
    "DatasetError",
    "EncodingConfig",
    "EncodingError",
    "ImageError",
    "Split",
    "encode_image_spikes",
    "iter_mnist_batches",
    "iter_mnist_samples",
    "load_mnist",
    "load_mnist_bundle",
    "split_official_train_val",
    "init_weights",
    "LayerConfig",
    "LIFConfig",
    "NetworkConfig",
    "PreprocessConfig",
    "SpikeEncoding",
    "flatten_spikes",
    "forward",
    "intensity_normalize",
    "load_grayscale",
    "plot_membrane_potential",
    "plot_spike_encoding",
    "plot_spikes",
    "relative_error",
    "data_partitions",
    "resize_image",
    "show",
    "simulate_layer",
    "simulate_timesteps",
    "simulate_vector_timesteps",
    "synaptic_drive",
    "unflatten_spikes",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DataLoaderConfig": ("spiking_neural_network.datasets", "DataLoaderConfig"),
    "Split": ("spiking_neural_network.datasets", "Split"),
    "encode_image_spikes": ("spiking_neural_network.datasets", "encode_image_spikes"),
    "iter_mnist_batches": ("spiking_neural_network.datasets", "iter_mnist_batches"),
    "iter_mnist_samples": ("spiking_neural_network.datasets", "iter_mnist_samples"),
    "load_mnist": ("spiking_neural_network.datasets", "load_mnist"),
    "load_mnist_bundle": ("spiking_neural_network.datasets", "load_mnist_bundle"),
    "split_official_train_val": (
        "spiking_neural_network.datasets",
        "split_official_train_val",
    ),
    "intensity_normalize": ("spiking_neural_network.images", "intensity_normalize"),
    "load_grayscale": ("spiking_neural_network.images", "load_grayscale"),
    "resize_image": ("spiking_neural_network.images", "resize_image"),
    "show": ("spiking_neural_network.images", "show"),
    "flatten_spikes": ("spiking_neural_network.lif", "flatten_spikes"),
    "simulate_timesteps": ("spiking_neural_network.lif", "simulate_timesteps"),
    "simulate_vector_timesteps": (
        "spiking_neural_network.lif",
        "simulate_vector_timesteps",
    ),
    "synaptic_drive": ("spiking_neural_network.lif", "synaptic_drive"),
    "unflatten_spikes": ("spiking_neural_network.lif", "unflatten_spikes"),
    "forward": ("spiking_neural_network.network", "forward"),
    "init_weights": ("spiking_neural_network.network", "init_weights"),
    "simulate_layer": ("spiking_neural_network.network", "simulate_layer"),
    "plot_membrane_potential": (
        "spiking_neural_network.plotting",
        "plot_membrane_potential",
    ),
    "plot_spike_encoding": ("spiking_neural_network.plotting", "plot_spike_encoding"),
    "plot_spikes": ("spiking_neural_network.plotting", "plot_spikes"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
