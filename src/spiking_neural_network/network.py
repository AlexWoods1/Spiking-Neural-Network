"""Feedforward spiking network composition."""

from typing import TypedDict

import numpy as np

from spiking_neural_network.config import LayerConfig, NetworkConfig
from spiking_neural_network.lif import (
    flatten_spikes,
    simulate_vector_timesteps,
    synaptic_drive,
)


class ForwardResult(TypedDict):
    input: np.ndarray
    weights_input_hidden: np.ndarray
    hidden_membrane: np.ndarray
    hidden_spikes: np.ndarray
    spatial_shape: tuple[int, ...] | None


def init_weights(
    n_pre: int,
    n_post: int,
    rng: np.random.Generator,
    scale: float = 0.01,
) -> np.ndarray:
    """Initialize a synaptic weight matrix with shape ``(N_post, N_pre)``."""
    if n_pre < 1 or n_post < 1:
        raise ValueError(
            f"n_pre and n_post must be at least 1: n_pre={n_pre}, n_post={n_post}"
        )
    if scale <= 0:
        raise ValueError(f"scale must be positive: {scale}")

    return scale * rng.standard_normal((n_post, n_pre))


def simulate_layer(
    spikes_in: np.ndarray,
    weights: np.ndarray,
    config: LayerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one fully connected LIF layer.

    Args:
        spikes_in: Presynaptic spikes with shape ``(T, N_pre)``.
        weights: Synaptic weights with shape ``(N_post, N_pre)``.
        config: Layer configuration including LIF parameters.

    Returns:
        Tuple of membrane potentials and spike outputs, each ``(T, N_post)``.
    """
    drive = synaptic_drive(spikes_in, weights)
    return simulate_vector_timesteps(drive, config.lif)


def forward(
    input_spikes: np.ndarray,
    network_config: NetworkConfig,
) -> ForwardResult:
    """Run a feedforward pass through the hidden layer.

    Args:
        input_spikes: Input spikes with shape ``(T, *spatial)`` or ``(T, N_in)``.
        network_config: Network configuration and weight initialization settings.

    Returns:
        Dictionary with flattened input, weights, hidden outputs, and spatial shape.
    """
    if input_spikes.ndim > 2:
        flat, spatial_shape = flatten_spikes(input_spikes)
    else:
        flat = input_spikes
        spatial_shape = None

    n_in = flat.shape[1]
    n_hidden = network_config.hidden.n_neurons
    rng = network_config.make_rng()
    weights = init_weights(
        n_pre=n_in,
        n_post=n_hidden,
        rng=rng,
        scale=network_config.weight_scale,
    )
    hidden_membrane, hidden_spikes = simulate_layer(
        flat,
        weights,
        network_config.hidden,
    )

    return {
        "input": flat,
        "weights_input_hidden": weights,
        "hidden_membrane": hidden_membrane,
        "hidden_spikes": hidden_spikes,
        "spatial_shape": spatial_shape,
    }
