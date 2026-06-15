"""Leaky integrate-and-fire neuron simulation.

U[t] = (U[t-1] - H[t-1]) * beta + input_weight * x[t]
H[t] = 0 if U[t] < threshold, else 1
"""

import numpy as np

from spiking_neural_network.config import LIFConfig


def heaviside(U: np.ndarray, threshold: float) -> np.ndarray:
    """Return 1 where membrane potential is at or above threshold."""
    return np.where(U < threshold, 0, 1)


def flatten_spikes(spikes: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flatten spike tensor ``(T, *spatial)`` to ``(T, N)``.

    Args:
        spikes: Spike array with time as the first axis.

    Returns:
        Tuple of the flattened array and the original spatial shape.
    """
    if spikes.ndim < 2:
        raise ValueError(f"spikes must have shape (T, ...); got {spikes.shape}")

    spatial_shape = spikes.shape[1:]
    flat = spikes.reshape(spikes.shape[0], -1)
    return flat, spatial_shape


def unflatten_spikes(spikes: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Restore ``(T, N)`` spikes to ``(T, *spatial)``.

    Args:
        spikes: Flat spike array with shape ``(T, N)``.
        spatial_shape: Target spatial dimensions after the time axis.

    Returns:
        Spike array with shape ``(T, *spatial_shape)``.
    """
    expected_neurons = int(np.prod(spatial_shape))
    if spikes.ndim != 2 or spikes.shape[1] != expected_neurons:
        raise ValueError(
            f"spikes must have shape (T, {expected_neurons}); got {spikes.shape}"
        )

    return spikes.reshape(spikes.shape[0], *spatial_shape)


def synaptic_drive(spikes_in: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute synaptic input drive from presynaptic spikes.

    For each timestep ``t``, ``drive[t] = weights @ spikes_in[t]``.

    Args:
        spikes_in: Presynaptic spikes with shape ``(T, N_pre)``.
        weights: Synaptic weight matrix with shape ``(N_post, N_pre)``.

    Returns:
        Synaptic drive with shape ``(T, N_post)``.
    """
    if spikes_in.ndim != 2:
        raise ValueError(f"spikes_in must have shape (T, N_pre); got {spikes_in.shape}")
    if weights.ndim != 2:
        raise ValueError(
            f"weights must have shape (N_post, N_pre); got {weights.shape}"
        )
    if spikes_in.shape[1] != weights.shape[1]:
        raise ValueError(
            "spikes_in neuron count must match weights columns: "
            f"{spikes_in.shape[1]} vs {weights.shape[1]}"
        )

    return spikes_in @ weights.T


def _simulate_vector_core(
    X: np.ndarray, config: LIFConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Run LIF dynamics on vector input ``(T, N)``."""
    timesteps, n_neurons = X.shape
    U = np.zeros((timesteps, n_neurons))
    H = np.zeros((timesteps, n_neurons))

    for i in range(timesteps):
        if i == 0:
            U[i] = config.input_weight * X[i]
        else:
            U[i] = (U[i - 1] - H[i - 1]) * config.beta + config.input_weight * X[i]
        H[i] = heaviside(U[i], config.threshold)

    return U, H


def simulate_vector_timesteps(
    X: np.ndarray, config: LIFConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate LIF neurons driven by a vector input per timestep.

    Args:
        X: Input drive with shape ``(T, N)``.
        config: LIF parameters. Defaults to ``LIFConfig()``.

    Returns:
        Tuple ``(U, H)`` of membrane potentials and spike outputs, each with
        shape ``(T, N)``.
    """
    if config is None:
        config = LIFConfig()
    if X.ndim != 2:
        raise ValueError(f"X must have shape (T, N); got {X.shape}")

    return _simulate_vector_core(X, config)


def simulate_timesteps(
    X: np.ndarray, config: LIFConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate timesteps of the leaky integrate-and-fire model.

    Args:
        X: Input spike drive with shape ``(T, *spatial)`` — one frame per timestep.
        config: LIF parameters. Defaults to ``LIFConfig()``.

    Returns:
        Tuple ``(U, H)`` of membrane potentials and spike outputs, each with
        shape ``(T, *spatial)``.
    """
    if config is None:
        config = LIFConfig()
    if X.ndim < 2:
        raise ValueError(f"X must have shape (T, ...); got {X.shape}")

    flat_input, spatial_shape = flatten_spikes(X)
    flat_membrane, flat_spikes = simulate_vector_timesteps(flat_input, config)
    membrane = unflatten_spikes(flat_membrane, spatial_shape)
    spikes = unflatten_spikes(flat_spikes, spatial_shape)
    return membrane, spikes
