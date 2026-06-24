"""Pure NumPy forward/backward ops for AdaLi SNN training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spiking_neural_network.lif import heaviside
from spiking_neural_network.schedules import BoundaryState

SurrogateFn = Callable[[np.ndarray, float, float], np.ndarray]


@dataclass
class ForwardCache:
    """Forward-pass tensors retained for AdaLi BPTT."""

    u_hist: list[list[np.ndarray]]
    pre_syn: list[np.ndarray]
    tw: np.ndarray
    logits: np.ndarray

    @property
    def timesteps(self) -> int:
        return len(self.tw)


def init_weights(
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    weight_scale: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Initialize layer weight matrices for a feedforward SNN."""
    dims = [input_dim, *hidden_dims, output_dim]
    return [
        weight_scale * rng.standard_normal((dims[index + 1], dims[index]))
        for index in range(len(dims) - 1)
    ]


def forward_pass(
    weights: list[np.ndarray],
    input_data: np.ndarray,
    *,
    leak: float,
    v_th: float,
    decay: float,
) -> ForwardCache:
    """Simulate layered LIF dynamics and decay-weighted output logits."""
    timesteps, _ = input_data.shape
    u_hist: list[list[np.ndarray]] = [[] for _ in weights]
    s_hist: list[list[np.ndarray]] = [[] for _ in weights]
    membrane = [np.zeros(layer.shape[0]) for layer in weights]

    for timestep in range(timesteps):
        pre_synaptic = input_data[timestep]
        for layer_index, layer_weights in enumerate(weights):
            membrane_potential = leak * membrane[layer_index] + (1.0 - leak) * (
                pre_synaptic @ layer_weights.T
            )
            spikes = heaviside(membrane_potential, v_th)
            u_hist[layer_index].append(membrane_potential)
            s_hist[layer_index].append(spikes)
            membrane[layer_index] = membrane_potential - v_th * spikes
            pre_synaptic = spikes

    spike_history = [np.stack(layer, axis=0) for layer in s_hist]
    temporal_weights = decay ** np.arange(timesteps - 1, -1, -1)
    logits = v_th * (temporal_weights @ spike_history[-1]) / temporal_weights.sum()

    return ForwardCache(
        u_hist=u_hist,
        pre_syn=[input_data, *spike_history[:-1]],
        tw=temporal_weights,
        logits=logits,
    )


def softmax_loss(logits: np.ndarray, label: int) -> tuple[float, np.ndarray]:
    """Return softmax cross-entropy loss and output gradient."""
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    loss = float(-np.log(probs[label]))
    d_out = probs.copy()
    d_out[label] -= 1.0
    return loss, d_out


def focal_loss(
    logits: np.ndarray,
    label: int,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> tuple[float, np.ndarray]:
    """Return multiclass focal loss and output gradient."""
    cross_entropy, d_cross_entropy = softmax_loss(logits, label)
    true_class_probability = float(np.exp(-cross_entropy))
    modulating = (1.0 - true_class_probability) ** gamma
    loss = modulating * cross_entropy
    d_loss_d_cross_entropy = modulating + (
        cross_entropy * gamma * (1.0 - true_class_probability) ** (gamma - 1) * true_class_probability
    )
    if alpha is not None:
        loss = alpha * loss
        d_loss_d_cross_entropy *= alpha
    d_out = d_cross_entropy * d_loss_d_cross_entropy
    return loss, d_out


def output_spike_gradients(
    weights: list[np.ndarray],
    cache: ForwardCache,
    d_out: np.ndarray,
    *,
    v_th: float,
) -> list[np.ndarray]:
    """Seed output-layer spike gradients from decay-weighted logits."""
    timesteps = cache.timesteps
    d_spikes = [np.zeros((timesteps, layer.shape[0])) for layer in weights]
    d_spikes[-1] = np.outer(cache.tw, d_out) * (v_th / cache.tw.sum())
    return d_spikes


def backward_pass(
    weights: list[np.ndarray],
    d_spikes: list[np.ndarray],
    cache: ForwardCache,
    boundary: BoundaryState,
    *,
    leak: float,
    surrogate: SurrogateFn,
) -> list[np.ndarray]:
    """Backpropagate through time with a surrogate spike gradient."""
    d_weights = [np.zeros_like(weight) for weight in weights]
    for layer in range(len(weights) - 1, -1, -1):
        d_u_next = np.zeros(weights[layer].shape[0])
        for timestep in range(cache.timesteps - 1, -1, -1):
            d_u = (
                d_spikes[layer][timestep]
                * surrogate(
                    cache.u_hist[layer][timestep],
                    boundary.v_minus,
                    boundary.v_plus,
                )
            )
            d_u += leak * d_u_next
            d_weights[layer] += np.outer(d_u, cache.pre_syn[layer][timestep])
            if layer > 0:
                d_spikes[layer - 1][timestep] += weights[layer].T @ d_u
            d_u_next = d_u
    return d_weights


def update_weights(
    weights: list[np.ndarray],
    gradient: list[np.ndarray],
    learning_rate: float,
) -> list[np.ndarray]:
    """Apply one SGD step to layer weights."""
    return [
        layer_weights - learning_rate * grad
        for layer_weights, grad in zip(weights, gradient, strict=True)
    ]


def adali_surrogate(
    u: np.ndarray,
    v_minus: float,
    v_plus: float,
    *,
    v_th: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Return AdaLi surrogate gradient for one membrane snapshot."""
    slope = np.where(
        u < v_th,
        alpha / (v_th - v_minus),
        beta / (v_plus - v_th),
    )
    return np.where((u < v_minus) | (u > v_plus), 0.0, slope)
