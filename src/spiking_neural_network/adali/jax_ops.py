"""Pure JAX forward/backward ops for AdaLi SNN training."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError as exc:
    raise ImportError("Install jax and optax to use the JAX backend") from exc

from spiking_neural_network.adali import jax_losses

_STATIC_TRAIN_ARGS = (
    "leak",
    "v_th",
    "decay",
    "v_minus",
    "v_plus",
    "alpha",
    "beta",
    "learning_rate",
    "focal_gamma",
    "focal_alpha",
)


@dataclass(frozen=True)
class ForwardCache:
    """Forward-pass tensors retained for AdaLi BPTT."""

    u_hist: tuple[jax.Array, ...]
    pre_syn: tuple[jax.Array, ...]
    tw: jax.Array
    logits: jax.Array

    @property
    def timesteps(self) -> int:
        return int(self.u_hist[0].shape[0])


def heaviside(u: jax.Array, v_th: float) -> jax.Array:
    """Return hard spikes with zero gradient through the forward pass."""
    spikes = jnp.where(u >= v_th, 1.0, 0.0)
    return lax.stop_gradient(spikes)


def adali_surrogate(
    u: jax.Array,
    v_minus: float,
    v_plus: float,
    *,
    v_th: float,
    alpha: float,
    beta: float,
) -> jax.Array:
    """Return AdaLi surrogate gradient for one membrane snapshot."""
    slope = jnp.where(
        u < v_th,
        alpha / (v_th - v_minus),
        beta / (v_plus - v_th),
    )
    return jnp.where((u < v_minus) | (u > v_plus), 0.0, slope)


def forward_pass(
    weights: tuple[jax.Array, ...],
    input_data: jax.Array,
    *,
    leak: float,
    v_th: float,
    decay: float,
) -> ForwardCache:
    """Simulate layered LIF dynamics for one sample with shape ``(T, F)``."""
    u_hist_list: list[jax.Array] = []
    s_hist_list: list[jax.Array] = []
    pre_synaptic = input_data

    for layer_weights in weights:
        n_neurons = layer_weights.shape[0]

        def layer_step(
            membrane: jax.Array,
            pre_t: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            membrane_potential = leak * membrane + (1.0 - leak) * (
                pre_t @ layer_weights.T
            )
            spikes = heaviside(membrane_potential, v_th)
            membrane_new = membrane_potential - v_th * spikes
            return membrane_new, (membrane_potential, spikes)

        _, (u_layer, s_layer) = lax.scan(
            layer_step,
            jnp.zeros(n_neurons, dtype=input_data.dtype),
            pre_synaptic,
        )
        u_hist_list.append(u_layer)
        s_hist_list.append(s_layer)
        pre_synaptic = s_layer

    timesteps = input_data.shape[0]
    temporal_weights = decay ** jnp.arange(
        timesteps - 1,
        -1,
        -1,
        dtype=input_data.dtype,
    )
    logits = v_th * (temporal_weights @ s_hist_list[-1]) / temporal_weights.sum()

    return ForwardCache(
        u_hist=tuple(u_hist_list),
        pre_syn=(input_data, *tuple(s_hist_list[:-1])),
        tw=temporal_weights,
        logits=logits,
    )


def forward_pass_batched(
    weights: tuple[jax.Array, ...],
    input_data: jax.Array,
    *,
    leak: float,
    v_th: float,
    decay: float,
) -> ForwardCache:
    """Simulate layered LIF dynamics for a batch with shape ``(B, T, F)``."""
    batch_size = input_data.shape[0]
    u_hist_list: list[jax.Array] = []
    s_hist_list: list[jax.Array] = []
    input_time_major = jnp.moveaxis(input_data, 1, 0)
    pre_synaptic = input_time_major

    for layer_weights in weights:
        n_neurons = layer_weights.shape[0]

        def layer_step(
            membrane: jax.Array,
            pre_t: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            membrane_potential = leak * membrane + (1.0 - leak) * (
                pre_t @ layer_weights.T
            )
            spikes = heaviside(membrane_potential, v_th)
            membrane_new = membrane_potential - v_th * spikes
            return membrane_new, (membrane_potential, spikes)

        _, (u_layer, s_layer) = lax.scan(
            layer_step,
            jnp.zeros((batch_size, n_neurons), dtype=input_data.dtype),
            pre_synaptic,
        )
        u_hist_list.append(u_layer)
        s_hist_list.append(s_layer)
        pre_synaptic = s_layer

    timesteps = input_data.shape[1]
    temporal_weights = decay ** jnp.arange(
        timesteps - 1,
        -1,
        -1,
        dtype=input_data.dtype,
    )
    weighted_spikes = jnp.einsum("t,tbo->bo", temporal_weights, s_hist_list[-1])
    logits = v_th * weighted_spikes / temporal_weights.sum()

    return ForwardCache(
        u_hist=tuple(u_hist_list),
        pre_syn=(input_time_major, *tuple(s_hist_list[:-1])),
        tw=temporal_weights,
        logits=logits,
    )


def softmax_loss(logits: jax.Array, label: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return softmax cross-entropy loss and output gradient."""
    losses, grads = jax_losses.loss_and_grad(
        logits[None, :],
        jnp.asarray([label], dtype=jnp.int32),
        loss="softmax",
    )
    return losses[0], grads[0]


def focal_loss(
    logits: jax.Array,
    label: jax.Array,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return focal loss and output gradient for one sample."""
    losses, grads = jax_losses.focal_loss_and_grad(
        logits[None, :],
        jnp.asarray([label], dtype=jnp.int32),
        focal_gamma=gamma,
        focal_alpha=alpha,
    )
    return losses[0], grads[0]


def output_spike_gradients(
    weights: tuple[jax.Array, ...],
    cache: ForwardCache,
    d_out: jax.Array,
    *,
    v_th: float,
) -> tuple[jax.Array, ...]:
    """Seed output-layer spike gradients from decay-weighted logits."""
    timesteps = cache.timesteps
    d_spikes = tuple(
        jnp.zeros((timesteps, layer.shape[0]), dtype=d_out.dtype) for layer in weights
    )
    output_grad = jnp.outer(cache.tw, d_out) * (v_th / cache.tw.sum())
    return (*d_spikes[:-1], output_grad)


def output_spike_gradients_batched(
    weights: tuple[jax.Array, ...],
    cache: ForwardCache,
    d_out: jax.Array,
    *,
    v_th: float,
) -> tuple[jax.Array, ...]:
    """Seed batched output-layer spike gradients from decay-weighted logits."""
    timesteps = cache.timesteps
    batch_size = d_out.shape[0]
    d_spikes = tuple(
        jnp.zeros((timesteps, batch_size, layer.shape[0]), dtype=d_out.dtype)
        for layer in weights
    )
    scale = v_th / cache.tw.sum()
    output_grad = jnp.einsum("t,bc->tbc", cache.tw, d_out) * scale
    return (*d_spikes[:-1], output_grad)


def backward_pass(
    weights: tuple[jax.Array, ...],
    d_spikes: tuple[jax.Array, ...],
    cache: ForwardCache,
    *,
    v_minus: float,
    v_plus: float,
    leak: float,
    v_th: float,
    alpha: float,
    beta: float,
) -> tuple[jax.Array, ...]:
    """Backpropagate through time with a surrogate spike gradient."""
    d_spikes_list = list(d_spikes)
    d_weights_list: list[jax.Array] = []

    for layer in range(len(weights) - 1, -1, -1):
        n_out = weights[layer].shape[0]
        u_rev = jnp.flip(cache.u_hist[layer], axis=0)
        pre_rev = jnp.flip(cache.pre_syn[layer], axis=0)
        ds_rev = jnp.flip(d_spikes_list[layer], axis=0)

        def rev_step(
            carry: tuple[jax.Array, jax.Array],
            inputs: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            d_u_next, d_w_acc = carry
            u_t, pre_t, ds_t = inputs
            d_u = ds_t * adali_surrogate(
                u_t,
                v_minus,
                v_plus,
                v_th=v_th,
                alpha=alpha,
                beta=beta,
            ) + leak * d_u_next
            d_w_acc = d_w_acc + jnp.outer(d_u, pre_t)
            return (d_u, d_w_acc), d_u

        (_, d_w_total), d_u_hist_rev = lax.scan(
            rev_step,
            (
                jnp.zeros(n_out, dtype=cache.logits.dtype),
                jnp.zeros_like(weights[layer]),
            ),
            (u_rev, pre_rev, ds_rev),
        )
        d_weights_list.insert(0, d_w_total)

        if layer > 0:
            d_u_hist = jnp.flip(d_u_hist_rev, axis=0)
            d_spikes_list[layer - 1] = d_spikes_list[layer - 1] + d_u_hist @ weights[layer]

    return tuple(d_weights_list)


def backward_pass_batched(
    weights: tuple[jax.Array, ...],
    d_spikes: tuple[jax.Array, ...],
    cache: ForwardCache,
    *,
    v_minus: float,
    v_plus: float,
    leak: float,
    v_th: float,
    alpha: float,
    beta: float,
) -> tuple[jax.Array, ...]:
    """Backpropagate through time for a batch and return mean weight gradients."""
    batch_size = int(d_spikes[-1].shape[1])
    d_spikes_list = list(d_spikes)
    d_weights_list: list[jax.Array] = []

    for layer in range(len(weights) - 1, -1, -1):
        n_out = weights[layer].shape[0]
        u_rev = jnp.flip(cache.u_hist[layer], axis=0)
        pre_rev = jnp.flip(cache.pre_syn[layer], axis=0)
        ds_rev = jnp.flip(d_spikes_list[layer], axis=0)

        def rev_step(
            carry: tuple[jax.Array, jax.Array],
            inputs: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            d_u_next, d_w_acc = carry
            u_t, pre_t, ds_t = inputs
            d_u = ds_t * adali_surrogate(
                u_t,
                v_minus,
                v_plus,
                v_th=v_th,
                alpha=alpha,
                beta=beta,
            ) + leak * d_u_next
            d_w_acc = d_w_acc + d_u.T @ pre_t
            return (d_u, d_w_acc), d_u

        (_, d_w_total), d_u_hist_rev = lax.scan(
            rev_step,
            (
                jnp.zeros((batch_size, n_out), dtype=cache.logits.dtype),
                jnp.zeros_like(weights[layer]),
            ),
            (u_rev, pre_rev, ds_rev),
        )
        d_weights_list.insert(0, d_w_total / batch_size)

        if layer > 0:
            d_u_hist = jnp.flip(d_u_hist_rev, axis=0)
            d_spikes_list[layer - 1] = d_spikes_list[layer - 1] + d_u_hist @ weights[layer]

    return tuple(d_weights_list)


def update_weights(
    weights: tuple[jax.Array, ...],
    gradient: tuple[jax.Array, ...],
    learning_rate: float,
) -> tuple[jax.Array, ...]:
    """Apply one SGD step to layer weights."""
    return tuple(
        layer_weights - learning_rate * grad
        for layer_weights, grad in zip(weights, gradient, strict=True)
    )


def _single_sample_grad(
    weights: tuple[jax.Array, ...],
    input_data: jax.Array,
    label: jax.Array,
    leak: float,
    v_th: float,
    decay: float,
    v_minus: float,
    v_plus: float,
    alpha: float,
    beta: float,
    focal_gamma: float,
    focal_alpha: float | None,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    cache = forward_pass(
        weights,
        input_data,
        leak=leak,
        v_th=v_th,
        decay=decay,
    )
    losses, d_out = jax_losses.focal_loss_and_grad(
        cache.logits[None, :],
        jnp.asarray([label], dtype=jnp.int32),
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    loss = losses[0]
    d_out = d_out[0]
    d_spikes = output_spike_gradients(weights, cache, d_out, v_th=v_th)
    grads = backward_pass(
        weights,
        d_spikes,
        cache,
        v_minus=v_minus,
        v_plus=v_plus,
        leak=leak,
        v_th=v_th,
        alpha=alpha,
        beta=beta,
    )
    return loss, grads


def _batch_grad(
    weights: tuple[jax.Array, ...],
    batch_x: jax.Array,
    batch_y: jax.Array,
    leak: float,
    v_th: float,
    decay: float,
    v_minus: float,
    v_plus: float,
    alpha: float,
    beta: float,
    focal_gamma: float,
    focal_alpha: float | None,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    cache = forward_pass_batched(
        weights,
        batch_x,
        leak=leak,
        v_th=v_th,
        decay=decay,
    )
    losses, d_out = jax_losses.focal_loss_and_grad(
        cache.logits,
        batch_y,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    d_spikes = output_spike_gradients_batched(weights, cache, d_out, v_th=v_th)
    grads = backward_pass_batched(
        weights,
        d_spikes,
        cache,
        v_minus=v_minus,
        v_plus=v_plus,
        leak=leak,
        v_th=v_th,
        alpha=alpha,
        beta=beta,
    )
    return losses, grads


@partial(jax.jit, static_argnames=_STATIC_TRAIN_ARGS)
def train_batch_step(
    weights: tuple[jax.Array, ...],
    batch_x: jax.Array,
    batch_y: jax.Array,
    *,
    leak: float,
    v_th: float,
    decay: float,
    v_minus: float,
    v_plus: float,
    alpha: float,
    beta: float,
    learning_rate: float,
    focal_gamma: float,
    focal_alpha: float | None,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Run one batched AdaLi training step and return updated weights and mean loss."""
    losses, grads = _batch_grad(
        weights,
        batch_x,
        batch_y,
        leak,
        v_th,
        decay,
        v_minus,
        v_plus,
        alpha,
        beta,
        focal_gamma,
        focal_alpha,
    )
    new_weights = update_weights(weights, grads, learning_rate)
    return new_weights, jnp.mean(losses)


@jax.jit
def forward_logits(
    weights: tuple[jax.Array, ...],
    input_data: jax.Array,
    *,
    leak: float,
    v_th: float,
    decay: float,
) -> jax.Array:
    """Return decay-weighted logits for one spike sample."""
    return forward_pass(
        weights,
        input_data,
        leak=leak,
        v_th=v_th,
        decay=decay,
    ).logits


def _batch_forward_logits_impl(
    weights: tuple[jax.Array, ...],
    batch_x: jax.Array,
    leak: float,
    v_th: float,
    decay: float,
) -> jax.Array:
    """Return logits for a batch of spike samples."""
    return forward_pass_batched(
        weights,
        batch_x,
        leak=leak,
        v_th=v_th,
        decay=decay,
    ).logits


batch_forward_logits = jax.jit(_batch_forward_logits_impl)
