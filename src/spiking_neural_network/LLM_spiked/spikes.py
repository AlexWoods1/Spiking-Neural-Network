"""AdaLi surrogate spike nonlinearity for Spiking LSTM gates."""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError("Install jax to use LLM_spiked spikes") from exc

from spiking_neural_network.adali.jax_ops import adali_surrogate


def _spike_fwd(
    u: jax.Array,
    v_th: float,
    v_minus: float,
    v_plus: float,
    alpha: float,
    beta: float,
) -> tuple[jax.Array, tuple[jax.Array, float, float, float, float, float]]:
    spikes = jnp.where(u >= v_th, 1.0, 0.0)
    return spikes, (u, v_th, v_minus, v_plus, alpha, beta)


def _spike_bwd(
    res: tuple[jax.Array, float, float, float, float, float],
    g: jax.Array,
) -> tuple[jax.Array, None, None, None, None, None]:
    u, v_th, v_minus, v_plus, alpha, beta = res
    # * AdaLi piecewise-linear surrogate through the threshold.
    sur = adali_surrogate(
        u,
        v_minus,
        v_plus,
        v_th=v_th,
        alpha=alpha,
        beta=beta,
    )
    return (g * sur, None, None, None, None, None)


@jax.custom_vjp
def spike(
    u: jax.Array,
    v_th: float,
    v_minus: float,
    v_plus: float,
    alpha: float,
    beta: float,
) -> jax.Array:
    """Hard threshold spike with AdaLi surrogate gradient.

    Forward pass is a Heaviside at ``v_th``. Backward pass multiplies the
    upstream gradient by the AdaLi surrogate slope.

    Args:
        u: Pre-spike membrane / gate pre-activations.
        v_th: Spike threshold.
        v_minus: Lower surrogate support bound.
        v_plus: Upper surrogate support bound.
        alpha: Surrogate slope scale below threshold.
        beta: Surrogate slope scale above threshold.

    Returns:
        Binary spike tensor with the same shape as ``u``.
    """
    spikes, _ = _spike_fwd(u, v_th, v_minus, v_plus, alpha, beta)
    return spikes


spike.defvjp(_spike_fwd, _spike_bwd)
