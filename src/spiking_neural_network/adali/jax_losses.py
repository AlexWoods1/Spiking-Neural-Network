"""Optax-backed classification losses with per-sample logits gradients."""

from __future__ import annotations

from typing import Literal

try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError as exc:
    raise ImportError("Install jax and optax to use JAX losses") from exc

LossName = Literal["softmax", "focal"]


def sparse_softmax_cross_entropy(logits: jax.Array, label: jax.Array) -> jax.Array:
    """Return sparse softmax cross-entropy for one logit vector."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, label)


def sparse_focal_loss(
    logits: jax.Array,
    label: jax.Array,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> jax.Array:
    """Return multiclass focal loss for one logit vector."""
    cross_entropy = sparse_softmax_cross_entropy(logits, label)
    true_class_probability = jnp.exp(-cross_entropy)
    focal_weight = (1.0 - true_class_probability) ** gamma
    loss = focal_weight * cross_entropy
    if alpha is not None:
        loss = alpha * loss
    return loss


def _loss_and_grad(
    logits: jax.Array,
    label: jax.Array,
    *,
    loss: LossName,
    focal_gamma: float,
    focal_alpha: float | None,
) -> tuple[jax.Array, jax.Array]:
    def scalar_loss(logits_row: jax.Array) -> jax.Array:
        if loss == "softmax":
            return sparse_softmax_cross_entropy(logits_row, label)
        if loss == "focal":
            return sparse_focal_loss(
                logits_row,
                label,
                gamma=focal_gamma,
                alpha=focal_alpha,
            )
        raise ValueError(f"Unsupported loss: {loss}")

    return jax.value_and_grad(scalar_loss)(logits)


def loss_and_grad(
    logits: jax.Array,
    labels: jax.Array,
    *,
    loss: LossName = "focal",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return per-sample losses and ``d(loss)/d(logits)`` for a batch."""
    return jax.vmap(
        lambda logits_row, label: _loss_and_grad(
            logits_row,
            label,
            loss=loss,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        ),
        in_axes=(0, 0),
    )(logits, labels)


def focal_loss_and_grad(
    logits: jax.Array,
    labels: jax.Array,
    *,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return per-sample focal losses and logits gradients."""
    return loss_and_grad(
        logits,
        labels,
        loss="focal",
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
