"""Tests for Optax-backed JAX classification losses."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from spiking_neural_network.adali import jax_losses, jax_ops
from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.config import AdaLiConfig
from tests.helpers import spike_batch


def test_sparse_softmax_cross_entropy_matches_manual() -> None:
    logits = jnp.array([0.1, 0.2, 0.3])
    label = jnp.int32(1)
    jax_loss = float(jax_losses.sparse_softmax_cross_entropy(logits, label))
    log_probs = jax.nn.log_softmax(logits)
    expected_loss = -log_probs[label]
    assert jax_loss == pytest.approx(float(expected_loss))


def test_focal_loss_and_grad_matches_manual_softmax_grad_at_gamma_zero() -> None:
    logits = jnp.array([0.1, 0.2, 0.3])
    labels = jnp.array([1], dtype=jnp.int32)
    losses, grads = jax_losses.focal_loss_and_grad(
        logits[None, :],
        labels,
        focal_gamma=0.0,
        focal_alpha=None,
    )
    probs = jax.nn.softmax(logits)
    expected_grad = probs.at[1].add(-1.0)

    assert float(losses[0]) > 0.0
    np.testing.assert_allclose(
        np.asarray(grads[0]), np.asarray(expected_grad), rtol=1e-6
    )


def test_focal_loss_returns_finite_gradient() -> None:
    logits = jnp.array([[0.1, 0.2, 0.3]])
    labels = jnp.array([1], dtype=jnp.int32)
    losses, grads = jax_losses.focal_loss_and_grad(
        logits,
        labels,
        focal_gamma=2.0,
        focal_alpha=0.25,
    )

    assert float(losses[0]) > 0.0
    assert np.all(np.isfinite(np.asarray(grads)))


def test_batched_focal_loss_shape() -> None:
    batch = jnp.asarray(spike_batch(4))
    model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3))
    weights = tuple(jnp.asarray(weight) for weight in model.weights)
    logits = jax_ops.batch_forward_logits(
        weights,
        batch,
        model.leak,
        model.config.v_th,
        model.config.decay,
    )
    labels = jnp.array([0, 1, 2, 1], dtype=jnp.int32)

    losses, grads = jax_losses.focal_loss_and_grad(
        logits,
        labels,
        focal_gamma=2.0,
        focal_alpha=0.5,
    )

    assert losses.shape == (4,)
    assert grads.shape == (4, 3)
