"""Tests for Optax-backed JAX classification losses."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from spiking_neural_network.adali import jax_losses, jax_ops, numpy_ops
from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.config import AdaLiConfig
from tests.helpers import spike_batch


def test_sparse_softmax_cross_entropy_matches_numpy() -> None:
    logits = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    label = 1
    numpy_loss, _ = numpy_ops.softmax_loss(logits, label)
    jax_loss = float(
        jax_losses.sparse_softmax_cross_entropy(jnp.asarray(logits), jnp.int32(label))
    )
    assert jax_loss == pytest.approx(numpy_loss)


def test_softmax_loss_and_grad_matches_manual() -> None:
    logits = jnp.array([0.1, 0.2, 0.3])
    label = jnp.int32(1)
    loss, grad = jax_ops.softmax_loss(logits, label)

    log_probs = jax.nn.log_softmax(logits)
    expected_loss = -log_probs[label]
    probs = jax.nn.softmax(logits)
    expected_grad = probs.at[label].add(-1.0)

    assert float(loss) == pytest.approx(float(expected_loss))
    np.testing.assert_allclose(np.asarray(grad), np.asarray(expected_grad), rtol=1e-6)


def test_focal_loss_returns_finite_gradient() -> None:
    logits = jnp.array([0.1, 0.2, 0.3])
    label = jnp.int32(1)
    loss, grad = jax_ops.focal_loss(logits, label, gamma=2.0, alpha=0.25)

    assert float(loss) > 0.0
    assert np.all(np.isfinite(np.asarray(grad)))


def test_focal_loss_reduces_to_softmax_when_gamma_zero() -> None:
    logits = jnp.array([0.1, 0.2, 0.3])
    label = jnp.int32(1)
    softmax_loss, _ = jax_ops.softmax_loss(logits, label)
    focal_loss, _ = jax_ops.focal_loss(logits, label, gamma=0.0, alpha=None)
    assert float(focal_loss) == pytest.approx(float(softmax_loss))


def test_numpy_focal_loss_matches_jax() -> None:
    logits = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    label = 1
    numpy_loss, numpy_grad = numpy_ops.focal_loss(logits, label, gamma=2.0, alpha=0.25)
    jax_loss, jax_grad = jax_ops.focal_loss(
        jnp.asarray(logits),
        jnp.int32(label),
        gamma=2.0,
        alpha=0.25,
    )
    assert numpy_loss == pytest.approx(float(jax_loss))
    np.testing.assert_allclose(numpy_grad, np.asarray(jax_grad), rtol=1e-5)


def test_batched_focal_loss_shape() -> None:
    batch = jnp.asarray(spike_batch(4))
    model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3), backend="jax")
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
