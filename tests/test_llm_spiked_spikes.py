"""Tests for AdaLi surrogate spike VJP."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from spiking_neural_network.LLM_spiked.spikes import spike


def test_spike_forward_is_binary() -> None:
    u = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    s = spike(u, 1.0, 0.0, 2.0, 1.0, 1.0)
    assert jnp.all((s == 0.0) | (s == 1.0))
    assert float(s[3]) == 1.0
    assert float(s[2]) == 0.0


def test_spike_surrogate_grad_nonzero_near_threshold() -> None:
    def loss(u: jax.Array) -> jax.Array:
        return spike(u, 1.0, 0.0, 2.0, 1.0, 1.0).sum()

    u = jnp.array([0.5, 1.0, 1.5])
    g = jax.grad(loss)(u)
    assert float(jnp.abs(g).sum()) > 0.0
