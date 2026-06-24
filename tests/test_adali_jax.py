"""Tests for the JAX AdaLi training ops."""

from __future__ import annotations

import builtins
import importlib
import sys
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from spiking_neural_network.adali import jax_ops
from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.adali.weights import init_weights
from spiking_neural_network.config import AdaLiConfig
from spiking_neural_network.schedules import EpochContext
from tests.helpers import spike_batch


def _numpy_weights_to_jax(weights: list[np.ndarray]) -> tuple[jnp.ndarray, ...]:
    return tuple(jnp.asarray(weight) for weight in weights)


class TestJaxOps:
    def test_forward_pass_returns_expected_logits_shape(self) -> None:
        rng = np.random.default_rng(0)
        weights = init_weights(
            input_dim=784,
            hidden_dims=(8,),
            output_dim=3,
            weight_scale=0.2,
            rng=rng,
        )
        sample = spike_batch(1, t_steps=5)[0]
        leak = float(np.exp(-0.5))
        v_th = 1.0
        decay = 0.9

        jax_cache = jax_ops.forward_pass(
            _numpy_weights_to_jax(weights),
            jnp.asarray(sample),
            leak=leak,
            v_th=v_th,
            decay=decay,
        )

        assert np.asarray(jax_cache.logits).shape == (3,)
        assert jax_cache.timesteps == 5

    def test_forward_logits_matches_forward_pass(self) -> None:
        rng = np.random.default_rng(0)
        weights = init_weights(
            input_dim=784,
            hidden_dims=(8,),
            output_dim=3,
            weight_scale=0.2,
            rng=rng,
        )
        sample = spike_batch(1, t_steps=5)[0]
        leak = float(np.exp(-0.5))
        v_th = 1.0
        decay = 0.9
        weights_jax = _numpy_weights_to_jax(weights)

        cache_logits = np.asarray(
            jax_ops.forward_pass(
                weights_jax,
                jnp.asarray(sample),
                leak=leak,
                v_th=v_th,
                decay=decay,
            ).logits
        )
        logits = np.asarray(
            jax_ops.forward_logits(
                weights_jax,
                jnp.asarray(sample),
                leak=leak,
                v_th=v_th,
                decay=decay,
            )
        )

        np.testing.assert_allclose(logits, cache_logits, rtol=1e-6)

    def test_batch_forward_logits_matches_single_sample(self) -> None:
        rng = np.random.default_rng(1)
        weights = init_weights(
            input_dim=784,
            hidden_dims=(8,),
            output_dim=3,
            weight_scale=0.2,
            rng=rng,
        )
        batch = spike_batch(4, t_steps=5)
        weights_jax = _numpy_weights_to_jax(weights)
        leak = float(np.exp(-0.5))
        v_th = 1.0
        decay = 0.9

        batch_logits = np.asarray(
            jax_ops.batch_forward_logits(
                weights_jax,
                jnp.asarray(batch),
                leak,
                v_th,
                decay,
            )
        )
        single_logits = np.stack(
            [
                np.asarray(
                    jax_ops.forward_logits(
                        weights_jax,
                        jnp.asarray(sample),
                        leak=leak,
                        v_th=v_th,
                        decay=decay,
                    )
                )
                for sample in batch
            ]
        )

        np.testing.assert_allclose(batch_logits, single_logits, rtol=1e-6)

    def test_train_batch_step_updates_weights(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05, hidden_dims=(8,), output_dim=3))
        boundary = model.boundaries(EpochContext(1, 3))
        weights = _numpy_weights_to_jax(model.weights)
        batch_x = jnp.asarray(spike_batch(4))
        batch_y = jnp.array([1, 2, 0, 1], dtype=jnp.int32)
        before = tuple(layer.copy() for layer in weights)

        new_weights, loss = jax_ops.train_batch_step(
            weights,
            batch_x,
            batch_y,
            leak=model.leak,
            v_th=model.config.v_th,
            decay=model.config.decay,
            v_minus=boundary.v_minus,
            v_plus=boundary.v_plus,
            alpha=model.config.alpha,
            beta=model.config.beta,
            learning_rate=0.05,
            focal_gamma=model.config.focal_gamma,
            focal_alpha=model.config.focal_alpha,
        )

        assert float(loss) > 0.0
        assert not np.allclose(np.asarray(new_weights[0]), np.asarray(before[0]))

    def test_batched_forward_matches_single_sample(self) -> None:
        rng = np.random.default_rng(2)
        weights = init_weights(
            input_dim=784,
            hidden_dims=(8,),
            output_dim=3,
            weight_scale=0.2,
            rng=rng,
        )
        batch = spike_batch(4, t_steps=5)
        weights_jax = _numpy_weights_to_jax(weights)
        leak = float(np.exp(-0.5))
        v_th = 1.0
        decay = 0.9

        batch_logits = np.asarray(
            jax_ops.forward_pass_batched(
                weights_jax,
                jnp.asarray(batch),
                leak=leak,
                v_th=v_th,
                decay=decay,
            ).logits
        )
        single_logits = np.stack(
            [
                np.asarray(
                    jax_ops.forward_pass(
                        weights_jax,
                        jnp.asarray(sample),
                        leak=leak,
                        v_th=v_th,
                        decay=decay,
                    ).logits
                )
                for sample in batch
            ]
        )

        np.testing.assert_allclose(batch_logits, single_logits, rtol=1e-6)

    def test_batched_grad_mean_matches_single_sample_grads(self) -> None:
        model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3))
        boundary = model.boundaries(EpochContext(1, 3))
        batch = spike_batch(4, t_steps=5)
        batch_y = jnp.array([1, 2, 0, 1], dtype=jnp.int32)
        weights = _numpy_weights_to_jax(model.weights)

        _, batched_grads = jax_ops._batch_grad(
            weights,
            jnp.asarray(batch),
            batch_y,
            model.leak,
            model.config.v_th,
            model.config.decay,
            boundary.v_minus,
            boundary.v_plus,
            model.config.alpha,
            model.config.beta,
            model.config.focal_gamma,
            model.config.focal_alpha,
        )
        single_grads = []
        for sample, label in zip(batch, np.asarray(batch_y), strict=True):
            _, grads = jax_ops._single_sample_grad(
                weights,
                jnp.asarray(sample),
                jnp.int32(label),
                model.leak,
                model.config.v_th,
                model.config.decay,
                boundary.v_minus,
                boundary.v_plus,
                model.config.alpha,
                model.config.beta,
                model.config.focal_gamma,
                model.config.focal_alpha,
            )
            single_grads.append(grads)
        mean_single_grads = tuple(
            jnp.mean(
                jnp.stack([grad[layer_index] for grad in single_grads], axis=0), axis=0
            )
            for layer_index in range(len(weights))
        )

        for batched_grad, mean_single_grad in zip(
            batched_grads, mean_single_grads, strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(batched_grad), np.asarray(mean_single_grad), rtol=1e-5
            )


def test_jax_ops_import_error_when_jax_missing() -> None:
    module_name = "spiking_neural_network.adali.jax_ops"
    saved_module = sys.modules.pop(module_name, None)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("no jax")
        return real_import(name, globals, locals, fromlist, level)

    try:
        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(
                ImportError, match="Install jax and optax to use the JAX backend"
            ):
                importlib.import_module(module_name)
    finally:
        if saved_module is not None:
            sys.modules[module_name] = saved_module
        else:
            importlib.import_module(module_name)
