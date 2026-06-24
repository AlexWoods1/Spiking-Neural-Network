"""Tests for the AdaLi model and NumPy training ops."""

from __future__ import annotations

import numpy as np
import pytest

from spiking_neural_network.adali import numpy_ops
from spiking_neural_network.adali.model import AdaLi, SNN_BaseModel, SNNEpochTrainingState
from spiking_neural_network.builder import ModelBuilder
from spiking_neural_network.config import AdaLiConfig, SNN_Config
from spiking_neural_network.exceptions import ParameterError
from spiking_neural_network.schedules import BoundaryState, EpochContext, EpochTrainingState, linear_learning_rate
from tests.helpers import spike_batch


class TestNumpyOps:
    def test_forward_pass_logits_shape(self) -> None:
        rng = np.random.default_rng(0)
        weights = numpy_ops.init_weights(
            input_dim=784,
            hidden_dims=(8,),
            output_dim=3,
            weight_scale=0.2,
            rng=rng,
        )
        sample = spike_batch(1, t_steps=5)[0]

        cache = numpy_ops.forward_pass(
            weights,
            sample,
            leak=float(np.exp(-0.5)),
            v_th=1.0,
            decay=0.9,
        )

        assert cache.logits.shape == (3,)
        assert cache.timesteps == 5

    def test_softmax_loss_returns_finite_gradient(self) -> None:
        logits = np.array([0.1, 0.2, 0.3])
        loss, d_out = numpy_ops.softmax_loss(logits, label=1)

        assert np.isfinite(loss)
        assert d_out.shape == (3,)
        assert d_out.sum() == pytest.approx(0.0)

    def test_adali_surrogate_is_zero_outside_window(self) -> None:
        u = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        slope = numpy_ops.adali_surrogate(
            u,
            v_minus=0.5,
            v_plus=1.5,
            v_th=1.0,
            alpha=0.5,
            beta=0.5,
        )

        assert slope[0] == 0.0
        assert slope[-1] == 0.0
        assert np.all(slope[1:4] > 0.0)

    def test_model_forward_matches_numpy_ops(self) -> None:
        model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3))
        sample = spike_batch(1, t_steps=5)[0]

        model_cache = model.forward(sample)
        ops_cache = numpy_ops.forward_pass(
            model.weights,
            sample,
            leak=model.leak,
            v_th=model.config.v_th,
            decay=model.config.decay,
        )

        np.testing.assert_allclose(model_cache.logits, ops_cache.logits)


class TestAdaLiModel:
    def test_forward_logits_shape(self) -> None:
        model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3))
        sample = spike_batch(1, t_steps=5, features=784)[0]

        cache = model.forward(sample)

        assert cache.logits.shape == (3,)
        assert cache.timesteps == 5

    def test_train_step_updates_weights(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05))
        sample = spike_batch(1)[0]
        before = [weights.copy() for weights in model.weights]

        loss = model.train_step(sample, label=1, ctx=EpochContext(1, 3))

        assert np.isfinite(loss)
        assert not np.allclose(model.weights[0], before[0])

    def test_train_batch_step_averages_gradients(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05))
        batch_x = spike_batch(4)
        batch_y = np.array([1, 2, 3, 4], dtype=int)
        before = [weights.copy() for weights in model.weights]
        state = model.resolve_epoch(EpochContext(1, 3))

        loss = model.train_batch_step(batch_x, batch_y, state=state)

        assert np.isfinite(loss)
        assert not np.allclose(model.weights[0], before[0])

    def test_train_batch_step_rejects_shape_mismatch(self) -> None:
        model = AdaLi(AdaLiConfig())
        batch_x = spike_batch(4)
        state = model.resolve_epoch(EpochContext(1, 1))
        with pytest.raises(ValueError, match="matching batch size"):
            model.train_batch_step(batch_x, np.array([0, 1]), state=state)

    def test_boundaries_shrink_over_epochs(self) -> None:
        model = AdaLi(AdaLiConfig())
        early = model.boundaries(EpochContext(1, 5))
        late = model.boundaries(EpochContext(5, 5))

        assert late.v_minus > early.v_minus
        assert late.v_plus < early.v_plus

    def test_learning_rate_at_supports_schedule(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=linear_learning_rate(0.08, 0.02)))

        assert model.learning_rate_at(EpochContext(1, 4)) == pytest.approx(0.08)
        assert model.learning_rate_at(EpochContext(4, 4)) == pytest.approx(0.02)

    def test_forward_rejects_invalid_input(self) -> None:
        model = AdaLi(AdaLiConfig())
        with pytest.raises(ValueError, match="2D array"):
            model.forward(np.zeros((4, 28, 28)))

    def test_predict_returns_int_label(self) -> None:
        model = AdaLi(AdaLiConfig())
        sample = spike_batch(1)[0]
        assert isinstance(model.predict(sample), int)

    def test_model_builder_returns_adali_instance(self) -> None:
        model = ModelBuilder.build("adali", AdaLiConfig())
        assert isinstance(model, AdaLi)

    def test_train_batch_step_rejects_invalid_batch_rank(self) -> None:
        model = AdaLi(AdaLiConfig())
        state = model.resolve_epoch(EpochContext(1, 1))
        with pytest.raises(ValueError, match="3D array"):
            model.train_batch_step(np.zeros((4, 784)), np.array([0]), state=state)

    def test_train_batch_step_rejects_empty_batch(self) -> None:
        model = AdaLi(AdaLiConfig())
        state = model.resolve_epoch(EpochContext(1, 1))
        with pytest.raises(ValueError, match="at least one sample"):
            model.train_batch_step(np.zeros((0, 4, 784)), np.array([], dtype=int), state=state)

    def test_train_batch_step_rejects_non_snn_state(self) -> None:
        model = AdaLi(AdaLiConfig())
        batch_x = spike_batch(2)
        state = EpochTrainingState(ctx=EpochContext(1, 1), learning_rate=0.01)
        with pytest.raises(TypeError, match="SNNEpochTrainingState"):
            model.train_batch_step(batch_x, np.array([0, 1]), state=state)

    def test_learning_rate_at_rejects_non_positive_runtime_value(self) -> None:
        def sneaky_schedule(ctx: EpochContext) -> float:
            return -1.0 if ctx.epoch == 2 else 0.01

        model = AdaLi(AdaLiConfig(learning_rate=sneaky_schedule))
        with pytest.raises(ParameterError, match="learning_rate must be positive"):
            model.learning_rate_at(EpochContext(2, 2))

    def test_forward_rejects_empty_weights(self) -> None:
        model = AdaLi(AdaLiConfig())
        model.weights = []
        with pytest.raises(ValueError, match="at least one layer"):
            model.forward(spike_batch(1)[0])

    def test_resolve_epoch_returns_snn_state(self) -> None:
        model = AdaLi(AdaLiConfig())
        state = model.resolve_epoch(EpochContext(1, 3))
        assert isinstance(state, SNNEpochTrainingState)


class TestAdaLiJaxBackend:
    def test_forward_matches_numpy_backend(self) -> None:
        config = AdaLiConfig(hidden_dims=(8,), output_dim=3)
        numpy_model = AdaLi(config, backend="numpy")
        jax_model = AdaLi(config, weights=[w.copy() for w in numpy_model.weights], backend="jax")
        sample = spike_batch(1, t_steps=5)[0]

        np.testing.assert_allclose(
            jax_model.forward(sample).logits,
            numpy_model.forward(sample).logits,
            rtol=1e-5,
        )

    def test_train_batch_step_updates_weights(self) -> None:
        model = AdaLi(AdaLiConfig(learning_rate=0.05, hidden_dims=(8,), output_dim=3), backend="jax")
        batch_x = spike_batch(4)
        batch_y = np.array([1, 2, 0, 1], dtype=int)
        import jax.numpy as jnp

        before = tuple(np.asarray(weight) for weight in model._jax_weights())
        state = model.resolve_epoch(EpochContext(1, 3))

        loss = model.train_batch_step(batch_x, batch_y, state=state)

        assert np.isfinite(loss)
        after = tuple(np.asarray(weight) for weight in model._jax_weights())
        assert not np.allclose(after[0], before[0])

    def test_predict_batch_matches_single_predictions(self) -> None:
        model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3), backend="jax")
        batch = spike_batch(4)

        batch_predictions = model.predict_batch(batch)
        single_predictions = np.array([model.predict(sample) for sample in batch])

        np.testing.assert_array_equal(batch_predictions, single_predictions)

    def test_predict_proba_batch_matches_single_predictions(self) -> None:
        model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3), backend="jax")
        batch = spike_batch(3)

        batch_probabilities = model.predict_proba_batch(batch)
        single_probabilities = np.stack([model.predict_proba(sample) for sample in batch])

        np.testing.assert_allclose(batch_probabilities, single_probabilities, rtol=1e-6)

    def test_model_builder_passes_jax_backend(self) -> None:
        model = ModelBuilder.build("adali", AdaLiConfig(), backend="jax")
        assert model.backend == "jax"

    def test_jax_batch_step_rejects_non_adali_config(self) -> None:
        class _GenericSnnModel(SNN_BaseModel):
            def _surrogate(
                self,
                u: np.ndarray,
                v_minus: float,
                v_plus: float,
            ) -> np.ndarray:
                return np.zeros_like(u)

            def boundaries(self, ctx: EpochContext) -> BoundaryState:
                return BoundaryState(v_minus=0.5, v_plus=1.5)

        model = _GenericSnnModel(SNN_Config(hidden_dims=(8,), output_dim=3), backend="jax")
        state = model.resolve_epoch(EpochContext(1, 1))
        batch_x = spike_batch(2)

        with pytest.raises(TypeError, match="JAX backend requires AdaLiConfig"):
            model._train_batch_step_jax(
                batch_x,
                np.array([0, 1], dtype=int),
                state.boundary,
                learning_rate=0.01,
            )
