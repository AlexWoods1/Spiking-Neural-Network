"""AdaLi SNN model with NumPy or JAX training backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from spiking_neural_network.adali import numpy_ops
from spiking_neural_network.config import AdaLiConfig, SNN_Config
from spiking_neural_network.exceptions import ParameterError
from spiking_neural_network.schedules import (
    BoundaryState,
    EpochContext,
    EpochTrainingState,
    _linear_schedule,
)
from spiking_neural_network.seeds import MODEL_SEED_OFFSET, derived_seed
from spiking_neural_network.trainer import BaseModel

ForwardCache = numpy_ops.ForwardCache
Backend = Literal["numpy", "jax"]


@dataclass(frozen=True)
class SNNEpochTrainingState(EpochTrainingState):
    boundary: BoundaryState


class SNN_BaseModel(BaseModel, ABC):
    """AdaLi-compatible SNN with forward pass, softmax readout, and BPTT."""

    weights: list[np.ndarray]
    backend: Backend
    _weights_jax: tuple[Any, ...] | None

    def __init__(
        self,
        config: SNN_Config,
        weights: list[np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
        *,
        backend: Backend = "numpy",
    ) -> None:
        super().__init__(config)
        self.config = config
        self.backend = backend
        self.weights = weights if weights is not None else self._init_weights(rng)
        self._weights_jax = None
        if backend == "jax":
            self._init_jax_weights()

    @property
    def leak(self) -> float:
        """Membrane leak factor ``exp(-dt / tau)``."""
        return float(np.exp(-self.config.dt / self.config.tau))

    def learning_rate_at(self, ctx: EpochContext) -> float:
        learning_rate = self.config.learning_rate
        if callable(learning_rate):
            value = float(learning_rate(ctx))
        else:
            value = float(learning_rate)
        if value <= 0:
            raise ParameterError("learning_rate must be positive")
        return value

    def train_step(
        self,
        data: np.ndarray,
        label: int,
        *,
        ctx: EpochContext,
    ) -> float:
        state = self.resolve_epoch(ctx)
        return self.train_batch_step(
            np.expand_dims(data, axis=0),
            np.array([label]),
            state=state,
        )

    def train_batch_step(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        *,
        state: EpochTrainingState,
    ) -> float:
        if batch_x.ndim != 3:
            raise ValueError("batch_x must be a 3D array")
        batch_size = int(batch_x.shape[0])
        if batch_y.shape[0] != batch_size:
            raise ValueError("batch_x and batch_y must have matching batch size")
        if batch_size == 0:
            raise ValueError("batch_x must contain at least one sample")
        if not isinstance(state, SNNEpochTrainingState):
            raise TypeError("SNN models require SNNEpochTrainingState")

        boundary = state.boundary
        lr = state.learning_rate

        if self.backend == "jax":
            return self._train_batch_step_jax(batch_x, batch_y, boundary, lr)

        total_loss = 0.0
        gradient_sum = [np.zeros_like(weights) for weights in self.weights]
        for index in range(batch_size):
            loss, sample_gradient = self._training_gradients(
                batch_x[index],
                int(batch_y[index]),
                boundary,
            )
            total_loss += loss
            for accumulated, sample in zip(gradient_sum, sample_gradient, strict=True):
                accumulated += sample

        mean_gradient = [gradient / batch_size for gradient in gradient_sum]
        self.weights = numpy_ops.update_weights(self.weights, mean_gradient, lr)
        return total_loss / batch_size

    def _init_jax_weights(self) -> None:
        import jax.numpy as jnp

        self._weights_jax = tuple(jnp.asarray(weight) for weight in self.weights)

    def _jax_weights(self) -> tuple[Any, ...]:
        if self._weights_jax is None:
            self._init_jax_weights()
        assert self._weights_jax is not None
        return self._weights_jax

    def _train_batch_step_jax(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        boundary: BoundaryState,
        learning_rate: float,
    ) -> float:
        from spiking_neural_network.adali import jax_ops
        import jax.numpy as jnp

        adali_config = self._adali_config()
        weights_jax, loss = jax_ops.train_batch_step(
            self._jax_weights(),
            jnp.asarray(batch_x),
            jnp.asarray(batch_y, dtype=jnp.int32),
            leak=self.leak,
            v_th=self.config.v_th,
            decay=self.config.decay,
            v_minus=boundary.v_minus,
            v_plus=boundary.v_plus,
            alpha=adali_config.alpha,
            beta=adali_config.beta,
            learning_rate=learning_rate,
            focal_gamma=adali_config.focal_gamma,
            focal_alpha=adali_config.focal_alpha,
        )
        self._weights_jax = weights_jax
        return float(loss)

    def _training_gradients(
        self,
        data: np.ndarray,
        label: int,
        boundary: BoundaryState,
    ) -> tuple[float, list[np.ndarray]]:
        cache = self.forward(data)
        if isinstance(self.config, AdaLiConfig):
            loss, d_out = numpy_ops.focal_loss(
                cache.logits,
                label,
                gamma=self.config.focal_gamma,
                alpha=self.config.focal_alpha,
            )
        else:
            loss, d_out = numpy_ops.softmax_loss(cache.logits, label)
        d_spikes = numpy_ops.output_spike_gradients(
            self.weights,
            cache,
            d_out,
            v_th=self.config.v_th,
        )
        d_weights = numpy_ops.backward_pass(
            self.weights,
            d_spikes,
            cache,
            boundary,
            leak=self.leak,
            surrogate=self._surrogate,
        )
        return loss, d_weights

    def forward(self, input_data: np.ndarray) -> ForwardCache:
        """Validate input shape and run the hard-reset LIF forward pass."""
        if not self.weights:
            raise ValueError("weights must contain at least one layer")
        if input_data.ndim != 2:
            raise ValueError("input_data must be a 2D array")

        if self.backend == "jax":
            return self._forward_jax(input_data)

        return numpy_ops.forward_pass(
            self.weights,
            input_data,
            leak=self.leak,
            v_th=self.config.v_th,
            decay=self.config.decay,
        )

    def _forward_jax(self, input_data: np.ndarray) -> ForwardCache:
        from spiking_neural_network.adali import jax_ops
        import jax.numpy as jnp

        jax_cache = jax_ops.forward_pass(
            self._jax_weights(),
            jnp.asarray(input_data),
            leak=self.leak,
            v_th=self.config.v_th,
            decay=self.config.decay,
        )
        u_hist = [
            [np.asarray(layer_u[t]) for t in range(layer_u.shape[0])]
            for layer_u in (np.asarray(layer) for layer in jax_cache.u_hist)
        ]
        return numpy_ops.ForwardCache(
            u_hist=u_hist,
            pre_syn=[np.asarray(pre) for pre in jax_cache.pre_syn],
            tw=np.asarray(jax_cache.tw),
            logits=np.asarray(jax_cache.logits),
        )

    @abstractmethod
    def _surrogate(self, u: np.ndarray, v_minus: float, v_plus: float) -> np.ndarray:
        """Return surrogate gradient for one membrane snapshot."""

    def _adali_config(self) -> AdaLiConfig:
        if not isinstance(self.config, AdaLiConfig):
            raise TypeError("JAX backend requires AdaLiConfig")
        return self.config

    def _init_weights(self, rng: np.random.Generator | None = None) -> list[np.ndarray]:
        if rng is None:
            rng = np.random.default_rng(
                derived_seed(self.config.seed, MODEL_SEED_OFFSET)
            )
        return numpy_ops.init_weights(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            weight_scale=self.config.weight_scale,
            rng=rng,
        )

    def predict(self, data: np.ndarray) -> int:
        return int(np.argmax(self.forward(data).logits))

    def predict_batch(self, batch_x: np.ndarray) -> np.ndarray:
        """Return predicted class indices for a batch of spike trains."""
        if batch_x.ndim != 3:
            raise ValueError("batch_x must be a 3D array")

        if self.backend == "jax":
            from spiking_neural_network.adali import jax_ops
            import jax.numpy as jnp

            logits = jax_ops.batch_forward_logits(
                self._jax_weights(),
                jnp.asarray(batch_x),
                leak=self.leak,
                v_th=self.config.v_th,
                decay=self.config.decay,
            )
            return np.asarray(jnp.argmax(logits, axis=1), dtype=int)

        return np.array([self.predict(sample) for sample in batch_x], dtype=int)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Return softmax class probabilities for one spike train."""
        from spiking_neural_network.evaluation import softmax

        return softmax(np.asarray(self.forward(data).logits))

    def predict_proba_batch(self, batch_x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for a batch of spike trains."""
        if batch_x.ndim != 3:
            raise ValueError("batch_x must be a 3D array")

        if self.backend == "jax":
            from spiking_neural_network.adali import jax_ops
            import jax.numpy as jnp

            logits = np.asarray(
                jax_ops.batch_forward_logits(
                    self._jax_weights(),
                    jnp.asarray(batch_x),
                    leak=self.leak,
                    v_th=self.config.v_th,
                    decay=self.config.decay,
                )
            )
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            return probabilities / probabilities.sum(axis=1, keepdims=True)

        return np.stack([self.predict_proba(sample) for sample in batch_x])

    @abstractmethod
    def boundaries(self, ctx: EpochContext) -> BoundaryState:
        """Return surrogate boundaries for the given training epoch."""

    def resolve_epoch(self, ctx: EpochContext) -> SNNEpochTrainingState:
        return SNNEpochTrainingState(
            ctx=ctx,
            learning_rate=self.learning_rate_at(ctx),
            boundary=self.boundaries(ctx),
        )


class AdaLi(SNN_BaseModel):
    """AdaLi classifier with linearly shrinking surrogate boundaries."""

    config: AdaLiConfig

    def __init__(
        self,
        config: AdaLiConfig,
        weights: list[np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
        *,
        backend: Backend = "numpy",
    ) -> None:
        super().__init__(config, weights, rng, backend=backend)

    def boundaries(self, ctx: EpochContext) -> BoundaryState:
        v_minus, v_plus = self._update_boundaries(ctx)
        return BoundaryState(v_minus, v_plus)

    def _update_boundaries(self, ctx: EpochContext) -> tuple[float, float]:
        left = _linear_schedule(
            ctx,
            self.config.left_initial,
            self.config.p * self.config.left_initial,
        )
        right = _linear_schedule(
            ctx,
            self.config.right_initial,
            self.config.p * self.config.right_initial,
        )
        return self.config.v_th - left, self.config.v_th + right

    def _surrogate(self, u: np.ndarray, v_minus: float, v_plus: float) -> np.ndarray:
        return numpy_ops.adali_surrogate(
            u,
            v_minus,
            v_plus,
            v_th=self.config.v_th,
            alpha=self.config.alpha,
            beta=self.config.beta,
        )
