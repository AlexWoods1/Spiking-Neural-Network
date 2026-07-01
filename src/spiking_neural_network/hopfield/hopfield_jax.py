"""JAX-accelerated Hopfield network.

This module mirrors the NumPy Hopfield API while compiling core operations with
``jax.jit`` and batching recall across patterns with ``jax.vmap``.

Update rule (bipolar states in {+1, -1}):

    h_i = sum_j w_ij S_j
    S_i <- sgn(h_i)

Weight learning:

    W = (1 / N) P^T P, with zero diagonal
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from spiking_neural_network.hopfield.hopfield import create_random_pattern


class HopfieldState(NamedTuple):
    """Immutable Hopfield network parameters stored on device.

    Attributes:
        weights: Symmetric weight matrix of shape ``(neurons, neurons)``.
        patterns: Stored bipolar patterns of shape ``(num_patterns, neurons)``.
        max_iterations: Upper bound on recall update steps.
    """

    weights: jnp.ndarray
    patterns: jnp.ndarray
    max_iterations: int


def _stack_bipolar_patterns(patterns: list[np.ndarray] | np.ndarray) -> jnp.ndarray:
    """Convert pattern inputs to a ``(num_patterns, neurons)`` bipolar JAX array."""
    if isinstance(patterns, list):
        flat = [np.asarray(pattern, dtype=np.float32).ravel() for pattern in patterns]
        if not flat:
            raise ValueError("At least one pattern is required.")
        neurons = len(flat[0])
        if any(len(row) != neurons for row in flat):
            raise ValueError("All patterns must have the same length.")
        array = jnp.asarray(np.stack(flat))
    else:
        array = jnp.asarray(patterns, dtype=jnp.float32)
        if array.ndim == 1:
            array = array[None, :]
        elif array.ndim == 3:
            array = array.reshape(array.shape[0], -1)
        elif array.ndim != 2:
            raise ValueError("Patterns must be 1D, 2D, or a stack of 2D grids.")

    return jnp.where(array >= 0, 1.0, -1.0)


@jax.jit
def store_weights(patterns: jnp.ndarray) -> jnp.ndarray:
    """Compute Hebbian weights for bipolar patterns.

    Args:
        patterns: Bipolar pattern matrix of shape ``(num_patterns, neurons)``.

    Returns:
        Weight matrix of shape ``(neurons, neurons)`` with a zero diagonal.
    """
    neurons = patterns.shape[1]
    weights = (patterns.T @ patterns) / neurons
    return weights.at[jnp.diag_indices(neurons)].set(0.0)


def _recall_step(state: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Apply one synchronous Hopfield update."""
    local_field = weights @ state
    new_state = jnp.sign(local_field)
    return jnp.where(new_state == 0, 1.0, new_state)


@partial(jax.jit, static_argnames=("max_iterations",))
def recall_state(
    state: jnp.ndarray,
    weights: jnp.ndarray,
    max_iterations: int,
) -> jnp.ndarray:
    """Recall a single pattern until convergence or ``max_iterations``.

    Args:
        state: Initial cue of shape ``(neurons,)``.
        weights: Weight matrix of shape ``(neurons, neurons)``.
        max_iterations: Maximum number of synchronous updates.

    Returns:
        Recalled bipolar state of shape ``(neurons,)``.
    """
    state = jnp.where(state >= 0, 1.0, -1.0)

    def cond_fn(carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        current, previous, step = carry
        return (step < max_iterations) & jnp.any(current != previous)

    def body_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        current, _, step = carry
        updated = _recall_step(current, weights)
        return updated, current, step + 1

    initial = (state, jnp.zeros_like(state), jnp.int32(0))
    final_state, _, _ = lax.while_loop(cond_fn, body_fn, initial)
    return final_state


@partial(jax.jit, static_argnames=("max_iterations",))
def recall_batch(
    states: jnp.ndarray,
    weights: jnp.ndarray,
    max_iterations: int,
) -> jnp.ndarray:
    """Recall a batch of cues in parallel.

    Args:
        states: Cues of shape ``(batch, neurons)``.
        weights: Weight matrix of shape ``(neurons, neurons)``.
        max_iterations: Maximum number of synchronous updates.

    Returns:
        Recalled states of shape ``(batch, neurons)``.
    """
    return jax.vmap(recall_state, in_axes=(0, None, None))(
        states, weights, max_iterations
    )


@partial(jax.jit, static_argnames=("max_iterations",))
def bit_error_batch(
    patterns: jnp.ndarray,
    weights: jnp.ndarray,
    max_iterations: int,
) -> jnp.ndarray:
    """Return per-pattern bit error rates for stored patterns.

    Args:
        patterns: Bipolar patterns of shape ``(num_patterns, neurons)``.
        weights: Weight matrix of shape ``(neurons, neurons)``.
        max_iterations: Maximum number of synchronous updates.

    Returns:
        Bit error rates of shape ``(num_patterns,)`` in ``[0, 1]``.
    """
    recalled = recall_batch(patterns, weights, max_iterations)
    return jnp.mean(recalled != patterns, axis=1)


@partial(jax.jit, static_argnames=("max_iterations",))
def recall_error_stats(
    patterns: jnp.ndarray,
    weights: jnp.ndarray,
    max_iterations: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return mean and standard deviation of per-pattern bit error rates."""
    errors = bit_error_batch(patterns, weights, max_iterations)
    return jnp.mean(errors), jnp.std(errors)


class HopfieldNetworkJAX:
    """JAX-backed Hopfield network with batched recall and JIT compilation."""

    def __init__(
        self, patterns: list[np.ndarray] | np.ndarray, max_iterations: int = 30
    ) -> None:
        self.patterns = _stack_bipolar_patterns(patterns)
        self.neurons = int(self.patterns.shape[1])
        self.max_iterations = max_iterations
        self.weights = store_weights(self.patterns)
        self._state = HopfieldState(
            weights=self.weights,
            patterns=self.patterns,
            max_iterations=max_iterations,
        )

    def store_patterns(self) -> None:
        """Recompute weights from stored patterns."""
        self.weights = store_weights(self.patterns)
        self._state = HopfieldState(
            weights=self.weights,
            patterns=self.patterns,
            max_iterations=self.max_iterations,
        )

    def recall_pattern(
        self,
        partial_pattern: np.ndarray,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        """Recall one pattern and return a NumPy bipolar vector."""
        iterations = self.max_iterations if max_iterations is None else max_iterations
        cue = jnp.asarray(partial_pattern, dtype=jnp.float32).ravel()
        recalled = recall_state(cue, self.weights, iterations)
        return np.asarray(recalled, dtype=np.int8)

    def recall_patterns(self, max_iterations: int | None = None) -> np.ndarray:
        """Recall all stored patterns in one batched JAX call."""
        iterations = self.max_iterations if max_iterations is None else max_iterations
        recalled = recall_batch(self.patterns, self.weights, iterations)
        return np.asarray(recalled, dtype=np.int8)

    def recall_error_for_pattern(self, pattern: np.ndarray) -> float:
        """Return bit error rate for one pattern in ``[0, 1]``."""
        recalled = self.recall_pattern(pattern)
        target = np.where(np.asarray(pattern).ravel() >= 0, 1, -1)
        return float(np.mean(recalled != target))

    def recall_error(self) -> float:
        """Return mean bit error rate across all stored patterns."""
        mean_error, _ = self.recall_error_distribution()
        return mean_error

    def recall_error_distribution(self) -> tuple[float, float]:
        """Return mean and standard deviation of per-pattern bit error rates."""
        mean_error, std_error = recall_error_stats(
            self.patterns,
            self.weights,
            self.max_iterations,
        )
        return float(mean_error), float(std_error)


def sample_simulation(
    patterns: list[np.ndarray] | np.ndarray,
    max_iterations: int = 30,
) -> tuple[float, float]:
    """Build a network and return recall error statistics."""
    network = HopfieldNetworkJAX(patterns=patterns, max_iterations=max_iterations)
    return network.recall_error_distribution()


def _benchmark(
    num_patterns: int, dimension: int, max_iterations: int, repeats: int
) -> None:
    """Compare NumPy and JAX implementations on the same random pattern set."""
    import time

    from spiking_neural_network.hopfield.hopfield import (
        HopfieldNetwork,
        sample_simulation as numpy_sample,
    )

    patterns = create_random_pattern(num_patterns, dimension)
    jax_network = HopfieldNetworkJAX(patterns, max_iterations=max_iterations)

    # * Warm up JIT compilation before timing.
    jax_network.recall_error_distribution()
    jax_network.recall_patterns()

    start = time.perf_counter()
    for _ in range(repeats):
        numpy_sample(patterns, max_iterations)
    numpy_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        jax_network.recall_error_distribution()
        jax_network.recall_patterns()
    jax_seconds = time.perf_counter() - start

    numpy_mean, _ = numpy_sample(patterns, max_iterations)
    jax_mean, _ = jax_network.recall_error_distribution()
    print(
        f"Patterns: {num_patterns}, grid: {dimension}x{dimension}, repeats: {repeats}"
    )
    print(f"NumPy mean error: {numpy_mean:.4f}, time: {numpy_seconds:.3f}s")
    print(f"JAX  mean error: {jax_mean:.4f}, time: {jax_seconds:.3f}s")
    print(f"Speedup: {numpy_seconds / jax_seconds:.2f}x")


def main() -> None:
    """Run a capacity sweep until mean recall error exceeds zero."""
    import time

    max_iterations = 60
    dimension = 128
    neurons = dimension**2
    expected_capacity = int(neurons / (2 * np.log(neurons)))
    max_patterns = max(expected_capacity * 2, 100)

    print(
        f"Expected capacity: ~{expected_capacity} patterns (N / (2 ln N)), N={neurons}"
    )
    print(f"Max patterns: {max_patterns}")
    all_patterns = create_random_pattern(max_patterns, dimension)

    print("JAX capacity sweep (until mean error > 0)")
    start = time.perf_counter()
    num_patterns = 500
    while num_patterns <= max_patterns:
        mean_error, std_error = sample_simulation(
            all_patterns[:num_patterns], max_iterations
        )
        print(
            f"patterns={num_patterns:4d}  "
            f"mean_error={mean_error:.6f}  std={std_error:.6f}"
        )
        if mean_error > 0:
            print(f"First non-zero error at {num_patterns} patterns.")
            break
        num_patterns += 10
    else:
        print(f"Mean error still 0 after {max_patterns} patterns.")

    elapsed = time.perf_counter() - start
    print(f"Capacity sweep time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
