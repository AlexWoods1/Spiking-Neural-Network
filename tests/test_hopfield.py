"""Tests for NumPy and JAX Hopfield network implementations."""

from __future__ import annotations

import numpy as np
import pytest

from spiking_neural_network.hopfield import (
    Grid,
    HopfieldNetwork,
    create_random_pattern,
    generate_empty_grid,
    sample_simulation,
)

pytest.importorskip("jax")
import jax.numpy as jnp

from spiking_neural_network.hopfield.hopfield_jax import (
    HopfieldNetworkJAX,
    bit_error_batch,
    recall_batch,
    recall_state,
    sample_simulation as jax_sample_simulation,
    store_weights,
)


@pytest.fixture
def square_patterns() -> list[np.ndarray]:
    """Three distinct 2x2 bipolar patterns."""
    return [
        np.array([1, -1, -1, -1]),
        np.array([1, 1, -1, -1]),
        np.array([-1, 1, 1, -1]),
    ]


@pytest.fixture
def seeded_random_patterns() -> list[np.ndarray]:
    """Ten unique random 4x4 patterns with a fixed seed."""
    rng = np.random.default_rng(0)
    patterns: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(patterns) < 10:
        pattern = np.where(rng.integers(0, 2, (4, 4)) == 1, 1, -1)
        key = tuple(pattern.ravel())
        if key not in seen:
            patterns.append(pattern)
            seen.add(key)
    return patterns


def _reference_weights(patterns: list[np.ndarray]) -> np.ndarray:
    pattern_matrix = np.stack(
        [np.where(pattern.ravel() >= 0, 1, -1) for pattern in patterns]
    )
    weights = (pattern_matrix.T @ pattern_matrix) / pattern_matrix.shape[1]
    np.fill_diagonal(weights, 0)
    return weights


class TestGrid:
    def test_generate_empty_grid_shape(self) -> None:
        grid = generate_empty_grid(4)
        assert grid.shape == (4, 4)
        assert np.all(grid == 0)

    def test_pattern_to_grid_resizes_flat_pattern(self) -> None:
        grid = Grid(size=2)
        grid.pattern_to_grid(np.array([1, -1, -1, 1]))
        np.testing.assert_array_equal(grid.grid, np.array([[1, -1], [-1, 1]]))


class TestHopfieldNetwork:
    def test_store_patterns_matches_reference(
        self, square_patterns: list[np.ndarray]
    ) -> None:
        network = HopfieldNetwork(square_patterns, max_iterations=30)
        network.store_patterns()

        np.testing.assert_allclose(network.weights, _reference_weights(square_patterns))
        assert np.all(np.diag(network.weights) == 0)

    def test_recall_single_stored_pattern(
        self, square_patterns: list[np.ndarray]
    ) -> None:
        network = HopfieldNetwork([square_patterns[0]], max_iterations=30)
        network.store_patterns()

        recalled = network.recall_pattern(square_patterns[0])
        np.testing.assert_array_equal(recalled, square_patterns[0])

    def test_recall_noisy_cue(self, square_patterns: list[np.ndarray]) -> None:
        network = HopfieldNetwork([square_patterns[0]], max_iterations=30)
        network.store_patterns()

        recalled = network.recall_pattern(np.array([0.9, -0.9, -0.9, -0.9]))
        np.testing.assert_array_equal(recalled, square_patterns[0])

    def test_recall_does_not_mutate_input(
        self, square_patterns: list[np.ndarray]
    ) -> None:
        network = HopfieldNetwork([square_patterns[0]], max_iterations=30)
        network.store_patterns()
        cue = np.array([0.9, -0.9, -0.9, -0.9])
        original = cue.copy()

        network.recall_pattern(cue)

        np.testing.assert_array_equal(cue, original)

    def test_recall_error_for_perfect_pattern_is_zero(
        self,
        square_patterns: list[np.ndarray],
    ) -> None:
        network = HopfieldNetwork([square_patterns[0]], max_iterations=30)
        network.store_patterns()

        assert network.recall_error_for_pattern(square_patterns[0]) == 0.0

    def test_recall_error_distribution_shape(
        self,
        seeded_random_patterns: list[np.ndarray],
    ) -> None:
        network = HopfieldNetwork(seeded_random_patterns, max_iterations=30)
        network.store_patterns()

        mean_error, std_error = network.recall_error_distribution()

        assert 0.0 <= mean_error <= 1.0
        assert std_error >= 0.0

    def test_mismatched_pattern_lengths_raise(self) -> None:
        patterns = [np.array([1, -1]), np.array([1, -1, 1, -1])]

        with pytest.raises(ValueError, match="same length"):
            HopfieldNetwork(patterns, max_iterations=30)

    def test_sample_simulation_returns_error_stats(
        self,
        seeded_random_patterns: list[np.ndarray],
    ) -> None:
        mean_error, std_error = sample_simulation(
            seeded_random_patterns, max_iterations=30
        )

        assert 0.0 <= mean_error <= 1.0
        assert std_error >= 0.0


class TestCreateRandomPattern:
    def test_returns_requested_count(self) -> None:
        np.random.seed(0)
        patterns = create_random_pattern(5, 3)

        assert len(patterns) == 5
        assert all(pattern.shape == (3, 3) for pattern in patterns)

    def test_patterns_are_unique(self) -> None:
        np.random.seed(1)
        patterns = create_random_pattern(8, 4)
        keys = {tuple(pattern.ravel()) for pattern in patterns}

        assert len(keys) == 8


class TestHopfieldNetworkJAX:
    def test_store_weights_matches_numpy(
        self,
        seeded_random_patterns: list[np.ndarray],
    ) -> None:
        numpy_network = HopfieldNetwork(seeded_random_patterns, max_iterations=30)
        numpy_network.store_patterns()
        jax_network = HopfieldNetworkJAX(seeded_random_patterns, max_iterations=30)

        np.testing.assert_allclose(
            numpy_network.weights,
            np.asarray(jax_network.weights),
            atol=1e-6,
        )

    def test_recall_matches_numpy(
        self,
        seeded_random_patterns: list[np.ndarray],
    ) -> None:
        numpy_network = HopfieldNetwork(seeded_random_patterns, max_iterations=30)
        numpy_network.store_patterns()
        jax_network = HopfieldNetworkJAX(seeded_random_patterns, max_iterations=30)

        for pattern in seeded_random_patterns[:3]:
            np.testing.assert_array_equal(
                numpy_network.recall_pattern(pattern),
                jax_network.recall_pattern(pattern),
            )

    def test_batch_recall_matches_single(
        self,
        seeded_random_patterns: list[np.ndarray],
    ) -> None:
        jax_network = HopfieldNetworkJAX(seeded_random_patterns, max_iterations=30)
        batch = np.asarray(
            recall_batch(jax_network.patterns, jax_network.weights, 30),
            dtype=np.int8,
        )
        single = jax_network.recall_patterns()

        np.testing.assert_array_equal(batch, single)

    def test_bit_error_batch_matches_manual(
        self,
        seeded_random_patterns: list[np.ndarray],
    ) -> None:
        jax_network = HopfieldNetworkJAX(seeded_random_patterns, max_iterations=30)
        batch_errors = np.asarray(
            bit_error_batch(jax_network.patterns, jax_network.weights, 30)
        )
        manual_errors = [
            jax_network.recall_error_for_pattern(pattern)
            for pattern in seeded_random_patterns
        ]

        np.testing.assert_allclose(batch_errors, manual_errors, atol=1e-6)

    def test_jax_sample_simulation_matches_numpy(
        self,
        seeded_random_patterns: list[np.ndarray],
    ) -> None:
        numpy_mean, numpy_std = sample_simulation(
            seeded_random_patterns, max_iterations=30
        )
        jax_mean, jax_std = jax_sample_simulation(
            seeded_random_patterns, max_iterations=30
        )

        assert jax_mean == pytest.approx(numpy_mean, abs=1e-5)
        assert jax_std == pytest.approx(numpy_std, abs=1e-5)

    def test_recall_state_is_jitted(
        self,
        square_patterns: list[np.ndarray],
    ) -> None:
        jax_network = HopfieldNetworkJAX([square_patterns[0]], max_iterations=30)
        cue = jnp.asarray(square_patterns[0], dtype=jnp.float32)
        recalled = recall_state(cue, jax_network.weights, 30)

        np.testing.assert_array_equal(
            np.asarray(recalled, dtype=np.int8),
            square_patterns[0],
        )

    def test_store_weights_is_jitted(
        self,
        square_patterns: list[np.ndarray],
    ) -> None:
        patterns = jnp.asarray(np.stack(square_patterns), dtype=jnp.float32)
        patterns = jnp.where(patterns >= 0, 1.0, -1.0)
        weights = store_weights(patterns)

        np.testing.assert_allclose(
            np.asarray(weights),
            _reference_weights(square_patterns),
            atol=1e-6,
        )

    def test_empty_pattern_list_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one pattern"):
            HopfieldNetworkJAX([], max_iterations=30)
