"""NumPy Hopfield network for associative memory.

The network stores bipolar patterns in a symmetric weight matrix and recalls them
by iterating the update rule until the state stabilizes:

    h_i = sum_j w_ij S_j
    S_i <- sgn(h_i)

Weight learning uses the Hebbian outer-product rule with zeroed diagonal:

    W = (1 / N) P^T P
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np


def generate_empty_grid(size: int) -> np.ndarray:
    """Return a square zero grid for visualization."""
    return np.zeros((size, size))


@dataclass
class Grid:
    """Square grid wrapper for rendering bipolar patterns."""

    size: int
    grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grid = generate_empty_grid(self.size)

    def pattern_to_grid(self, pattern: np.ndarray) -> None:
        """Map a flat or square pattern to a ``size x size`` bipolar grid."""
        self.grid = np.where(pattern == 1, 1, -1).reshape(self.size, self.size)

    def plot(self) -> None:
        """Display the current grid."""
        plt.imshow(self.grid, interpolation="nearest")
        plt.grid(which="minor", color="black", linestyle="-", linewidth=2)
        plt.axis("off")
        plt.title("Hopfield Network Grid")
        plt.xticks(np.arange(-0.5, self.size, 1), minor=True)
        plt.yticks(np.arange(-0.5, self.size, 1), minor=True)
        plt.tight_layout()
        plt.show()


class HopfieldNetwork:
    """Classical Hopfield network with synchronous recall updates."""

    def __init__(self, patterns: list[np.ndarray], max_iterations: int) -> None:
        """Build a network from stored patterns.

        Args:
            patterns: List of 1D or 2D bipolar-compatible pattern arrays.
            max_iterations: Maximum synchronous recall steps per query.
        """
        self.patterns = [np.where(pattern.ravel() >= 0, 1, -1) for pattern in patterns]
        pattern_length = len(self.patterns[0])
        if any(len(pattern) != pattern_length for pattern in self.patterns):
            raise ValueError("All patterns must have the same length.")
        self.neurons = pattern_length
        self.m = len(patterns)
        self.weights = np.zeros((self.neurons, self.neurons))
        self.neuron_values = np.zeros((self.neurons,))
        self.max_iterations = max_iterations

    def store_patterns(self) -> None:
        """Compute ``W = (1 / N) P^T P`` and zero the diagonal."""
        pattern_matrix = np.stack(self.patterns)
        self.weights = (pattern_matrix.T @ pattern_matrix) / self.neurons
        np.fill_diagonal(self.weights, 0)

    def recall_pattern(
        self,
        partial_pattern: np.ndarray,
        max_iterations: int | None = None,
    ) -> np.ndarray:
        """Recall a pattern from a partial or noisy cue.

        Args:
            partial_pattern: Initial neuron state cue.
            max_iterations: Optional override for the iteration cap.

        Returns:
            Bipolar recalled state in ``{-1, 1}``.
        """
        state = np.asarray(partial_pattern, dtype=np.float32).ravel().copy()
        state = np.where(state >= 0, 1, -1)
        iterations = self.max_iterations if max_iterations is None else max_iterations
        for _ in range(iterations):
            local = self.weights @ state
            new_state = np.sign(local)
            new_state[new_state == 0] = 1
            if np.all(new_state == state):
                break
            state = new_state
        self.neuron_values = state
        return state

    def visualize_recall(
        self,
        pattern: np.ndarray,
        recalled_pattern: np.ndarray,
        grid_size: int,
    ) -> None:
        """Plot original, recalled, and difference grids side by side."""
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        grids = []
        for arr in [
            pattern,
            recalled_pattern,
            np.where(pattern.ravel() == recalled_pattern.ravel(), 1, -1),
        ]:
            grid = Grid(size=grid_size)
            grid.pattern_to_grid(arr)
            grids.append(grid.grid)
        titles = ["Original Pattern", "Recalled Pattern", "Difference"]
        for axis, grid, title in zip(axes, grids, titles):
            axis.imshow(grid, interpolation="nearest")
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle("Hopfield Network Recall")
        fig.tight_layout()
        plt.show()

    def recall_error_for_pattern(self, pattern: np.ndarray) -> float:
        """Return the bit error rate for one pattern in ``[0, 1]``."""
        recalled_pattern = self.recall_pattern(pattern)
        agreement = np.mean(pattern.ravel() == recalled_pattern.ravel())
        return float(1.0 - agreement)

    def recall_error(self) -> float:
        """Return mean bit error rate across all stored patterns."""
        return float(
            np.mean(
                [self.recall_error_for_pattern(pattern) for pattern in self.patterns]
            )
        )

    def recall_error_distribution(self) -> tuple[float, float]:
        """Return mean and standard deviation of per-pattern bit error rates."""
        errors = [self.recall_error_for_pattern(pattern) for pattern in self.patterns]
        return float(np.mean(errors)), float(np.std(errors))


def create_random_pattern(size: int, dimension: int) -> list[np.ndarray]:
    """Sample unique random bipolar ``dimension x dimension`` patterns."""
    out: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(out) < size:
        pattern = np.where(np.random.randint(0, 2, (dimension, dimension)) == 1, 1, -1)
        pattern_tuple = tuple(pattern.flatten())
        if pattern_tuple not in seen:
            out.append(pattern)
            seen.add(pattern_tuple)
    return out


def visualize_recall(
    pattern: np.ndarray,
    recalled_pattern: np.ndarray,
    grid_size: int,
) -> None:
    """Plot original, recalled, and difference grids side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    grids = []
    for arr in [
        pattern,
        recalled_pattern,
        np.where(pattern.ravel() == recalled_pattern.ravel(), 1, -1),
    ]:
        grid = Grid(size=grid_size)
        grid.pattern_to_grid(arr)
        grids.append(grid.grid)
    titles = ["Original Pattern", "Recalled Pattern", "Difference"]
    for axis, grid, title in zip(axes, grids, titles):
        axis.imshow(grid, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")

    plt.title("Hopfield Network Recall")
    plt.tight_layout()
    plt.show()


def sample_simulation(
    patterns: list[np.ndarray],
    max_iterations: int,
) -> tuple[float, float]:
    """Train on ``patterns`` and return recall error statistics."""
    network = HopfieldNetwork(patterns=patterns, max_iterations=max_iterations)
    network.store_patterns()
    return network.recall_error_distribution()


def main() -> None:
    """Run a small capacity sweep and plot mean recall error."""
    max_iterations = 30
    results = []
    num_patterns = 100
    all_patterns = create_random_pattern(num_patterns, 16)
    for count in np.arange(10, num_patterns, int(num_patterns / 10)):
        print(
            f"Running simulation for {count} patterns with {max_iterations} iterations..."
        )
        mean, std = sample_simulation(all_patterns[:count], max_iterations)
        print(
            f"Number of patterns: {count}, Mean recall error: {mean:.4f}, "
            f"Standard deviation: {std:.4f}"
        )
        results.append((count, mean, std))
    plt.plot(
        [result[0] for result in results],
        [result[1] for result in results],
        label="Mean recall error",
    )
    plt.plot(
        [result[0] for result in results],
        [result[2] for result in results],
        label="Standard deviation",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
