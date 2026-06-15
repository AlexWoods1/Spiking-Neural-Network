from dataclasses import dataclass

import numpy as np


def _poisson_samples(
    distribution: np.ndarray, t: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return rng.poisson(distribution, size=(t,) + distribution.shape)


def _first_spike(samples: np.ndarray) -> np.ndarray:
    mask = np.cumsum(samples, axis=0) > 0
    first = np.argmax(mask, axis=0)
    first[~mask.any(axis=0)] = -1
    return first


class EncodingError(Exception):
    """Raised when spike encoding inputs or indices are invalid."""


def _validate_rates(rates: np.ndarray) -> None:
    if rates.ndim != 2:
        raise EncodingError(f"Rates must be 2D: {rates.shape}")
    if rates.shape[0] == 0 or rates.shape[1] == 0:
        raise EncodingError(f"Rates must have non-zero dimensions: {rates.shape}")
    if not np.all(np.isfinite(rates)):
        raise EncodingError("Rates contain non-finite values")
    if np.any(rates < 0):
        raise EncodingError("Rates must be non-negative")


def _validate_t(t: int) -> None:
    if t < 1:
        raise EncodingError(f"t must be at least 1: {t}")


@dataclass
class SpikeEncoding:
    """Poisson spike encoding of a 2D rate map over discrete time steps."""

    rates: np.ndarray  # (H, W)
    samples: np.ndarray  # (T, H, W)
    first_spikes: np.ndarray  # (H, W), -1 if never spikes

    @classmethod
    def from_rates(
        cls,
        rates: np.ndarray,
        t: int,
        rng: np.random.Generator | None = None,
    ) -> "SpikeEncoding":
        """Build spike trains by sampling Poisson counts at each pixel.

        Args:
            rates: Non-negative 2D firing-rate map, typically in [0, 1].
            t: Number of time steps to simulate.
            rng: Optional NumPy random generator.

        Returns:
            A ``SpikeEncoding`` with sampled spike counts and first-spike times.

        Raises:
            EncodingError: If ``rates`` or ``t`` are invalid.
        """
        if rng is None:
            rng = np.random.default_rng()
        _validate_rates(rates)
        _validate_t(t)

        samples = _poisson_samples(rates, t, rng)
        return cls(rates=rates, samples=samples, first_spikes=_first_spike(samples))

    @classmethod
    def from_samples(
        cls,
        rates: np.ndarray,
        samples: np.ndarray,
    ) -> "SpikeEncoding":
        """Build an encoding object from an existing sample tensor."""
        _validate_rates(rates)
        if samples.ndim != rates.ndim + 1:
            raise EncodingError(
                f"Samples must have shape (T, H, W); got {samples.shape} for rates {rates.shape}"
            )
        if samples.shape[1:] != rates.shape:
            raise EncodingError(
                f"Sample spatial shape {samples.shape[1:]} must match rates {rates.shape}"
            )
        return cls(rates=rates, samples=samples, first_spikes=_first_spike(samples))

    @property
    def cumulative_sum(self) -> np.ndarray:
        """Cumulative spike counts over time with shape ``(T, H, W)``."""
        return np.cumsum(self.samples, axis=0)

    @property
    def never_spike_mask(self) -> np.ndarray:
        """Boolean mask of pixels that never spiked within ``T`` steps."""
        return self.first_spikes == -1

    @property
    def never_spike_count(self) -> int:
        """Number of pixels that never spiked within ``T`` steps."""
        return int(self.never_spike_mask.sum())

    def expected_never_spike(self, t: int) -> float:
        """Return the expected number of pixels with no spikes in ``t`` steps.

        Args:
            t: Number of time steps used in the expectation.

        Returns:
            Sum of ``exp(-t * rate)`` over all pixels.

        Raises:
            EncodingError: If ``t`` is less than 1.
        """
        _validate_t(t)
        return float(np.exp(-t * self.rates).sum())

    def trace_at(self, row: int, col: int) -> np.ndarray:
        """Return the spike trace at one pixel.

        Args:
            row: Row index in ``[0, height)``.
            col: Column index in ``[0, width)``.

        Returns:
            1D array of spike counts with length ``T``.

        Raises:
            EncodingError: If ``row`` or ``col`` are out of bounds.
        """
        height, width = self.rates.shape
        if not (0 <= row < height):
            raise EncodingError(f"row out of bounds: {row} (height={height})")
        if not (0 <= col < width):
            raise EncodingError(f"col out of bounds: {col} (width={width})")
        return self.samples[:, row, col]
