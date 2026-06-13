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


@dataclass
class SpikeEncoding:
    rates: np.ndarray  # (H, W)
    samples: np.ndarray  # (T, H, W)
    first_spikes: np.ndarray  # (H, W), -1 if never spikes

    @classmethod
    def from_rates(cls, rates: np.ndarray, t: int, rng=None) -> "SpikeEncoding":
        samples = _poisson_samples(rates, t, rng)
        return cls(rates=rates, samples=samples, first_spikes=_first_spike(samples))

    @property
    def cumulative_sum(self) -> np.ndarray:
        return np.cumsum(self.samples, axis=0)

    @property
    def never_spike_mask(self) -> np.ndarray:
        return self.first_spikes == -1

    @property
    def never_spike_count(self) -> int:
        return int(self.never_spike_mask.sum())

    def expected_never_spike(self, t: int) -> float:
        return float(np.exp(-t * self.rates).sum())

    def trace_at(self, row: int, col: int) -> np.ndarray:
        return self.samples[:, row, col]
