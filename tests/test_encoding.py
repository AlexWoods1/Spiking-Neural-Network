import numpy as np
import pytest

from spiking_neural_network.encoding import EncodingError, SpikeEncoding


def test_from_rates_output_shapes() -> None:
    rates = np.full((4, 4), 0.5)
    encoding = SpikeEncoding.from_rates(rates=rates, t=10, rng=np.random.default_rng(0))

    assert encoding.samples.shape == (10, 4, 4)
    assert encoding.first_spikes.shape == (4, 4)
    assert encoding.rates.shape == (4, 4)


def test_from_rates_rejects_non_positive_t() -> None:
    rates = np.full((2, 2), 0.1)
    with pytest.raises(EncodingError, match="t must be at least 1"):
        SpikeEncoding.from_rates(rates=rates, t=0)


def test_from_rates_rejects_negative_rates() -> None:
    rates = np.array([[0.1, -0.1], [0.2, 0.3]])
    with pytest.raises(EncodingError, match="non-negative"):
        SpikeEncoding.from_rates(rates=rates, t=5)


def test_from_rates_rejects_non_2d_rates() -> None:
    with pytest.raises(EncodingError, match="Rates must be 2D"):
        SpikeEncoding.from_rates(rates=np.ones((2, 2, 2)), t=5)


def test_trace_at_returns_time_series() -> None:
    rates = np.full((3, 3), 0.8)
    encoding = SpikeEncoding.from_rates(rates=rates, t=7, rng=np.random.default_rng(1))

    trace = encoding.trace_at(row=1, col=2)
    assert trace.shape == (7,)


def test_trace_at_rejects_out_of_bounds_index() -> None:
    rates = np.full((3, 3), 0.8)
    encoding = SpikeEncoding.from_rates(rates=rates, t=5, rng=np.random.default_rng(1))

    with pytest.raises(EncodingError, match="row out of bounds"):
        encoding.trace_at(row=3, col=0)


def test_zero_rates_never_spike() -> None:
    rates = np.zeros((5, 5))
    encoding = SpikeEncoding.from_rates(rates=rates, t=20, rng=np.random.default_rng(2))

    assert encoding.never_spike_count == 25
    assert encoding.expected_never_spike(t=20) == pytest.approx(25.0)


def test_expected_never_spike_uses_rate_formula() -> None:
    rates = np.array([[0.0, 0.5], [0.25, 0.0]])
    encoding = SpikeEncoding.from_rates(rates=rates, t=10, rng=np.random.default_rng(3))

    expected = float(np.exp(-10 * rates).sum())
    assert encoding.expected_never_spike(t=10) == pytest.approx(expected)


def test_from_rates_is_reproducible_with_seed() -> None:
    rates = np.full((3, 3), 0.4)
    first = SpikeEncoding.from_rates(rates=rates, t=8, rng=np.random.default_rng(99))
    second = SpikeEncoding.from_rates(rates=rates, t=8, rng=np.random.default_rng(99))

    assert np.array_equal(first.samples, second.samples)
