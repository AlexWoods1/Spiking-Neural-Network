import numpy as np
import pytest

from spiking_neural_network.encoding import EncodingError, SpikeEncoding, _poisson_samples


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


def test_from_rates_rejects_empty_rates() -> None:
    with pytest.raises(EncodingError, match="non-zero dimensions"):
        SpikeEncoding.from_rates(rates=np.empty((0, 2)), t=5)


def test_from_rates_rejects_non_finite_rates() -> None:
    rates = np.array([[0.1, np.nan], [0.2, 0.3]])
    with pytest.raises(EncodingError, match="non-finite"):
        SpikeEncoding.from_rates(rates=rates, t=5)


def test_from_rates_without_rng_still_produces_samples() -> None:
    rates = np.full((2, 2), 0.5)
    encoding = SpikeEncoding.from_rates(rates=rates, t=4)

    assert encoding.samples.shape == (4, 2, 2)


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


def test_trace_at_rejects_out_of_bounds_col() -> None:
    rates = np.full((3, 3), 0.8)
    encoding = SpikeEncoding.from_rates(rates=rates, t=5, rng=np.random.default_rng(1))

    with pytest.raises(EncodingError, match="col out of bounds"):
        encoding.trace_at(row=0, col=3)


def test_cumulative_sum_tracks_spikes_over_time() -> None:
    rates = np.full((2, 2), 1.0)
    encoding = SpikeEncoding.from_rates(rates=rates, t=3, rng=np.random.default_rng(4))

    cumulative = encoding.cumulative_sum
    assert cumulative.shape == (3, 2, 2)
    assert np.all(cumulative[-1] == encoding.samples.sum(axis=0))


def test_expected_never_spike_rejects_invalid_t() -> None:
    rates = np.full((2, 2), 0.1)
    encoding = SpikeEncoding.from_rates(rates=rates, t=5, rng=np.random.default_rng(0))

    with pytest.raises(EncodingError, match="t must be at least 1"):
        encoding.expected_never_spike(t=0)


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


def test_from_samples_builds_encoding() -> None:
    rates = np.full((2, 2), 0.5)
    samples = np.zeros((5, 2, 2))
    samples[1, 0, 0] = 1

    encoding = SpikeEncoding.from_samples(rates=rates, samples=samples)

    assert encoding.samples.shape == (5, 2, 2)
    assert encoding.first_spikes[0, 0] == 1


def test_from_samples_rejects_invalid_sample_shape() -> None:
    rates = np.full((2, 2), 0.5)

    with pytest.raises(EncodingError, match="Samples must have shape"):
        SpikeEncoding.from_samples(rates=rates, samples=np.zeros((5, 2)))

    with pytest.raises(EncodingError, match="Sample spatial shape"):
        SpikeEncoding.from_samples(rates=rates, samples=np.zeros((5, 2, 3)))


def test_poisson_samples_without_rng_still_produces_samples() -> None:
    rates = np.full((2, 2), 0.5)
    samples = _poisson_samples(rates, t=3, rng=None)

    assert samples.shape == (3, 2, 2)
