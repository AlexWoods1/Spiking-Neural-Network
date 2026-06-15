import numpy as np
import pytest

from spiking_neural_network.config import LIFConfig, LayerConfig, NetworkConfig
from spiking_neural_network.network import forward, init_weights, simulate_layer


def test_init_weights_shape_and_reproducibility() -> None:
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)

    weights_a = init_weights(12, 4, rng_a, scale=0.02)
    weights_b = init_weights(12, 4, rng_b, scale=0.02)

    assert weights_a.shape == (4, 12)
    np.testing.assert_array_equal(weights_a, weights_b)


def test_init_weights_rejects_invalid_arguments() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_pre and n_post must be at least 1"):
        init_weights(0, 4, rng)
    with pytest.raises(ValueError, match="scale must be positive"):
        init_weights(4, 2, rng, scale=0)


def test_simulate_layer_zero_input_produces_no_spikes() -> None:
    spikes_in = np.zeros((10, 6))
    weights = np.full((3, 6), 0.5)
    layer = LayerConfig(n_neurons=3)

    _, spikes = simulate_layer(spikes_in, weights, layer)

    assert spikes.sum() == 0


def test_simulate_layer_strong_input_can_spike() -> None:
    spikes_in = np.zeros((10, 2))
    spikes_in[1, 0] = 1.0
    weights = np.array([[10.0, 0.0], [0.0, 0.0]])
    layer = LayerConfig(n_neurons=2, lif=LIFConfig(input_weight=1.0))

    _, spikes = simulate_layer(spikes_in, weights, layer)

    assert spikes.sum() > 0


def test_forward_output_shapes_from_spatial_input() -> None:
    input_spikes = np.zeros((15, 4, 4))
    input_spikes[4, 2, 2] = 1.0
    network = NetworkConfig.default()

    result = forward(input_spikes, network)

    assert result["input"].shape == (15, 16)
    assert result["weights_input_hidden"].shape == (64, 16)
    assert result["hidden_membrane"].shape == (15, 64)
    assert result["hidden_spikes"].shape == (15, 64)
    assert result["spatial_shape"] == (4, 4)


def test_forward_accepts_flat_input() -> None:
    input_spikes = np.zeros((8, 5))
    hidden = LayerConfig(n_neurons=3)
    network = NetworkConfig(hidden=hidden, weight_seed=1)

    result = forward(input_spikes, network)

    assert result["input"].shape == (8, 5)
    assert result["weights_input_hidden"].shape == (3, 5)
    assert result["spatial_shape"] is None
