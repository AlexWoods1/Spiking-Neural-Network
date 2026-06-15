import numpy as np
import pytest

from spiking_neural_network.config import LIFConfig
from spiking_neural_network.lif import (
    flatten_spikes,
    simulate_timesteps,
    simulate_vector_timesteps,
    synaptic_drive,
    unflatten_spikes,
)


def test_simulate_timesteps_output_shapes() -> None:
    inputs = np.zeros((10, 4, 4))
    inputs[2, 1, 1] = 1.0

    membrane, spikes = simulate_timesteps(inputs)

    assert membrane.shape == (10, 4, 4)
    assert spikes.shape == (10, 4, 4)


def test_simulate_timesteps_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="X must have shape"):
        simulate_timesteps(np.ones(4))


def test_simulate_timesteps_uses_lif_config() -> None:
    inputs = np.zeros((5, 2, 2))
    inputs[1, 0, 0] = 1.0
    low_gain = LIFConfig(input_weight=0.1)
    high_gain = LIFConfig(input_weight=0.9)

    _, low_spikes = simulate_timesteps(inputs, low_gain)
    _, high_spikes = simulate_timesteps(inputs, high_gain)

    assert low_spikes.sum() <= high_spikes.sum()


def test_flatten_and_unflatten_spikes_roundtrip() -> None:
    spikes = np.zeros((6, 2, 3))
    spikes[1, 0, 2] = 1.0

    flat, spatial_shape = flatten_spikes(spikes)
    restored = unflatten_spikes(flat, spatial_shape)

    assert flat.shape == (6, 6)
    assert spatial_shape == (2, 3)
    assert restored.shape == spikes.shape
    np.testing.assert_array_equal(restored, spikes)


def test_flatten_spikes_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="spikes must have shape"):
        flatten_spikes(np.ones(4))


def test_simulate_vector_timesteps_output_shapes() -> None:
    inputs = np.zeros((8, 5))
    inputs[3, 2] = 1.0

    membrane, spikes = simulate_vector_timesteps(inputs)

    assert membrane.shape == (8, 5)
    assert spikes.shape == (8, 5)


def test_synaptic_drive_applies_weight_matrix() -> None:
    spikes_in = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    weights = np.array([[1.0, 0.0, 1.0], [0.5, 0.5, 0.0]])

    drive = synaptic_drive(spikes_in, weights)

    assert drive.shape == (2, 2)
    np.testing.assert_array_equal(drive[0], [2.0, 0.5])
    np.testing.assert_array_equal(drive[1], [0.0, 0.5])


def test_synaptic_drive_rejects_shape_mismatch() -> None:
    spikes_in = np.zeros((4, 3))
    weights = np.zeros((2, 2))

    with pytest.raises(ValueError, match="neuron count must match"):
        synaptic_drive(spikes_in, weights)
