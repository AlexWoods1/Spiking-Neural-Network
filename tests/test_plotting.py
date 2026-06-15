import numpy as np
import pytest
from unittest.mock import patch

from spiking_neural_network.encoding import SpikeEncoding
from spiking_neural_network.plotting import (
    plot_membrane_potential,
    plot_spike_encoding,
    plot_spikes,
)


def test_plot_spike_encoding_saves_png(tmp_path) -> None:
    rates = np.full((4, 4), 0.25)
    encoding = SpikeEncoding.from_rates(rates=rates, t=5, rng=np.random.default_rng(0))
    image = np.full((4, 4), 128, dtype=np.uint8)
    output = tmp_path / "spike_plot.png"

    plot_spike_encoding(
        encoding,
        image=image,
        show=False,
        save_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_spike_encoding_without_image(tmp_path) -> None:
    rates = np.full((3, 3), 0.5)
    encoding = SpikeEncoding.from_rates(rates=rates, t=4, rng=np.random.default_rng(1))
    output = tmp_path / "no_image_plot.png"

    plot_spike_encoding(
        encoding,
        title="Test encoding",
        show=False,
        save_path=output,
    )

    assert output.is_file()


def test_plot_spike_encoding_show_calls_pyplot() -> None:
    rates = np.full((2, 2), 0.2)
    encoding = SpikeEncoding.from_rates(rates=rates, t=3, rng=np.random.default_rng(2))

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_spike_encoding(encoding, show=True)

    mock_show.assert_called_once()


def test_plot_spikes_saves_png(tmp_path) -> None:
    spikes = np.zeros((5, 4, 4))
    spikes[0, 0, 0] = 1
    spikes[2, 1, 2] = 1
    output = tmp_path / "spikes.png"

    plot_spikes(spikes, title="Test spikes", show=False, save_path=output)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_spikes_show_calls_pyplot() -> None:
    spikes = np.zeros((3, 2, 2))
    spikes[1, 0, 1] = 1

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_spikes(spikes, show=True)

    mock_show.assert_called_once()


def test_plot_spikes_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="spikes must have shape"):
        plot_spikes(np.ones((4, 4)))


def test_plot_membrane_potential_saves_png(tmp_path) -> None:
    membrane = np.linspace(0, 1.2, 10).reshape(10, 1, 1)
    input_spikes = np.zeros((10, 1, 1))
    input_spikes[2, 0, 0] = 1
    output = tmp_path / "membrane.png"

    plot_membrane_potential(
        membrane,
        threshold=1.0,
        input_spikes=input_spikes,
        trace_at=(0, 0),
        show=False,
        save_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_membrane_potential_show_calls_pyplot() -> None:
    membrane = np.zeros((5, 2, 2))

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_membrane_potential(membrane, threshold=1.0, show=True)

    mock_show.assert_called_once()


def test_plot_membrane_potential_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="membrane must have shape"):
        plot_membrane_potential(np.ones((4, 4)), threshold=1.0)
