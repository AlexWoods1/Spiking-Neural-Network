import numpy as np
from unittest.mock import patch

from spiking_neural_network.encoding import SpikeEncoding
from spiking_neural_network.plotting import plot_spike_encoding


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
