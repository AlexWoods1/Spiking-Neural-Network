import numpy as np

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
