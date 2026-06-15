import sys

from pathlib import Path


import numpy as np


from spiking_neural_network.config import NetworkConfig, PreprocessConfig

from spiking_neural_network.encoding import EncodingError, SpikeEncoding

from spiking_neural_network.images import (
    ImageError,
    intensity_normalize,
    load_grayscale,
    resize_image,
)

from spiking_neural_network.lif import simulate_timesteps

from spiking_neural_network.network import forward

from spiking_neural_network.plotting import (
    plot_membrane_potential,
    plot_spike_encoding,
    plot_spikes,
)

from spiking_neural_network.validation import relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_preprocess(config: PreprocessConfig) -> None:
    """Run the preprocess pipeline for a given configuration."""

    image = load_grayscale(config.image_path)

    if config.resize_shape is not None:

        image = resize_image(image, config.resize_shape)

    rates = intensity_normalize(image)

    encoding = SpikeEncoding.from_rates(
        rates=rates,
        t=config.encoding.t_steps,
        rng=config.encoding.make_rng(),
    )

    plot_spike_encoding(
        encoding,
        image=image,
        title="Poisson encoding",
        show=config.show_plot,
        save_path=config.save_plot,
    )

    input_spikes = (encoding.samples > 0).astype(float)

    t_steps = input_spikes.shape[0]

    network = NetworkConfig.default()

    results = forward(input_spikes, network)

    n_hidden = network.hidden.n_neurons

    hidden_save_path = None

    lif_save_path = None

    membrane_save_path = None

    if config.save_plot is not None:

        hidden_save_path = config.save_plot.with_name(
            f"{config.save_plot.stem}_hidden_spikes{config.save_plot.suffix}"
        )

        lif_save_path = config.save_plot.with_name(
            f"{config.save_plot.stem}_lif_spikes{config.save_plot.suffix}"
        )

        membrane_save_path = config.save_plot.with_name(
            f"{config.save_plot.stem}_membrane{config.save_plot.suffix}"
        )

    plot_spikes(
        results["hidden_spikes"].reshape(t_steps, n_hidden, 1),
        title="Hidden layer spikes",
        show=config.show_plot,
        save_path=hidden_save_path,
    )

    membrane, lif_spikes = simulate_timesteps(input_spikes, config.lif)

    plot_spikes(
        lif_spikes,
        title="Input LIF spikes",
        show=config.show_plot,
        save_path=lif_save_path,
    )

    bright_pixel = np.unravel_index(np.argmax(rates), rates.shape)

    plot_membrane_potential(
        membrane,
        threshold=config.lif.threshold,
        input_spikes=input_spikes,
        trace_at=bright_pixel,
        title="Input LIF membrane potential",
        show=config.show_plot,
        save_path=membrane_save_path,
    )


def main() -> None:
    """Load an image, encode spikes, print validation error, and plot spikes."""

    try:

        run_preprocess(PreprocessConfig.default(PROJECT_ROOT))

    except (ImageError, EncodingError, ValueError) as exc:

        print(f"Error: {exc}", file=sys.stderr)

        raise SystemExit(1) from exc


if __name__ == "__main__":

    main()
