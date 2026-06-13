import sys
from pathlib import Path

from spiking_neural_network.config import PreprocessConfig
from spiking_neural_network.encoding import EncodingError, SpikeEncoding
from spiking_neural_network.images import (
    ImageError,
    intensity_normalize,
    load_grayscale,
    resize_image,
)
from spiking_neural_network.plotting import plot_spike_encoding
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
    print(
        relative_error(
            expected=encoding.expected_never_spike(t=config.encoding.t_steps),
            actual=encoding.never_spike_count,
        )
    )
    plot_spike_encoding(
        encoding,
        image=image,
        show=config.show_plot,
        save_path=config.save_plot,
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
