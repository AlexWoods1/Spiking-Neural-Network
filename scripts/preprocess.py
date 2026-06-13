import sys
from pathlib import Path

from spiking_neural_network.config import EncodingConfig
from spiking_neural_network.encoding import EncodingError, SpikeEncoding
from spiking_neural_network.images import (
    ImageError,
    intensity_normalize,
    load_grayscale,
)
from spiking_neural_network.validation import relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "Images" / "Screenshot 2025-05-02 185435.png"


def main() -> None:
    """Load an image, encode spikes, and print validation relative error."""
    try:
        config = EncodingConfig(t_steps=100, seed=42)
        image = load_grayscale(DEFAULT_IMAGE_PATH)
        rates = intensity_normalize(image)

        encoding = SpikeEncoding.from_rates(
            rates=rates, t=config.t_steps, rng=config.make_rng()
        )
        print(
            relative_error(
                expected=encoding.expected_never_spike(t=config.t_steps),
                actual=encoding.never_spike_count,
            )
        )
    except (ImageError, EncodingError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
