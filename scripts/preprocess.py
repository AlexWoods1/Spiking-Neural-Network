import cv2

from spiking_neural_network.config import EncodingConfig
from spiking_neural_network.encoding import SpikeEncoding
from spiking_neural_network.images import intensity_normalize
from spiking_neural_network.validation import relative_error

DEFAULT_IMAGE_PATH = "Images/Screenshot 2025-05-02 185435.png"


def main() -> None:
    config = EncodingConfig(t_steps=100, seed=42)
    image = cv2.imread(DEFAULT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
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


if __name__ == "__main__":
    main()
