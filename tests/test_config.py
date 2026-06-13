import pytest

from spiking_neural_network.config import EncodingConfig


def test_encoding_config_defaults() -> None:
    config = EncodingConfig()
    assert config.t_steps == 100
    assert config.seed is None


def test_encoding_config_rejects_invalid_t_steps() -> None:
    with pytest.raises(ValueError, match="t_steps must be at least 1"):
        EncodingConfig(t_steps=0)


def test_make_rng_is_reproducible_with_seed() -> None:
    config = EncodingConfig(seed=42)
    first = config.make_rng().integers(0, 1000, size=5)
    second = config.make_rng().integers(0, 1000, size=5)
    assert first.tolist() == second.tolist()
