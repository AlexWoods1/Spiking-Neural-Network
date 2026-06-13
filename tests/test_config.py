from pathlib import Path

import pytest

from spiking_neural_network.config import EncodingConfig, PreprocessConfig


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


def test_preprocess_config_default_uses_project_image(tmp_path: Path) -> None:
    config = PreprocessConfig.default(tmp_path)

    assert config.image_path == tmp_path / "Images" / "Screenshot 2025-05-02 185435.png"
    assert config.resize_shape == (32, 32)
    assert config.encoding.t_steps == 100
    assert config.encoding.seed == 42
    assert config.show_plot is True
    assert config.save_plot is None


def test_preprocess_config_rejects_invalid_resize_shape() -> None:
    with pytest.raises(ValueError, match="resize_shape dimensions must be positive"):
        PreprocessConfig(image_path=Path("image.png"), resize_shape=(0, 32))
