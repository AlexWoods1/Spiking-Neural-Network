from pathlib import Path

import numpy as np
import pytest

from spiking_neural_network.config import (
    EncodingConfig,
    LIFConfig,
    LayerConfig,
    NetworkConfig,
    PreprocessConfig,
)


def test_encoding_config_defaults() -> None:
    config = EncodingConfig()
    assert config.t_steps == 500
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


def test_lif_config_computes_tau_and_beta() -> None:
    config = LIFConfig(threshold=1.0, input_weight=0.3, dt=1.0, R=2.0, C=3.0)

    assert config.tau == pytest.approx(6.0)
    assert config.beta == pytest.approx(np.exp(-1.0 / 6.0))
    assert config.input_weight == pytest.approx(0.3)


def test_lif_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        LIFConfig(threshold=0)
    with pytest.raises(ValueError, match="input_weight must be positive"):
        LIFConfig(input_weight=0)
    with pytest.raises(ValueError, match="dt must be positive"):
        LIFConfig(dt=0)
    with pytest.raises(ValueError, match="R must be positive"):
        LIFConfig(R=0)
    with pytest.raises(ValueError, match="C must be positive"):
        LIFConfig(C=0)


def test_preprocess_config_includes_lif_defaults() -> None:
    config = PreprocessConfig.default(Path("project"))

    assert config.lif.threshold == 1.0
    assert config.lif.input_weight == 0.3
    assert config.lif.C == 5.0
    assert config.lif.tau == pytest.approx(5.0)


def test_layer_config_rejects_invalid_n_neurons() -> None:
    with pytest.raises(ValueError, match="n_neurons must be at least 1"):
        LayerConfig(n_neurons=0)


def test_network_config_default_hidden_size() -> None:
    config = NetworkConfig.default()

    assert config.hidden.n_neurons == 64
    assert config.hidden.lif.tau == pytest.approx(5.0)
    assert config.weight_seed == 42
    assert config.weight_scale == pytest.approx(0.1)


def test_network_config_make_rng_is_reproducible() -> None:
    config = NetworkConfig.default()
    first = config.make_rng().standard_normal(3)
    second = config.make_rng().standard_normal(3)
    assert first.tolist() == second.tolist()


def test_network_config_rejects_invalid_weight_scale() -> None:
    hidden = LayerConfig(n_neurons=8)
    with pytest.raises(ValueError, match="weight_scale must be positive"):
        NetworkConfig(hidden=hidden, weight_scale=0)
