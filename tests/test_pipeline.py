"""Tests for the training pipeline public API."""

from __future__ import annotations

import spiking_neural_network.pipeline as pipeline_module
from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.config import AdaLiConfig
from spiking_neural_network.pipeline import build_adali_model, full_mnist_split_sizes


class TestPipeline:
    def test_build_adali_model_returns_adali(self) -> None:
        model = build_adali_model(
            hidden=8,
            learning_rate=0.1,
            lr_final=0.01,
            weight_scale=0.2,
            seed=1,
        )
        assert isinstance(model, AdaLi)

    def test_full_mnist_split_sizes(self) -> None:
        assert full_mnist_split_sizes() == (50_000, 10_000, 10_000)

    def test_pipeline_reexports_public_api(self) -> None:
        for name in pipeline_module.__all__:
            assert hasattr(pipeline_module, name)

    def test_build_adali_model_uses_adali_config(self) -> None:
        model = build_adali_model(
            hidden=16,
            learning_rate=0.2,
            lr_final=0.02,
            weight_scale=0.1,
            seed=42,
            focal_gamma=1.5,
            focal_alpha=0.3,
        )
        assert isinstance(model.config, AdaLiConfig)
        assert model.config.hidden_dims == (16,)
