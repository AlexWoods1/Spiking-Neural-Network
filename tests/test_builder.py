"""Tests for the ``ModelBuilder`` facade."""

from __future__ import annotations

import pytest

import spiking_neural_network.builder as builder_module
from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.builder import ModelBuilder
from spiking_neural_network.config import AdaLiConfig, SNN_Config


class TestModelBuilder:
    def test_build_adali(self) -> None:
        model = ModelBuilder.build("adali", AdaLiConfig())
        assert isinstance(model, AdaLi)

    def test_build_adali_defaults_to_numpy_backend(self) -> None:
        model = ModelBuilder.build("adali", AdaLiConfig())
        assert model.backend == "numpy"

    def test_build_adali_accepts_jax_backend(self) -> None:
        model = ModelBuilder.build("adali", AdaLiConfig(), backend="jax")
        assert model.backend == "jax"

    def test_build_rejects_wrong_config_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid configuration for AdaLi"):
            ModelBuilder.build("adali", SNN_Config())

    def test_build_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid model name"):
            ModelBuilder.build("lif", AdaLiConfig())

    def test_facade_reexports_public_api(self) -> None:
        for name in builder_module.__all__:
            assert hasattr(builder_module, name)
