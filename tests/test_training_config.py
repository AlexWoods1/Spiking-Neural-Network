"""Tests for training and model configuration dataclasses."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from spiking_neural_network.config import (
    AdaLiConfig,
    BaseModelConfig,
    DataModuleConfig,
    SNN_Config,
    TrainingConfig,
)
from spiking_neural_network.data_module import MNISTDataConfig
from spiking_neural_network.exceptions import ParameterError
from spiking_neural_network.schedules import (
    EpochContext,
    cosine_learning_rate,
    linear_learning_rate,
)


class TestTrainingConfigs:
    def test_adali_config_rejects_invalid_boundary_order(self) -> None:
        with pytest.raises(
            ParameterError, match="left_initial must be less than right_initial"
        ):
            AdaLiConfig(left_initial=2.0, right_initial=1.0)

    def test_training_config_rejects_zero_epochs(self) -> None:
        with pytest.raises(ParameterError, match="total_epochs must be at least 1"):
            TrainingConfig(train_name="x", total_epochs=0)

    def test_training_config_rejects_invalid_train_name_type(self) -> None:
        with pytest.raises(ParameterError, match="train_name must be a string"):
            TrainingConfig(train_name=123, total_epochs=1)  # type: ignore[arg-type]

    def test_training_config_rejects_invalid_total_epochs_type(self) -> None:
        with pytest.raises(ParameterError, match="epochs must be an integer"):
            TrainingConfig(train_name="x", total_epochs=1.5)  # type: ignore[arg-type]

    def test_data_module_config_rejects_invalid_seed(self) -> None:
        with pytest.raises(ParameterError, match="seed must be positive"):
            DataModuleConfig(seed=0)

    def test_data_module_config_rejects_invalid_batch_size(self) -> None:
        with pytest.raises(ParameterError, match="batch_size must be at least 1"):
            DataModuleConfig(batch_size=0)

    def test_mnist_data_config_rejects_invalid_val_size_and_seed(self) -> None:
        with pytest.raises(ParameterError, match="val_size must be at least 1"):
            MNISTDataConfig(data_dir=Path("data"), val_size=0)
        with pytest.raises(ParameterError, match="seed must be positive"):
            MNISTDataConfig(data_dir=Path("data"), seed=0)

    def test_mnist_data_config_rejects_invalid_limit(self) -> None:
        with pytest.raises(ParameterError, match="train_limit must be at least 1"):
            MNISTDataConfig(data_dir=Path("data"), train_limit=0)

    def test_snn_config_rejects_invalid_learning_rate_schedule(self) -> None:
        def bad_schedule(ctx: EpochContext) -> float:
            return 0.0

        with pytest.raises(
            ParameterError, match="learning_rate schedule must return a positive number"
        ):
            SNN_Config(learning_rate=bad_schedule)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"seed": 0}, "seed must be positive"),
            ({"input_dim": 0}, "input_dim must be positive"),
            ({"output_dim": 0}, "output_dim must be positive"),
            ({"learning_rate": 0.0}, "learning_rate must be positive"),
            ({"dt": 0.0}, "dt must be positive"),
            ({"tau": 0.0}, "tau must be positive"),
            ({"weight_scale": 0.0}, "weight_scale must be positive"),
            ({"hidden_dims": ()}, "hidden_dim must be provided"),
            ({"hidden_dims": (0,)}, "hidden_dim must be positive"),
            ({"decay": 0.0}, "decay must be between 0 and 1"),
            ({"decay": 1.0}, "decay must be between 0 and 1"),
        ],
    )
    def test_snn_config_rejects_invalid_values(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ParameterError, match=match):
            SNN_Config(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"alpha": 0.0}, "alpha must be positive"),
            ({"beta": 0.0}, "beta must be positive"),
            ({"p": 0.0}, "p must be between 0 and 1"),
            ({"p": 1.0}, "p must be between 0 and 1"),
            ({"left_initial": 0.0}, "left_initial must be positive"),
            ({"right_initial": 0.0}, "right_initial must be positive"),
            (
                {"left_initial": 1.5, "right_initial": 1.0},
                "left_initial must be less than right_initial",
            ),
            (
                {"left_initial": 1.5, "right_initial": 2.0, "v_th": 1.0},
                "left_initial must be less than v_th",
            ),
            (
                {"left_initial": 0.3, "right_initial": 0.8, "v_th": 1.0},
                "right_initial must be greater than v_th",
            ),
        ],
    )
    def test_adali_config_rejects_invalid_values(
        self, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ParameterError, match=match):
            AdaLiConfig(**kwargs)

    def test_base_model_config_rejects_invalid_seed(self) -> None:
        with pytest.raises(ParameterError, match="seed must be positive"):
            BaseModelConfig(model_name="x", input_dim=1, output_dim=1, seed=0)

    def test_base_model_config_rejects_invalid_types(self) -> None:
        with pytest.raises(ParameterError, match="model_name must be a string"):
            BaseModelConfig(model_name=123, input_dim=1, output_dim=1)  # type: ignore[arg-type]
        with pytest.raises(ParameterError, match="input_dim must be an integer"):
            BaseModelConfig(model_name="x", input_dim=1.5, output_dim=1)  # type: ignore[arg-type]
        with pytest.raises(ParameterError, match="output_dim must be an integer"):
            BaseModelConfig(model_name="x", input_dim=1, output_dim=1.5)  # type: ignore[arg-type]
        with pytest.raises(ParameterError, match="seed must be an integer"):
            BaseModelConfig(model_name="x", input_dim=1, output_dim=1, seed=1.5)  # type: ignore[arg-type]

    def test_snn_config_rejects_non_positive_input_and_output_dims(self) -> None:
        with pytest.raises(ParameterError, match="input_dim must be positive"):
            SNN_Config(input_dim=0)
        with pytest.raises(ParameterError, match="output_dim must be positive"):
            SNN_Config(output_dim=0)

    def test_snn_config_rejects_invalid_learning_rate_type(self) -> None:
        with pytest.raises(
            ParameterError, match="learning_rate must be a positive number or callable"
        ):
            SNN_Config(learning_rate="bad")  # type: ignore[arg-type]

    def test_adali_config_rejects_invalid_v_th(self) -> None:
        with pytest.raises(ParameterError, match="v_th must be positive"):
            AdaLiConfig(v_th=0)

    def test_training_config_rejects_none_train_name(self) -> None:
        with pytest.raises(ParameterError, match="train_name must be provided"):
            TrainingConfig(train_name=None, total_epochs=1)  # type: ignore[arg-type]

    def test_base_model_config_rejects_none_model_name(self) -> None:
        with pytest.raises(ParameterError, match="model_name must be provided"):
            BaseModelConfig(model_name=None, input_dim=1, output_dim=1)  # type: ignore[arg-type]

    def test_snn_config_post_init_validates_dims_when_base_checks_bypassed(
        self,
    ) -> None:
        config = SNN_Config.__new__(SNN_Config)
        for name, value in (
            ("model_name", "SNN"),
            ("input_dim", 0),
            ("output_dim", 10),
            ("hidden_dims", (8,)),
            ("weight_scale", 0.5),
            ("dt", 1.0),
            ("tau", 2.0),
            ("v_th", 1.0),
            ("decay", 0.9),
            ("seed", 42),
            ("learning_rate", 0.01),
        ):
            object.__setattr__(config, name, value)

        with patch.object(BaseModelConfig, "__post_init__", lambda self: None):
            with pytest.raises(ParameterError, match="input_dim must be positive"):
                SNN_Config.__post_init__(config)

            object.__setattr__(config, "input_dim", 784)
            object.__setattr__(config, "output_dim", 0)
            with pytest.raises(ParameterError, match="output_dim must be positive"):
                SNN_Config.__post_init__(config)

    def test_mnist_data_config_rejects_invalid_t_steps_and_limits(self) -> None:
        with pytest.raises(ParameterError, match="t_steps must be at least 1"):
            MNISTDataConfig(data_dir=Path("data"), t_steps=0)
        with pytest.raises(ParameterError, match="val_limit must be at least 1"):
            MNISTDataConfig(data_dir=Path("data"), val_limit=0)
