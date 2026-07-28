"""Tests for SpikedLM config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spiking_neural_network.LLM_spiked.config import (
    ModelConfig,
    load_config,
)


def test_model_config_rejects_bad_head_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            n_layer=1,
            n_head=3,
            n_embd=64,
            block_size=16,
            vocab_size=10,
        )


def test_model_config_rejects_bad_surrogate_bounds() -> None:
    with pytest.raises(ValueError, match="v_minus"):
        ModelConfig(
            n_layer=1,
            n_head=2,
            n_embd=4,
            block_size=8,
            vocab_size=10,
            v_minus=1.5,
            v_th=1.0,
            v_plus=2.0,
        )


def test_load_config_smoke_yaml(tmp_path: Path) -> None:
    raw = {
        "model": {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 8,
            "block_size": 16,
            "vocab_size": 20,
        },
        "train": {
            "batch_size": 2,
            "max_steps": 5,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.99,
            "warmup_steps": 1,
            "grad_clip": 1.0,
            "eval_interval": 1,
            "eval_batches": 1,
            "sample_interval": 1,
            "checkpoint_interval": 1,
            "seed": 0,
        },
        "data": {
            "dataset": "shakespeare",
            "data_dir": "data/x",
            "train_frac": 0.9,
        },
        "paths": {
            "out_dir": "out",
            "tokenizer_path": "tok.json",
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.n_embd == 8
    assert cfg.data.data_dir == Path("data/x")


def test_model_config_type_errors() -> None:
    with pytest.raises(TypeError):
        ModelConfig(
            n_layer="1",  # type: ignore[arg-type]
            n_head=2,
            n_embd=4,
            block_size=8,
            vocab_size=10,
        )
    with pytest.raises(ValueError, match="v_plus"):
        ModelConfig(
            n_layer=1,
            n_head=2,
            n_embd=4,
            block_size=8,
            vocab_size=10,
            v_th=1.0,
            v_minus=0.0,
            v_plus=0.5,
        )


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text(yaml.dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="dictionary"):
        load_config(path)


def test_data_config_empty_dataset() -> None:
    from spiking_neural_network.LLM_spiked.config import DataConfig

    with pytest.raises(ValueError, match="dataset"):
        DataConfig(dataset="", data_dir="x", train_frac=0.9)


def test_load_config_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.dump(
            {
                "model": {
                    "n_layer": 1,
                    "n_head": 1,
                    "n_embd": 4,
                    "block_size": 8,
                    "vocab_size": 10,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing required key"):
        load_config(path)


def test_load_config_invalid_model_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad_model.yaml"
    path.write_text(yaml.dump({"model": {"n_layer": 1}}), encoding="utf-8")
    with pytest.raises(TypeError, match="Invalid config fields"):
        load_config(path)
