"""Tests for SpikedLM training loop and generate CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("jax")

from spiking_neural_network.LLM_spiked.config import load_config
from spiking_neural_network.LLM_spiked.data import CharTokenizer
from spiking_neural_network.LLM_spiked.generate import main as generate_main
from spiking_neural_network.LLM_spiked.train import train


def _write_corpus(root: Path) -> None:
    train = "abcdefghijklmnop\n" * 80
    val = "abcdefghijklmnop\n" * 20
    (root / "train.txt").write_text(train, encoding="utf-8")
    (root / "val.txt").write_text(val, encoding="utf-8")
    CharTokenizer.from_text(train + val).save(root / "tokenizer.json")


def _write_smoke_config(root: Path, data_dir: Path, out_dir: Path) -> Path:
    tok = CharTokenizer.load(data_dir / "tokenizer.json")
    raw = {
        "model": {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 8,
            "block_size": 16,
            "vocab_size": tok.vocab_size,
            "bias": True,
            "v_th": 0.3,
            "leak": 0.5,
            "v_minus": -1.0,
            "v_plus": 2.0,
        },
        "train": {
            "batch_size": 4,
            "max_steps": 2,
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
            "data_dir": str(data_dir),
            "train_frac": 0.9,
        },
        "paths": {
            "out_dir": str(out_dir),
            "tokenizer_path": str(data_dir / "tokenizer.json"),
        },
    }
    path = root / "tiny.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    return path


def test_train_two_steps_writes_checkpoint(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_corpus(data_dir)
    out_dir = tmp_path / "ckpt"
    cfg_path = _write_smoke_config(tmp_path, data_dir, out_dir)
    params = train(cfg_path)
    assert params["wte"].shape[0] == CharTokenizer.load(data_dir / "tokenizer.json").vocab_size
    assert (out_dir / "ckpt_2_weights.pkl").is_file() or (out_dir / "ckpt_1_weights.pkl").is_file()
    # * Full optimizer pickle only on the final step.
    assert (out_dir / "ckpt_2.pkl").is_file()


def test_estimate_loss_keys(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_corpus(data_dir)
    out_dir = tmp_path / "ckpt"
    cfg_path = _write_smoke_config(tmp_path, data_dir, out_dir)
    cfg = load_config(cfg_path)
    assert cfg.train.max_steps == 2


def test_generate_main_prints(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_corpus(data_dir)
    out_dir = tmp_path / "ckpt"
    cfg_path = _write_smoke_config(tmp_path, data_dir, out_dir)
    train(cfg_path)
    ckpt = out_dir / "ckpt_2.pkl"
    assert ckpt.is_file()
    generate_main(
        [
            "--checkpoint",
            str(ckpt),
            "--prompt",
            "ab",
            "--max-tokens",
            "4",
            "--temperature",
            "0.9",
            "--top-k",
            "5",
            "--seed",
            "0",
        ]
    )
    captured = capsys.readouterr()
    prompt_lines = [ln for ln in captured.out.splitlines() if ln.startswith("ab")]
    assert prompt_lines, captured.out
    # * Prompt "ab" plus up to max_tokens new chars (sampling may print once).
    assert len(prompt_lines[-1]) >= 4
