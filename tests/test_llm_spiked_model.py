"""Tests for SpikedLM forward pass, causality, and training step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
import optax

from spiking_neural_network.LLM_spiked.config import ModelConfig
from spiking_neural_network.LLM_spiked.data import CharDataset, CharTokenizer
from spiking_neural_network.LLM_spiked.generate import apply_top_k, generate, load_checkpoint
from spiking_neural_network.LLM_spiked.model import (
    count_parameters,
    forward,
    init_params,
    loss_fn,
)
from spiking_neural_network.LLM_spiked.train import (
    get_lr,
    save_checkpoint,
    sample_text,
)
from spiking_neural_network.LLM_spiked.config import (
    Config,
    DataConfig,
    PathsConfig,
    TrainConfig,
)


def _tiny_cfg(vocab_size: int = 12) -> ModelConfig:
    return ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=16,
        vocab_size=vocab_size,
        bias=True,
    )


def test_forward_shapes_and_finite_loss() -> None:
    cfg = _tiny_cfg()
    params = init_params(cfg, jax.random.PRNGKey(0))
    assert count_parameters(params) > 0
    idx = jnp.zeros((2, 8), dtype=jnp.int32)
    targets = jnp.ones((2, 8), dtype=jnp.int32)
    logits, loss = forward(params, idx, cfg, targets)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None
    assert jnp.isfinite(loss)


def test_forward_rejects_overlong_sequence() -> None:
    cfg = _tiny_cfg()
    params = init_params(cfg, jax.random.PRNGKey(0))
    idx = jnp.zeros((1, cfg.block_size + 1), dtype=jnp.int32)
    with pytest.raises(ValueError, match="block size"):
        forward(params, idx, cfg)


def test_causal_mask_blocks_future_tokens() -> None:
    cfg = _tiny_cfg()
    params = init_params(cfg, jax.random.PRNGKey(1))
    t = 8
    base = jnp.zeros((1, t), dtype=jnp.int32)
    # * Flip only the last token; logits at earlier positions must match.
    flipped = base.at[0, -1].set(3)
    logits_a, _ = forward(params, base, cfg)
    logits_b, _ = forward(params, flipped, cfg)
    assert jnp.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)
    # Last position may differ.
    assert not jnp.allclose(logits_a[0, -1], logits_b[0, -1], atol=1e-5)


def test_spiking_lstm_path_has_nonzero_gate_grad() -> None:
    # * Lower threshold so LIF gate membranes fire and surrogates are active.
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=16,
        vocab_size=12,
        bias=True,
        v_th=0.2,
        leak=0.5,
        v_minus=-1.0,
        v_plus=1.0,
    )
    params = init_params(cfg, jax.random.PRNGKey(2))
    idx = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % cfg.vocab_size
    targets = jnp.roll(idx, -1, axis=1)
    loss, grads = jax.value_and_grad(loss_fn)(params, idx, targets, cfg)
    assert jnp.isfinite(loss)
    lstm_w_grad = grads["blocks"][0]["lstm"]["W"]
    assert float(jnp.abs(lstm_w_grad).sum()) > 0.0


def test_one_step_train_updates_params() -> None:
    cfg = _tiny_cfg()
    params = init_params(cfg, jax.random.PRNGKey(3))
    idx = jnp.zeros((2, 8), dtype=jnp.int32)
    targets = jnp.ones((2, 8), dtype=jnp.int32)
    optimizer = optax.adamw(learning_rate=1e-2)
    opt_state = optimizer.init(params)
    loss, grads = jax.value_and_grad(loss_fn)(params, idx, targets, cfg)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    delta = jnp.sum(
        jnp.abs(new_params["wte"] - params["wte"])
    )
    assert float(delta) > 0.0
    assert jnp.isfinite(loss)


def test_get_lr_warmup_and_floor() -> None:
    train_cfg = TrainConfig(
        batch_size=2,
        max_steps=100,
        learning_rate=1e-3,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.99,
        warmup_steps=10,
        grad_clip=1.0,
        eval_interval=1,
        eval_batches=1,
        sample_interval=1,
        checkpoint_interval=1,
        seed=0,
    )
    assert get_lr(5, train_cfg) == pytest.approx(5e-4)
    assert get_lr(100, train_cfg) == pytest.approx(1e-4)
    assert get_lr(200, train_cfg) == pytest.approx(1e-4)


def _write_tiny_corpus(root: Path) -> None:
    train = "abcdefghij\n" * 50
    val = "abcdefghij\n" * 12
    (root / "train.txt").write_text(train, encoding="utf-8")
    (root / "val.txt").write_text(val, encoding="utf-8")
    CharTokenizer.from_text(train + val).save(root / "tokenizer.json")


def test_checkpoint_roundtrip_and_generate(tmp_path: Path) -> None:
    _write_tiny_corpus(tmp_path)
    tok = CharTokenizer.load(tmp_path / "tokenizer.json")
    cfg_model = _tiny_cfg(vocab_size=tok.vocab_size)
    params = init_params(cfg_model, jax.random.PRNGKey(4))
    cfg = Config(
        model=cfg_model,
        train=TrainConfig(
            batch_size=2,
            max_steps=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.99,
            warmup_steps=0,
            grad_clip=1.0,
            eval_interval=1,
            eval_batches=1,
            sample_interval=1,
            checkpoint_interval=1,
            seed=0,
        ),
        data=DataConfig(dataset="shakespeare", data_dir=tmp_path, train_frac=0.9),
        paths=PathsConfig(out_dir=tmp_path / "ckpt", tokenizer_path=tmp_path / "tokenizer.json"),
    )
    ckpt_path = tmp_path / "ckpt" / "ckpt_1.pkl"
    save_checkpoint(
        ckpt_path,
        params=params,
        opt_state=None,
        step=1,
        cfg=cfg,
        tokenizer=tok,
    )
    loaded_params, loaded_cfg, loaded_tok = load_checkpoint(ckpt_path)
    assert loaded_cfg.vocab_size == tok.vocab_size
    assert loaded_tok.vocab_size == tok.vocab_size

    text = generate(
        loaded_params,
        loaded_tok,
        loaded_cfg,
        "ab",
        max_tokens=5,
        temperature=0.8,
        top_k=5,
        seed=0,
    )
    assert text.startswith("ab")
    assert len(text) == 2 + 5

    data = CharDataset(tmp_path, tmp_path / "tokenizer.json", rng=np.random.default_rng(0))
    sample = sample_text(loaded_params, data, cfg, max_new_tokens=4, rng=np.random.default_rng(1))
    assert len(sample) >= 1


def test_forward_without_bias() -> None:
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=16,
        vocab_size=12,
        bias=False,
    )
    params = init_params(cfg, jax.random.PRNGKey(5))
    idx = jnp.zeros((1, 4), dtype=jnp.int32)
    logits, loss = forward(params, idx, cfg)
    assert logits.shape == (1, 4, 12)
    assert loss is None


def test_ignore_index_in_loss() -> None:
    cfg = _tiny_cfg()
    params = init_params(cfg, jax.random.PRNGKey(6))
    idx = jnp.zeros((1, 4), dtype=jnp.int32)
    targets = jnp.array([[0, -1, -1, -1]], dtype=jnp.int32)
    _, loss = forward(params, idx, cfg, targets)
    assert loss is not None
    assert jnp.isfinite(loss)


def test_apply_top_k() -> None:
    logits = np.array([1.0, 3.0, 2.0, 0.0])
    masked = apply_top_k(logits, 2)
    assert np.isneginf(masked[0])
    assert np.isneginf(masked[3])
    assert masked[1] == 3.0


def test_load_checkpoint_uses_embedded_chars(tmp_path: Path) -> None:
    _write_tiny_corpus(tmp_path)
    tok = CharTokenizer.load(tmp_path / "tokenizer.json")
    cfg_model = _tiny_cfg(vocab_size=tok.vocab_size)
    params = init_params(cfg_model, jax.random.PRNGKey(7))
    cfg = Config(
        model=cfg_model,
        train=TrainConfig(
            batch_size=2,
            max_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.99,
            warmup_steps=0,
            grad_clip=1.0,
            eval_interval=1,
            eval_batches=1,
            sample_interval=1,
            checkpoint_interval=1,
            seed=0,
        ),
        data=DataConfig(dataset="shakespeare", data_dir=tmp_path, train_frac=0.9),
        paths=PathsConfig(
            out_dir=tmp_path / "ckpt",
            tokenizer_path=tmp_path / "missing_tokenizer.json",
        ),
    )
    ckpt_path = tmp_path / "ckpt" / "ckpt_chars.pkl"
    save_checkpoint(
        ckpt_path,
        params=params,
        opt_state=None,
        step=1,
        cfg=cfg,
        tokenizer=tok,
    )
    _, _, loaded_tok = load_checkpoint(ckpt_path)
    assert loaded_tok.chars == tok.chars
