"""JAX SpikedLM: causal attention + Spiking LSTM decoder for next-token LM."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError as exc:
    raise ImportError("Install jax to use LLM_spiked.model") from exc

from spiking_neural_network.LLM_spiked.config import ModelConfig
from spiking_neural_network.LLM_spiked.spikes import spike

Params = dict[str, Any]


def _layer_norm(
    x: jax.Array,
    scale: jax.Array,
    bias: jax.Array | None,
    eps: float = 1e-5,
) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + eps)
    y = y * scale
    if bias is not None:
        y = y + bias
    return y


def _causal_self_attention(
    x: jax.Array,
    params: Params,
    *,
    n_head: int,
    bias: bool,
) -> jax.Array:
    """Fused QKV causal attention (STLM-style)."""
    b, t, c = x.shape
    head_dim = c // n_head
    qkv = x @ params["c_attn"]
    if bias and params.get("c_attn_bias") is not None:
        qkv = qkv + params["c_attn_bias"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    # (B, n_head, T, head_dim)
    q = q.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)
    att = (q @ jnp.swapaxes(k, -2, -1)) * (1.0 / jnp.sqrt(head_dim))
    mask = jnp.tril(jnp.ones((t, t), dtype=x.dtype))
    att = jnp.where(mask[None, None, :, :] == 0, jnp.asarray(-1e10, dtype=x.dtype), att)
    att = jax.nn.softmax(att, axis=-1)
    y = att @ v
    y = y.transpose(0, 2, 1, 3).reshape(b, t, c)
    y = y @ params["c_proj"]
    if bias and params.get("c_proj_bias") is not None:
        y = y + params["c_proj_bias"]
    return y


def _spiking_lstm(
    x: jax.Array,
    params: Params,
    cfg: ModelConfig,
) -> jax.Array:
    """Scan Spiking LSTM over sequence length ``T``.

    Gate drives integrate as LIF membranes (leak + soft reset), then fire via
    AdaLi spike nonlinearities. The cell state also leaks by ``cfg.leak``.
    """
    b, _t, c = x.shape
    w = params["W"]  # (C, 4C)
    u_rec = params["U"]  # (C, 4C)
    bias_vec = params.get("b")

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        x_t: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        h, cell, gate_u = carry
        drive = x_t @ w + h @ u_rec
        if bias_vec is not None:
            drive = drive + bias_vec
        # * LIF membrane on concatenated gate channels (i, f, g, o).
        gate_u = cfg.leak * gate_u + (1.0 - cfg.leak) * drive
        i, f, g, o = jnp.split(
            spike(
                gate_u,
                cfg.v_th,
                cfg.v_minus,
                cfg.v_plus,
                cfg.alpha,
                cfg.beta,
            ),
            4,
            axis=-1,
        )
        gate_u = gate_u - cfg.v_th * jnp.concatenate([i, f, g, o], axis=-1)
        # * Leaky cell + spike-gated write; hidden is spike-gated cell spike.
        cell_new = f * (cfg.leak * cell) + i * g
        h_new = o * spike(
            cell_new, cfg.v_th, cfg.v_minus, cfg.v_plus, cfg.alpha, cfg.beta
        )
        return (h_new, cell_new, gate_u), h_new

    h0 = jnp.zeros((b, c), dtype=x.dtype)
    c0 = jnp.zeros((b, c), dtype=x.dtype)
    gate_u0 = jnp.zeros((b, 4 * c), dtype=x.dtype)
    # x is (B, T, C); scan expects time-major (T, B, C)
    x_tm = jnp.swapaxes(x, 0, 1)
    _, h_seq = lax.scan(step, (h0, c0, gate_u0), x_tm)
    return jnp.swapaxes(h_seq, 0, 1)


def _block(x: jax.Array, params: Params, cfg: ModelConfig) -> jax.Array:
    """Pre-norm residual: attention then Spiking LSTM."""
    ln1 = params["ln1"]
    x = x + _causal_self_attention(
        _layer_norm(x, ln1["scale"], ln1.get("bias"), eps=1e-5),
        params["attn"],
        n_head=cfg.n_head,
        bias=cfg.bias,
    )
    ln2 = params["ln2"]
    x = x + _spiking_lstm(
        _layer_norm(x, ln2["scale"], ln2.get("bias"), eps=1e-5),
        params["lstm"],
        cfg,
    )
    return x


def init_params(cfg: ModelConfig, rng: jax.Array) -> Params:
    """Initialize SpikedLM parameters ~ N(0, 0.02).

    Args:
        cfg: Model configuration.
        rng: JAX PRNG key.

    Returns:
        Nested parameter pytree.
    """
    keys = jax.random.split(rng, 2 + cfg.n_layer * 8)
    k = 0

    def normal(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jax.random.normal(key, shape) * 0.02

    c = cfg.n_embd
    params: Params = {
        "wte": normal(keys[k], (cfg.vocab_size, c)),
        "wpe": normal(keys[k + 1], (cfg.block_size, c)),
        "blocks": [],
        "ln_f": {
            "scale": jnp.ones((c,)),
            "bias": jnp.zeros((c,)) if cfg.bias else None,
        },
    }
    k += 2

    for _ in range(cfg.n_layer):
        attn = {
            "c_attn": normal(keys[k], (c, 3 * c)),
            "c_proj": normal(keys[k + 1], (c, c)),
            "c_attn_bias": jnp.zeros((3 * c,)) if cfg.bias else None,
            "c_proj_bias": jnp.zeros((c,)) if cfg.bias else None,
        }
        # * Larger LSTM scale so gate membranes reach threshold under LIF leak.
        lstm_scale = 1.0 / jnp.sqrt(c)
        lstm = {
            "W": jax.random.normal(keys[k + 2], (c, 4 * c)) * lstm_scale,
            "U": jax.random.normal(keys[k + 3], (c, 4 * c)) * lstm_scale,
            "b": jnp.zeros((4 * c,)) if cfg.bias else None,
        }
        ln1 = {
            "scale": jnp.ones((c,)),
            "bias": jnp.zeros((c,)) if cfg.bias else None,
        }
        ln2 = {
            "scale": jnp.ones((c,)),
            "bias": jnp.zeros((c,)) if cfg.bias else None,
        }
        params["blocks"].append(
            {"ln1": ln1, "attn": attn, "ln2": ln2, "lstm": lstm}
        )
        k += 4

    return params


def count_parameters(params: Params) -> int:
    """Return the number of scalar parameters in the pytree."""
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves if leaf is not None))


def forward(
    params: Params,
    idx: jax.Array,
    cfg: ModelConfig,
    targets: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Forward pass returning logits and optional next-token CE loss.

    Args:
        params: Model parameters.
        idx: Token ids with shape ``(B, T)``.
        cfg: Model configuration.
        targets: Optional target ids ``(B, T)`` for cross-entropy.

    Returns:
        ``(logits, loss)`` where ``logits`` has shape ``(B, T, V)`` and ``loss``
        is a scalar or ``None``.
    """
    _, t = idx.shape
    if t > cfg.block_size:
        raise ValueError(
            f"Cannot forward sequence of length {t}, "
            f"block size is only {cfg.block_size}"
        )
    pos = jnp.arange(t)
    x = params["wte"][idx] + params["wpe"][pos]
    for block_params in params["blocks"]:
        x = _block(x, block_params, cfg)
    x = _layer_norm(x, params["ln_f"]["scale"], params["ln_f"].get("bias"))
    # * Weight tying: LM head shares wte.
    logits = x @ params["wte"].T
    loss = None
    if targets is not None:
        # ignore_index=-1: mask positions with target -1
        vocab = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab)
        flat_targets = targets.reshape(-1)
        loss = _cross_entropy(flat_logits, flat_targets)
    return logits, loss


def _cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean token cross-entropy; targets equal to -1 are ignored."""
    valid = targets != -1
    # * Clamp ignored targets to 0 so one_hot is well-defined; masked out below.
    safe_targets = jnp.where(valid, targets, 0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, safe_targets[:, None], axis=-1).squeeze(-1)
    nll = jnp.where(valid, nll, 0.0)
    denom = jnp.maximum(valid.sum(), 1)
    return nll.sum() / denom


def loss_fn(
    params: Params,
    idx: jax.Array,
    targets: jax.Array,
    cfg: ModelConfig,
) -> jax.Array:
    """Scalar training loss for ``jax.value_and_grad``."""
    _, loss = forward(params, idx, cfg, targets)
    assert loss is not None
    return loss


def bind_forward_logits(cfg: ModelConfig):
    """Return a JIT logits function closed over ``cfg`` (fixed compile).

    Args:
        cfg: Model configuration captured as compile-time constants.

    Returns:
        ``forward_logits(params, idx) -> logits`` with ``idx`` shape ``(B, T)``.
    """

    @jax.jit
    def forward_logits(params: Params, idx: jax.Array) -> jax.Array:
        logits, _ = forward(params, idx, cfg, targets=None)
        return logits

    return forward_logits


def left_pad_block(token_ids: list[int], block_size: int) -> np.ndarray:
    """Left-pad / crop ``token_ids`` to length ``block_size`` (int32).

    Padding uses id ``0`` so the tensor shape stays fixed for a single JIT
    compile during autoregressive sampling. The last index is always the
    newest real token.
    """
    ctx = token_ids[-block_size:]
    t = len(ctx)
    out = np.zeros((block_size,), dtype=np.int32)
    out[block_size - t :] = np.asarray(ctx, dtype=np.int32)
    return out
