"""JAX SpikedLM: causal attention + Spiking LSTM decoder for next-token LM."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

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
LogitsFn = Callable[[Params, jax.Array], jax.Array]


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
    """Fused QKV causal attention; prefers ``jax.nn.dot_product_attention``."""
    b, t, c = x.shape
    head_dim = c // n_head
    qkv = x @ params["c_attn"]
    if bias and params.get("c_attn_bias") is not None:
        qkv = qkv + params["c_attn_bias"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    # * BTNH layout for SDPA / cuDNN flash when available.
    q = q.reshape(b, t, n_head, head_dim)
    k = k.reshape(b, t, n_head, head_dim)
    v = v.reshape(b, t, n_head, head_dim)
    sdpa = getattr(jax.nn, "dot_product_attention", None)
    if sdpa is not None:
        y = sdpa(q, k, v, is_causal=True)
    else:
        # Fallback: explicit scores (older JAX).
        q_t = q.transpose(0, 2, 1, 3)
        k_t = k.transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)
        att = (q_t @ jnp.swapaxes(k_t, -2, -1)) * (1.0 / jnp.sqrt(head_dim))
        mask = jnp.tril(jnp.ones((t, t), dtype=x.dtype))
        att = jnp.where(
            mask[None, None, :, :] == 0, jnp.asarray(-1e10, dtype=x.dtype), att
        )
        att = jax.nn.softmax(att, axis=-1)
        y = (att @ v_t).transpose(0, 2, 1, 3)
    y = y.reshape(b, t, c)
    y = y @ params["c_proj"]
    if bias and params.get("c_proj_bias") is not None:
        y = y + params["c_proj_bias"]
    return y


def _spiking_lstm_step(
    carry: tuple[jax.Array, jax.Array, jax.Array],
    x_t: jax.Array,
    params: Params,
    cfg: ModelConfig,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """One Spiking LSTM step: spike ``i/f/o``, tanh candidate / hidden."""
    h, cell, gate_u = carry
    w = params["W"]
    u_rec = params["U"]
    bias_vec = params.get("b")
    drive = x_t @ w + h @ u_rec
    if bias_vec is not None:
        drive = drive + bias_vec
    # * LIF membrane on gate channels; soft-reset after spikes.
    gate_u = cfg.leak * gate_u + (1.0 - cfg.leak) * drive
    i_pre, f_pre, g_pre, o_pre = jnp.split(gate_u, 4, axis=-1)
    i = spike(i_pre, cfg.v_th, cfg.v_minus, cfg.v_plus, cfg.alpha, cfg.beta)
    f = spike(f_pre, cfg.v_th, cfg.v_minus, cfg.v_plus, cfg.alpha, cfg.beta)
    o = spike(o_pre, cfg.v_th, cfg.v_minus, cfg.v_plus, cfg.alpha, cfg.beta)
    # * Continuous cell write (tanh) — avoids dead binary hidden states.
    g = jnp.tanh(g_pre)
    gate_u = gate_u - cfg.v_th * jnp.concatenate(
        [i, f, jnp.zeros_like(g), o], axis=-1
    )
    cell_new = f * (cfg.leak * cell) + i * g
    h_new = o * jnp.tanh(cell_new)
    return (h_new, cell_new, gate_u), h_new


def _spiking_lstm(
    x: jax.Array,
    params: Params,
    cfg: ModelConfig,
) -> jax.Array:
    """Scan Spiking LSTM over sequence length ``T``."""
    b, _t, c = x.shape

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        x_t: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        return _spiking_lstm_step(carry, x_t, params, cfg)

    h0 = jnp.zeros((b, c), dtype=x.dtype)
    c0 = jnp.zeros((b, c), dtype=x.dtype)
    gate_u0 = jnp.zeros((b, 4 * c), dtype=x.dtype)
    x_tm = jnp.swapaxes(x, 0, 1)
    # * Unroll cuts scan overhead on GPU for moderate T.
    _, h_seq = lax.scan(step, (h0, c0, gate_u0), x_tm, unroll=8)
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
        lstm_b = None
        if cfg.bias:
            lstm_b = jnp.zeros((4 * c,))
            # * Forget-gate bias > 0 so memory starts open.
            lstm_b = lstm_b.at[c : 2 * c].set(1.0)
        lstm = {
            "W": jax.random.normal(keys[k + 2], (c, 4 * c)) * lstm_scale,
            "U": jax.random.normal(keys[k + 3], (c, 4 * c)) * lstm_scale,
            "b": lstm_b,
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
        vocab = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab)
        flat_targets = targets.reshape(-1)
        loss = _cross_entropy(flat_logits, flat_targets)
    return logits, loss


def _cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean token cross-entropy; targets equal to -1 are ignored."""
    valid = targets != -1
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


def _cfg_cache_key(cfg: ModelConfig) -> tuple[Any, ...]:
    return (
        cfg.n_layer,
        cfg.n_head,
        cfg.n_embd,
        cfg.block_size,
        cfg.vocab_size,
        cfg.bias,
        cfg.v_th,
        cfg.leak,
        cfg.v_minus,
        cfg.v_plus,
        cfg.alpha,
        cfg.beta,
    )


@lru_cache(maxsize=8)
def _bind_forward_logits_cached(key: tuple[Any, ...]) -> LogitsFn:
    (
        n_layer,
        n_head,
        n_embd,
        block_size,
        vocab_size,
        bias,
        v_th,
        leak,
        v_minus,
        v_plus,
        alpha,
        beta,
    ) = key
    cfg = ModelConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=vocab_size,
        bias=bias,
        v_th=v_th,
        leak=leak,
        v_minus=v_minus,
        v_plus=v_plus,
        alpha=alpha,
        beta=beta,
    )

    @jax.jit
    def forward_logits(params: Params, idx: jax.Array) -> jax.Array:
        logits, _ = forward(params, idx, cfg, targets=None)
        return logits

    return forward_logits


def bind_forward_logits(cfg: ModelConfig) -> LogitsFn:
    """Return a cached JIT logits function closed over ``cfg``."""
    return _bind_forward_logits_cached(_cfg_cache_key(cfg))


def bind_loss_fn(cfg: ModelConfig) -> Callable[[Params, jax.Array, jax.Array], jax.Array]:
    """Return a JIT loss function closed over ``cfg``."""

    @jax.jit
    def _loss(params: Params, idx: jax.Array, targets: jax.Array) -> jax.Array:
        return loss_fn(params, idx, targets, cfg)

    return _loss


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
