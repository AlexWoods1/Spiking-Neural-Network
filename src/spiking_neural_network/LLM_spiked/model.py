"""JAX SpikedLM: causal attention + Spiking LSTM decoder for next-token LM."""

from __future__ import annotations

from functools import lru_cache, partial
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
# * Mean spike rates for gates i, f, o.
SpikeRates = tuple[jax.Array, jax.Array, jax.Array]


def _leaky_clamp(x: jax.Array, slope: float) -> jax.Array:
    """Pass positives unchanged; leak negatives with slope in ``[0, 1]``."""
    return jnp.where(x >= 0.0, x, slope * x)


def _softplus_inv(y: float) -> float:
    """Inverse softplus for initializing ``v_th_raw`` from a positive target."""
    # * softplus(x) = log(1+exp(x)); for y>0 use log(expm1(y)).
    if y <= 0.0:
        raise ValueError("softplus_inv requires y > 0")
    return float(np.log(np.expm1(y)))


def _resolve_lif(
    lstm_params: Params, cfg: ModelConfig
) -> tuple[jax.Array, jax.Array]:
    """Return ``(leak, v_th)`` from learnable params or config defaults."""
    if cfg.learnable_lif and "leak_logit" in lstm_params:
        leak = jax.nn.sigmoid(lstm_params["leak_logit"])
        v_th = jax.nn.softplus(lstm_params["v_th_raw"]) + 1e-3
        # * Keep threshold inside the AdaLi support window.
        v_th = jnp.clip(v_th, cfg.v_minus + 1e-3, cfg.v_plus - 1e-3)
        return leak, v_th
    return jnp.asarray(cfg.leak), jnp.asarray(cfg.v_th)


def _spike_rate_loss(
    rates: SpikeRates,
    cfg: ModelConfig,
) -> jax.Array:
    """Asymmetric penalty keeping gate rates near targets and above a floor."""
    targets = (
        cfg.target_rate_i,
        cfg.target_rate_f,
        cfg.target_rate_o,
    )
    floor = cfg.rate_floor
    total = jnp.asarray(0.0, dtype=rates[0].dtype)
    for rate, target in zip(rates, targets):
        under = jnp.maximum(floor - rate, 0.0)
        # * Dead gates hurt more than mild overshoot.
        total = total + 2.0 * under * under + (rate - target) ** 2
    return total / 3.0


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
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array],
    jax.Array,
    SpikeRates,
]:
    """One Spiking LSTM step: spike ``i/f/o``, tanh candidate / hidden."""
    h, cell, gate_u = carry
    w = params["W"]
    u_rec = params["U"]
    bias_vec = params.get("b")
    leak, v_th = _resolve_lif(params, cfg)
    drive = x_t @ w + h @ u_rec
    if bias_vec is not None:
        drive = drive + bias_vec
    # * LIF membrane on gate channels; soft-reset after spikes.
    gate_u = leak * gate_u + (1.0 - leak) * drive
    gate_u = _leaky_clamp(gate_u, cfg.leaky_clamp_slope)
    i_pre, f_pre, g_pre, o_pre = jnp.split(gate_u, 4, axis=-1)
    i = spike(i_pre, v_th, cfg.v_minus, cfg.v_plus, cfg.alpha, cfg.beta)
    f = spike(f_pre, v_th, cfg.v_minus, cfg.v_plus, cfg.alpha, cfg.beta)
    o = spike(o_pre, v_th, cfg.v_minus, cfg.v_plus, cfg.alpha, cfg.beta)
    # * Continuous cell write (tanh) — avoids dead binary hidden states.
    g = jnp.tanh(g_pre)
    gate_u = gate_u - v_th * jnp.concatenate(
        [i, f, jnp.zeros_like(g), o], axis=-1
    )
    cell_new = f * (leak * cell) + i * g
    h_new = o * jnp.tanh(cell_new)
    rates = (i.mean(), f.mean(), o.mean())
    return (h_new, cell_new, gate_u), h_new, rates


def _spiking_lstm(
    x: jax.Array,
    params: Params,
    cfg: ModelConfig,
) -> tuple[jax.Array, SpikeRates]:
    """Scan Spiking LSTM over sequence length ``T``; return h and mean rates."""
    b, _t, c = x.shape

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        x_t: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, SpikeRates]]:
        new_carry, h_t, rates = _spiking_lstm_step(carry, x_t, params, cfg)
        return new_carry, (h_t, rates)

    h0 = jnp.zeros((b, c), dtype=x.dtype)
    c0 = jnp.zeros((b, c), dtype=x.dtype)
    gate_u0 = jnp.zeros((b, 4 * c), dtype=x.dtype)
    x_tm = jnp.swapaxes(x, 0, 1)
    # * Unroll cuts scan overhead on GPU for moderate T.
    _, (h_seq, rates_seq) = lax.scan(
        step, (h0, c0, gate_u0), x_tm, unroll=8
    )
    # rates_seq is a tuple of (T,) arrays after scan over pytree leaves.
    mean_rates = (
        rates_seq[0].mean(),
        rates_seq[1].mean(),
        rates_seq[2].mean(),
    )
    return jnp.swapaxes(h_seq, 0, 1), mean_rates


def _block(
    x: jax.Array, params: Params, cfg: ModelConfig
) -> tuple[jax.Array, SpikeRates]:
    """Pre-norm residual: attention then Spiking LSTM."""
    ln1 = params["ln1"]
    x = x + _causal_self_attention(
        _layer_norm(x, ln1["scale"], ln1.get("bias"), eps=1e-5),
        params["attn"],
        n_head=cfg.n_head,
        bias=cfg.bias,
    )
    ln2 = params["ln2"]
    h, rates = _spiking_lstm(
        _layer_norm(x, ln2["scale"], ln2.get("bias"), eps=1e-5),
        params["lstm"],
        cfg,
    )
    x = x + h
    return x, rates


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
        if cfg.learnable_lif:
            # * sigmoid(0)=0.5; softplus(softplus_inv(v_th-eps))+eps ≈ v_th.
            lstm["leak_logit"] = jnp.asarray(0.0)
            lstm["v_th_raw"] = jnp.asarray(
                _softplus_inv(max(cfg.v_th - 1e-3, 1e-3))
            )
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
) -> tuple[jax.Array, jax.Array | None, SpikeRates | None]:
    """Forward pass returning logits, optional CE loss, and optional spike rates.

    Args:
        params: Model parameters.
        idx: Token ids with shape ``(B, T)``.
        cfg: Model configuration.
        targets: Optional target ids ``(B, T)`` for cross-entropy.

    Returns:
        ``(logits, loss, rates)`` where ``logits`` has shape ``(B, T, V)``,
        ``loss`` is CE (no rate aux) or ``None``, and ``rates`` is mean
        ``(i, f, o)`` over layers when computed.
    """
    _, t = idx.shape
    if t > cfg.block_size:
        raise ValueError(
            f"Cannot forward sequence of length {t}, "
            f"block size is only {cfg.block_size}"
        )
    pos = jnp.arange(t)
    x = params["wte"][idx] + params["wpe"][pos]
    rate_i = jnp.asarray(0.0, dtype=x.dtype)
    rate_f = jnp.asarray(0.0, dtype=x.dtype)
    rate_o = jnp.asarray(0.0, dtype=x.dtype)
    n_blocks = 0
    for block_params in params["blocks"]:
        x, rates = _block(x, block_params, cfg)
        rate_i = rate_i + rates[0]
        rate_f = rate_f + rates[1]
        rate_o = rate_o + rates[2]
        n_blocks += 1
    mean_rates: SpikeRates = (
        rate_i / max(n_blocks, 1),
        rate_f / max(n_blocks, 1),
        rate_o / max(n_blocks, 1),
    )
    x = _layer_norm(x, params["ln_f"]["scale"], params["ln_f"].get("bias"))
    # * Weight tying: LM head shares wte.
    logits = x @ params["wte"].T
    loss = None
    if targets is not None:
        vocab = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab)
        flat_targets = targets.reshape(-1)
        loss = _cross_entropy(flat_logits, flat_targets)
    return logits, loss, mean_rates


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
    """Scalar training loss (CE + optional spike-rate aux) for ``value_and_grad``."""
    _, ce, rates = forward(params, idx, cfg, targets)
    assert ce is not None
    assert rates is not None
    if cfg.spike_rate_weight <= 0.0:
        return ce
    return ce + cfg.spike_rate_weight * _spike_rate_loss(rates, cfg)


def ce_loss_fn(
    params: Params,
    idx: jax.Array,
    targets: jax.Array,
    cfg: ModelConfig,
) -> jax.Array:
    """CE-only loss for evaluation (ignores spike-rate aux weight)."""
    _, ce, _ = forward(params, idx, cfg, targets)
    assert ce is not None
    return ce


def spike_rates_fn(
    params: Params,
    idx: jax.Array,
    cfg: ModelConfig,
) -> SpikeRates:
    """Mean gate spike rates ``(i, f, o)`` for a batch (no targets)."""
    _, _, rates = forward(params, idx, cfg, targets=None)
    assert rates is not None
    return rates


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
        cfg.learnable_lif,
        cfg.leaky_clamp_slope,
        cfg.spike_rate_weight,
        cfg.target_rate_i,
        cfg.target_rate_f,
        cfg.target_rate_o,
        cfg.rate_floor,
    )


@lru_cache(maxsize=8)
def _bind_forward_logits_cached(key: tuple[Any, ...]) -> LogitsFn:
    cfg = ModelConfig(
        n_layer=key[0],
        n_head=key[1],
        n_embd=key[2],
        block_size=key[3],
        vocab_size=key[4],
        bias=key[5],
        v_th=key[6],
        leak=key[7],
        v_minus=key[8],
        v_plus=key[9],
        alpha=key[10],
        beta=key[11],
        learnable_lif=key[12],
        leaky_clamp_slope=key[13],
        spike_rate_weight=key[14],
        target_rate_i=key[15],
        target_rate_f=key[16],
        target_rate_o=key[17],
        rate_floor=key[18],
    )

    @jax.jit
    def forward_logits(params: Params, idx: jax.Array) -> jax.Array:
        logits, _, _ = forward(params, idx, cfg, targets=None)
        return logits

    return forward_logits


def bind_forward_logits(cfg: ModelConfig) -> LogitsFn:
    """Return a cached JIT logits function closed over ``cfg``."""
    return _bind_forward_logits_cached(_cfg_cache_key(cfg))


def bind_loss_fn(cfg: ModelConfig) -> Callable[[Params, jax.Array, jax.Array], jax.Array]:
    """Return a JIT training loss (CE + rate aux) closed over ``cfg``."""

    @jax.jit
    def _loss(params: Params, idx: jax.Array, targets: jax.Array) -> jax.Array:
        return loss_fn(params, idx, targets, cfg)

    return _loss


def bind_ce_loss_fn(
    cfg: ModelConfig,
) -> Callable[[Params, jax.Array, jax.Array], jax.Array]:
    """Return a JIT CE-only loss for eval logging."""

    @jax.jit
    def _loss(params: Params, idx: jax.Array, targets: jax.Array) -> jax.Array:
        return ce_loss_fn(params, idx, targets, cfg)

    return _loss


def bind_spike_rates_fn(
    cfg: ModelConfig,
) -> Callable[[Params, jax.Array], SpikeRates]:
    """Return a JIT spike-rate probe closed over ``cfg``."""

    @jax.jit
    def _rates(params: Params, idx: jax.Array) -> SpikeRates:
        return spike_rates_fn(params, idx, cfg)

    return _rates

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


def right_pad_block(token_ids: list[int], block_size: int) -> tuple[np.ndarray, int]:
    """Right-pad / crop to ``block_size``; returns ``(tokens, length)`` left-aligned."""
    ctx = token_ids[-block_size:]
    length = len(ctx)
    out = np.zeros((block_size,), dtype=np.int32)
    out[:length] = np.asarray(ctx, dtype=np.int32)
    return out, length


def _empty_layer_cache(
    cfg: ModelConfig, batch_size: int, dtype: jnp.dtype
) -> dict[str, jax.Array]:
    head_dim = cfg.n_embd // cfg.n_head
    return {
        "k": jnp.zeros(
            (batch_size, cfg.block_size, cfg.n_head, head_dim), dtype=dtype
        ),
        "v": jnp.zeros(
            (batch_size, cfg.block_size, cfg.n_head, head_dim), dtype=dtype
        ),
        "h": jnp.zeros((batch_size, cfg.n_embd), dtype=dtype),
        "cell": jnp.zeros((batch_size, cfg.n_embd), dtype=dtype),
        "gate_u": jnp.zeros((batch_size, 4 * cfg.n_embd), dtype=dtype),
    }


def init_kv_cache(
    cfg: ModelConfig, batch_size: int = 1, *, dtype: jnp.dtype = jnp.float32
) -> list[dict[str, jax.Array]]:
    """Allocate empty per-layer KV + LSTM caches for incremental decode."""
    return [_empty_layer_cache(cfg, batch_size, dtype) for _ in range(cfg.n_layer)]


def _qkv_proj(
    x: jax.Array,
    params: Params,
    *,
    n_head: int,
    bias: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Project ``x`` to Q,K,V in BTNH layout."""
    b, t, c = x.shape
    head_dim = c // n_head
    qkv = x @ params["c_attn"]
    if bias and params.get("c_attn_bias") is not None:
        qkv = qkv + params["c_attn_bias"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(b, t, n_head, head_dim)
    k = k.reshape(b, t, n_head, head_dim)
    v = v.reshape(b, t, n_head, head_dim)
    return q, k, v


def _attn_from_qkv(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    params: Params,
    *,
    bias: bool,
    is_causal: bool,
    key_len: int | None = None,
) -> jax.Array:
    """Attention output from BTNH tensors; optional prefix key length mask."""
    b, t, n_head, head_dim = q.shape
    sdpa = getattr(jax.nn, "dot_product_attention", None)
    if sdpa is not None and key_len is None:
        y = sdpa(q, k, v, is_causal=is_causal)
    else:
        q_t = q.transpose(0, 2, 1, 3)
        k_t = k.transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)
        att = (q_t @ jnp.swapaxes(k_t, -2, -1)) * (1.0 / jnp.sqrt(head_dim))
        s = k.shape[1]
        if is_causal and t == s:
            mask = jnp.tril(jnp.ones((t, s), dtype=q.dtype))
            att = jnp.where(mask[None, None, :, :] == 0, jnp.asarray(-1e10, dtype=q.dtype), att)
        if key_len is not None:
            valid = jnp.arange(s)[None, None, None, :] < key_len
            att = jnp.where(valid, att, jnp.asarray(-1e10, dtype=q.dtype))
        att = jax.nn.softmax(att, axis=-1)
        y = (att @ v_t).transpose(0, 2, 1, 3)
    y = y.reshape(b, t, n_head * head_dim)
    y = y @ params["c_proj"]
    if bias and params.get("c_proj_bias") is not None:
        y = y + params["c_proj_bias"]
    return y


def _attn_prefill(
    x: jax.Array,
    params: Params,
    *,
    n_head: int,
    bias: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Full-sequence attention; returns ``(out, k, v)``."""
    q, k, v = _qkv_proj(x, params, n_head=n_head, bias=bias)
    y = _attn_from_qkv(q, k, v, params, bias=bias, is_causal=True)
    return y, k, v


def _attn_decode(
    x_t: jax.Array,
    params: Params,
    layer_cache: dict[str, jax.Array],
    *,
    n_head: int,
    bias: bool,
    t: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Single-token attention against KV cache; writes slot ``t``."""
    q, k_new, v_new = _qkv_proj(x_t, params, n_head=n_head, bias=bias)
    k = layer_cache["k"].at[:, t].set(k_new[:, 0])
    v = layer_cache["v"].at[:, t].set(v_new[:, 0])
    # * Attend only to keys 0..t (inclusive); t is a traced 0-d int.
    key_len = t + 1
    y = _attn_from_qkv(
        q, k, v, params, bias=bias, is_causal=False, key_len=key_len
    )
    return y, {**layer_cache, "k": k, "v": v}


def _lstm_prefill(
    x: jax.Array,
    params: Params,
    cfg: ModelConfig,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """LSTM over ``(B, T, C)``; returns hidden sequence and final carry."""
    b, _t, c = x.shape

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        x_t: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        new_carry, h_t, _rates = _spiking_lstm_step(carry, x_t, params, cfg)
        return new_carry, h_t

    carry0 = (
        jnp.zeros((b, c), dtype=x.dtype),
        jnp.zeros((b, c), dtype=x.dtype),
        jnp.zeros((b, 4 * c), dtype=x.dtype),
    )
    carry, h_seq = lax.scan(step, carry0, jnp.swapaxes(x, 0, 1), unroll=8)
    return jnp.swapaxes(h_seq, 0, 1), carry

def _block_prefill(
    x: jax.Array,
    block_params: Params,
    cfg: ModelConfig,
    layer_cache: dict[str, jax.Array],
    length: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Prefill one block; fill KV slots ``0..length-1`` and LSTM carry."""
    ln1 = block_params["ln1"]
    a_in = _layer_norm(x, ln1["scale"], ln1.get("bias"))
    a_out, k, v = _attn_prefill(
        a_in, block_params["attn"], n_head=cfg.n_head, bias=cfg.bias
    )
    x = x + a_out
    layer_cache = {
        **layer_cache,
        "k": layer_cache["k"].at[:, :length].set(k),
        "v": layer_cache["v"].at[:, :length].set(v),
    }
    ln2 = block_params["ln2"]
    h_seq, (h, cell, gate_u) = _lstm_prefill(
        _layer_norm(x, ln2["scale"], ln2.get("bias")),
        block_params["lstm"],
        cfg,
    )
    x = x + h_seq
    layer_cache = {
        **layer_cache,
        "h": h,
        "cell": cell,
        "gate_u": gate_u,
    }
    return x, layer_cache


def _block_decode(
    x_t: jax.Array,
    block_params: Params,
    cfg: ModelConfig,
    layer_cache: dict[str, jax.Array],
    t: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Decode one token at cache index ``t``."""
    ln1 = block_params["ln1"]
    a_out, layer_cache = _attn_decode(
        _layer_norm(x_t, ln1["scale"], ln1.get("bias")),
        block_params["attn"],
        layer_cache,
        n_head=cfg.n_head,
        bias=cfg.bias,
        t=t,
    )
    x_t = x_t + a_out
    ln2 = block_params["ln2"]
    carry = (layer_cache["h"], layer_cache["cell"], layer_cache["gate_u"])
    x_lstm = _layer_norm(x_t, ln2["scale"], ln2.get("bias"))[:, 0, :]
    carry, h_t, _rates = _spiking_lstm_step(
        carry, x_lstm, block_params["lstm"], cfg
    )
    x_t = x_t + h_t[:, None, :]
    layer_cache = {
        **layer_cache,
        "h": carry[0],
        "cell": carry[1],
        "gate_u": carry[2],
    }
    return x_t, layer_cache


def prefill(
    params: Params,
    tokens: jax.Array,
    cfg: ModelConfig,
    length: int,
) -> tuple[jax.Array, list[dict[str, jax.Array]]]:
    """Run prompt prefill with KV/LSTM cache.

    Args:
        params: Model parameters.
        tokens: Right-padded token ids ``(B, block_size)``.
        cfg: Model config.
        length: Number of valid leading tokens (``1..block_size``).

    Returns:
        ``(logits_last, cache)`` where ``logits_last`` is ``(B, V)``.
    """
    tokens = tokens[:, :length]
    pos = jnp.arange(length)
    x = params["wte"][tokens] + params["wpe"][pos]
    cache = init_kv_cache(cfg, batch_size=int(tokens.shape[0]), dtype=x.dtype)
    for i, block_params in enumerate(params["blocks"]):
        x, cache[i] = _block_prefill(x, block_params, cfg, cache[i], length)
    x = _layer_norm(x, params["ln_f"]["scale"], params["ln_f"].get("bias"))
    logits = x @ params["wte"].T
    return logits[:, -1, :], cache


def decode_step(
    params: Params,
    token_id: jax.Array,
    cache: list[dict[str, jax.Array]],
    cfg: ModelConfig,
    t: jax.Array,
) -> tuple[jax.Array, list[dict[str, jax.Array]]]:
    """Single-token decode into cache slot ``t``.

    Args:
        params: Model parameters.
        token_id: New token ids ``(B,)``.
        cache: Per-layer KV + LSTM state from prefill/prior steps.
        cfg: Model config.
        t: Absolute position / cache index (``0-d int32``; traced, not static).

    Returns:
        ``(logits, cache)`` with ``logits`` shape ``(B, V)``.
    """
    b = token_id.shape[0]
    x = params["wte"][token_id] + params["wpe"][t]
    x = x.reshape(b, 1, -1)
    new_cache: list[dict[str, jax.Array]] = []
    for block_params, layer_cache in zip(params["blocks"], cache):
        x, layer_cache = _block_decode(x, block_params, cfg, layer_cache, t)
        new_cache.append(layer_cache)
    x = _layer_norm(x, params["ln_f"]["scale"], params["ln_f"].get("bias"))
    logits = (x @ params["wte"].T)[:, 0, :]
    return logits, new_cache


GenerateFns = tuple[
    Callable[[Params, jax.Array, int], tuple[jax.Array, list[dict[str, jax.Array]]]],
    Callable[
        [Params, jax.Array, list[dict[str, jax.Array]], jax.Array],
        tuple[jax.Array, list[dict[str, jax.Array]]],
    ],
]


@lru_cache(maxsize=8)
def _bind_generate_fns_cached(key: tuple[Any, ...]) -> GenerateFns:
    cfg = ModelConfig(
        n_layer=key[0],
        n_head=key[1],
        n_embd=key[2],
        block_size=key[3],
        vocab_size=key[4],
        bias=key[5],
        v_th=key[6],
        leak=key[7],
        v_minus=key[8],
        v_plus=key[9],
        alpha=key[10],
        beta=key[11],
        learnable_lif=key[12],
        leaky_clamp_slope=key[13],
        spike_rate_weight=key[14],
        target_rate_i=key[15],
        target_rate_f=key[16],
        target_rate_o=key[17],
        rate_floor=key[18],
    )

    @partial(jax.jit, static_argnames=("length",))
    def prefill_jit(
        params: Params, tokens: jax.Array, length: int
    ) -> tuple[jax.Array, list[dict[str, jax.Array]]]:
        return prefill(params, tokens, cfg, length)

    @jax.jit
    def decode_jit(
        params: Params,
        token_id: jax.Array,
        cache: list[dict[str, jax.Array]],
        t: jax.Array,
    ) -> tuple[jax.Array, list[dict[str, jax.Array]]]:
        return decode_step(params, token_id, cache, cfg, t)

    return prefill_jit, decode_jit


def bind_generate_fns(cfg: ModelConfig) -> GenerateFns:
    """Return JIT ``(prefill, decode_step)`` closed over ``cfg``."""
    return _bind_generate_fns_cached(_cfg_cache_key(cfg))
