"""Optax training loop for SpikedLM."""

from __future__ import annotations

import math
import pickle
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError as exc:
    raise ImportError("Install jax and optax to train SpikedLM") from exc

from spiking_neural_network.LLM_spiked.config import Config, TrainConfig, load_config
from spiking_neural_network.LLM_spiked.data import CharDataset, CharTokenizer
from spiking_neural_network.LLM_spiked.generate import generate, weights_checkpoint_path
from spiking_neural_network.LLM_spiked.model import (
    Params,
    bind_ce_loss_fn,
    bind_loss_fn,
    bind_spike_rates_fn,
    count_parameters,
    init_params,
)
from spiking_neural_network.LLM_spiked.tokenizer import ByteBPETokenizer


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to 10% of peak LR."""
    min_lr = 0.1 * cfg.learning_rate
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    if step >= cfg.max_steps:
        return min_lr
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (cfg.learning_rate - min_lr)


def configure_runtime() -> None:
    """Enable GPU-friendly XLA defaults when available."""
    backend = jax.default_backend()
    print(f"jax backend={backend} devices={jax.devices()}")
    if backend == "gpu":
        try:
            # * Tensor-core matmuls; params stay fp32 via Optax updates.
            jax.config.update("jax_default_matmul_precision", "bfloat16")
            print("matmul precision=bfloat16")
        except Exception as exc:  # noqa: BLE001
            print(f"bf16 matmul not applied: {exc}")


def estimate_loss(
    params: Params,
    data: CharDataset,
    cfg: Config,
    *,
    loss_jit: Any | None = None,
) -> dict[str, float]:
    """Average CE over ``eval_batches`` random batches per split (JIT loss)."""
    loss_jit = loss_jit or bind_ce_loss_fn(cfg.model)
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses: list[float] = []
        for _ in range(cfg.train.eval_batches):
            xb, yb = data.get_batch(
                split, cfg.train.batch_size, cfg.model.block_size
            )
            loss = loss_jit(
                params,
                jax.device_put(jnp.asarray(xb)),
                jax.device_put(jnp.asarray(yb)),
            )
            losses.append(float(loss))
        out[split] = float(np.mean(losses))
    return out


def estimate_spike_rates(
    params: Params,
    data: CharDataset,
    cfg: Config,
    *,
    rates_jit: Any | None = None,
) -> dict[str, float]:
    """Mean gate spike rates over one train batch."""
    rates_jit = rates_jit or bind_spike_rates_fn(cfg.model)
    xb, _yb = data.get_batch(
        "train", cfg.train.batch_size, cfg.model.block_size
    )
    rate_i, rate_f, rate_o = rates_jit(
        params, jax.device_put(jnp.asarray(xb))
    )
    return {
        "rate_i": float(rate_i),
        "rate_f": float(rate_f),
        "rate_o": float(rate_o),
    }


def sample_text(
    params: Params,
    data: CharDataset,
    cfg: Config,
    *,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    rng: np.random.Generator | None = None,
    forward_logits: Any | None = None,
) -> str:
    """Generate a short continuation via KV-cached decode."""
    # * forward_logits kept for call-site compatibility; unused after KV path.
    del forward_logits
    rng = rng if rng is not None else np.random.default_rng()
    seed = int(rng.integers(0, 2**31 - 1))
    return generate(
        params,
        data.tokenizer,
        cfg.model,
        "\n",
        max_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )


def _config_dict_for_checkpoint(cfg: Config) -> dict[str, Any]:
    """Serialize config with paths as strings (portable across OS)."""
    raw = asdict(cfg)
    raw["data"]["data_dir"] = str(raw["data"]["data_dir"])
    raw["paths"]["out_dir"] = str(raw["paths"]["out_dir"])
    raw["paths"]["tokenizer_path"] = str(raw["paths"]["tokenizer_path"])
    return raw


def _tokenizer_checkpoint_fields(
    tokenizer: CharTokenizer | ByteBPETokenizer,
) -> dict[str, Any]:
    """Embed tokenizer payload so generate works without the JSON file."""
    if isinstance(tokenizer, CharTokenizer):
        return {"tokenizer_kind": "char", "chars": tokenizer.chars}
    return {
        "tokenizer_kind": "bpe",
        "merges": [list(pair) for pair in tokenizer.merges],
    }


def save_checkpoint(
    path: Path,
    *,
    params: Params,
    opt_state: Any,
    step: int,
    cfg: Config,
    tokenizer: CharTokenizer | ByteBPETokenizer,
    save_optimizer: bool = True,
) -> None:
    """Write checkpoint pickles.

    Always writes a light ``*_weights.pkl`` (params + config + vocab) for fast
    generate loads. Optionally also writes the full checkpoint with optimizer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    params_np = jax.tree.map(np.asarray, params)
    meta = {
        "params": params_np,
        "step": step,
        "config": _config_dict_for_checkpoint(cfg),
        "tokenizer_path": str(cfg.paths.tokenizer_path),
        **_tokenizer_checkpoint_fields(tokenizer),
    }
    weights_path = weights_checkpoint_path(path)
    with weights_path.open("wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote {weights_path}")

    if save_optimizer:
        payload = {**meta, "opt_state": opt_state}
        with path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"Wrote {path}")


def train(config_path: str | Path = "configs/llm_smoke.yaml") -> Params:
    """Run the SpikedLM training loop from a YAML config.

    Args:
        config_path: Path to ``llm_smoke.yaml`` (or similar).

    Returns:
        Trained parameter pytree.
    """
    cfg = load_config(config_path)
    configure_runtime()
    rng_np = np.random.default_rng(cfg.train.seed)
    data = CharDataset(
        cfg.data.data_dir,
        cfg.paths.tokenizer_path,
        rng=rng_np,
    )
    cfg.model.vocab_size = data.tokenizer.vocab_size

    key = jax.random.PRNGKey(cfg.train.seed)
    params = init_params(cfg.model, key)
    n_params = count_parameters(params)
    print(f"params={n_params:,} vocab={cfg.model.vocab_size}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.train.grad_clip),
        optax.adamw(
            learning_rate=1.0,  # * Scaled per step via optax.scale below.
            b1=cfg.train.beta1,
            b2=cfg.train.beta2,
            weight_decay=cfg.train.weight_decay,
        ),
    )
    opt_state = optimizer.init(params)
    loss_jit = bind_loss_fn(cfg.model)
    ce_jit = bind_ce_loss_fn(cfg.model)
    rates_jit = bind_spike_rates_fn(cfg.model)

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(
        params: Params,
        opt_state: Any,
        xb: jax.Array,
        yb: jax.Array,
        lr: jax.Array,
    ) -> tuple[Params, Any, jax.Array]:
        def _loss(p: Params) -> jax.Array:
            return loss_jit(p, xb, yb)

        loss, grads = jax.value_and_grad(_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = jax.tree.map(lambda u: u * lr, updates)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    cfg.paths.out_dir.mkdir(parents=True, exist_ok=True)

    # * Prefetch first batch onto device.
    xb_np, yb_np = data.get_batch(
        "train", cfg.train.batch_size, cfg.model.block_size
    )
    xb = jax.device_put(jnp.asarray(xb_np))
    yb = jax.device_put(jnp.asarray(yb_np))

    pbar = tqdm(range(1, cfg.train.max_steps + 1), desc="train")
    for step in pbar:
        lr = get_lr(step, cfg.train)
        # * Kick off host→device copy for the *next* step while this step runs.
        next_xb_np, next_yb_np = data.get_batch(
            "train", cfg.train.batch_size, cfg.model.block_size
        )
        next_xb = jax.device_put(jnp.asarray(next_xb_np))
        next_yb = jax.device_put(jnp.asarray(next_yb_np))

        params, opt_state, loss = train_step(
            params,
            opt_state,
            xb,
            yb,
            jnp.asarray(lr, dtype=jnp.float32),
        )
        pbar.set_postfix(loss=float(loss), lr=lr)
        xb, yb = next_xb, next_yb

        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps:
            losses = estimate_loss(params, data, cfg, loss_jit=ce_jit)
            rates = estimate_spike_rates(params, data, cfg, rates_jit=rates_jit)
            print(
                f"step {step}: train {losses['train']:.4f} "
                f"val {losses['val']:.4f} "
                f"rate_i={rates['rate_i']:.3f} "
                f"rate_f={rates['rate_f']:.3f} "
                f"rate_o={rates['rate_o']:.3f}"
            )

        if step % cfg.train.sample_interval == 0:
            text = sample_text(
                params,
                data,
                cfg,
                max_new_tokens=40,
                rng=rng_np,
            )
            print(f"--- sample @ {step} ---\n{text}\n---------------")

        if step % cfg.train.checkpoint_interval == 0 or step == cfg.train.max_steps:
            # * Full opt-state ckpt only at the end; weights every interval.
            save_checkpoint(
                cfg.paths.out_dir / f"ckpt_{step}.pkl",
                params=params,
                opt_state=opt_state,
                step=step,
                cfg=cfg,
                tokenizer=data.tokenizer,
                save_optimizer=(step == cfg.train.max_steps),
            )

    return params
