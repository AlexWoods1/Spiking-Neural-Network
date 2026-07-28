"""Optax training loop for SpikedLM."""

from __future__ import annotations

import math
import pickle
from dataclasses import asdict
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
from spiking_neural_network.LLM_spiked.model import (
    Params,
    bind_forward_logits,
    bind_loss_fn,
    count_parameters,
    init_params,
    left_pad_block,
)
from spiking_neural_network.LLM_spiked.generate import weights_checkpoint_path


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


def estimate_loss(
    params: Params,
    data: CharDataset,
    cfg: Config,
    *,
    loss_jit: Any | None = None,
) -> dict[str, float]:
    """Average CE over ``eval_batches`` random batches per split (JIT loss)."""
    loss_jit = loss_jit or bind_loss_fn(cfg.model)
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
    """Generate a short continuation from a newline prompt (JIT logits)."""
    rng = rng if rng is not None else np.random.default_rng()
    prompt = "\n"
    ids = data.tokenizer.encode(prompt)
    block_size = cfg.model.block_size
    forward_logits = forward_logits or bind_forward_logits(cfg.model)
    warm = left_pad_block(ids, block_size)
    _ = forward_logits(
        params, jax.device_put(jnp.asarray(warm[None, :], dtype=jnp.int32))
    ).block_until_ready()
    for _ in range(max_new_tokens):
        ctx = left_pad_block(ids, block_size)
        idx = jax.device_put(jnp.asarray(ctx[None, :], dtype=jnp.int32))
        logits = forward_logits(params, idx)
        logits_last = np.asarray(logits[0, -1, :], dtype=np.float64)
        logits_last = logits_last / max(temperature, 1e-6)
        logits_last = logits_last - logits_last.max()
        probs = np.exp(logits_last)
        probs = probs / probs.sum()
        next_id = int(rng.choice(len(probs), p=probs))
        ids.append(next_id)
    return data.tokenizer.decode(ids)


def _config_dict_for_checkpoint(cfg: Config) -> dict[str, Any]:
    """Serialize config with paths as strings (portable across OS)."""
    raw = asdict(cfg)
    raw["data"]["data_dir"] = str(raw["data"]["data_dir"])
    raw["paths"]["out_dir"] = str(raw["paths"]["out_dir"])
    raw["paths"]["tokenizer_path"] = str(raw["paths"]["tokenizer_path"])
    return raw


def save_checkpoint(
    path: Path,
    *,
    params: Params,
    opt_state: Any,
    step: int,
    cfg: Config,
    tokenizer: CharTokenizer,
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
        "chars": tokenizer.chars,
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
    sample_logits = bind_forward_logits(cfg.model)

    @jax.jit
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
            losses = estimate_loss(params, data, cfg, loss_jit=loss_jit)
            print(
                f"step {step}: train {losses['train']:.4f} "
                f"val {losses['val']:.4f}"
            )

        if step % cfg.train.sample_interval == 0:
            text = sample_text(
                params,
                data,
                cfg,
                max_new_tokens=40,
                rng=rng_np,
                forward_logits=sample_logits,
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
