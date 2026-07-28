"""Checkpoint loading and autoregressive generation for SpikedLM."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

try:
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError("Install jax to use LLM_spiked.generate") from exc

from spiking_neural_network.LLM_spiked.config import ModelConfig
from spiking_neural_network.LLM_spiked.data import CharTokenizer
from spiking_neural_network.LLM_spiked.model import (
    Params,
    bind_forward_logits,
    left_pad_block,
)


def load_checkpoint(
    chkpt_path: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
) -> tuple[Params, ModelConfig, CharTokenizer]:
    """Load params, model config, and tokenizer from a pickle checkpoint.

    Args:
        chkpt_path: Path to ``ckpt_*.pkl``.
        tokenizer_path: Optional override for the tokenizer JSON path.

    Returns:
        ``(params, model_config, tokenizer)``.
    """
    chkpt_path = Path(chkpt_path)
    with chkpt_path.open("rb") as f:
        chkpt = pickle.load(f)

    model_cfg = ModelConfig(**chkpt["config"]["model"])
    params: Params = jax_tree_to_jnp(chkpt["params"])

    tok_path = Path(tokenizer_path or chkpt.get("tokenizer_path", ""))
    if tok_path.is_file():
        tok = CharTokenizer.load(tok_path)
    elif "chars" in chkpt:
        tok = CharTokenizer(list(chkpt["chars"]))
    else:
        raise FileNotFoundError(
            f"Tokenizer not found: {tok_path}. Pass --tokenizer-path or "
            "ensure the checkpoint stores chars."
        )
    return params, model_cfg, tok


def jax_tree_to_jnp(tree: Params) -> Params:
    """Convert numpy leaves in a pytree to JAX arrays."""
    import jax

    return jax.tree.map(
        lambda x: jnp.asarray(x) if x is not None else None,
        tree,
    )


def apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Mask logits below the top-k threshold to ``-inf``."""
    k = min(top_k, logits.shape[-1])
    thresh = np.partition(logits, -k)[-k]
    return np.where(logits < thresh, -np.inf, logits)


def generate(
    params: Params,
    tok: CharTokenizer,
    model_cfg: ModelConfig,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    seed: int = 0,
) -> str:
    """Autoregressive character sampling from a SpikedLM checkpoint.

    Uses a JIT logits fn with fixed ``block_size`` left-padding so each new
    token reuses one compiled kernel instead of re-tracing every step.

    Args:
        params: Model parameters.
        tok: Character tokenizer.
        model_cfg: Model configuration (block size, spike hypers, …).
        prompt: Seed text.
        max_tokens: Number of new characters to sample.
        temperature: Softmax temperature.
        top_k: Optional top-k filtering.
        seed: Numpy RNG seed.

    Returns:
        Prompt plus generated continuation.
    """
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt) if prompt else tok.encode("\n")
    block_size = model_cfg.block_size
    forward_logits = bind_forward_logits(model_cfg)

    # * Warmup compile once with the fixed (1, block_size) shape.
    warm = left_pad_block(ids, block_size)
    _ = forward_logits(params, jnp.asarray(warm[None, :], dtype=jnp.int32)).block_until_ready()

    for _ in range(max_tokens):
        ctx = left_pad_block(ids, block_size)
        idx = jnp.asarray(ctx[None, :], dtype=jnp.int32)
        logits = forward_logits(params, idx)
        logits_last = np.asarray(logits[0, -1, :], dtype=np.float64)
        logits_last = logits_last / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            logits_last = apply_top_k(logits_last, top_k)
        logits_last = logits_last - np.nanmax(logits_last)
        # * Replace -inf after top-k so exp stays well-behaved.
        probs = np.exp(np.where(np.isfinite(logits_last), logits_last, -1e10))
        probs = probs / probs.sum()
        next_id = int(rng.choice(len(probs), p=probs))
        ids.append(next_id)
    return tok.decode(ids)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate text with SpikedLM.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", type=str, default="ROMEO:")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--tokenizer-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    params, model_cfg, tok = load_checkpoint(
        args.checkpoint, tokenizer_path=args.tokenizer_path
    )
    text = generate(
        params,
        tok,
        model_cfg,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
