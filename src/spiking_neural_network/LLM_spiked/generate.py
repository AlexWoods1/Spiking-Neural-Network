"""Checkpoint loading and autoregressive generation for SpikedLM."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError("Install jax to use LLM_spiked.generate") from exc

from spiking_neural_network.LLM_spiked.config import ModelConfig
from spiking_neural_network.LLM_spiked.data import CharTokenizer, load_tokenizer
from spiking_neural_network.LLM_spiked.tokenizer import ByteBPETokenizer
from spiking_neural_network.LLM_spiked.model import (
    Params,
    bind_generate_fns,
    right_pad_block,
)

TokenizerLike = CharTokenizer | ByteBPETokenizer


def weights_checkpoint_path(path: Path) -> Path:
    """Return the params-only sibling path for ``ckpt_N.pkl``."""
    path = Path(path)
    if path.name.endswith("_weights.pkl"):
        return path
    return path.with_name(f"{path.stem}_weights.pkl")


def resolve_checkpoint_path(chkpt_path: Path) -> Path:
    """Prefer a light ``*_weights.pkl`` sibling when present."""
    chkpt_path = Path(chkpt_path)
    weights = weights_checkpoint_path(chkpt_path)
    if weights.is_file():
        return weights
    if chkpt_path.name.endswith("_weights.pkl") and chkpt_path.is_file():
        return chkpt_path
    if not chkpt_path.is_file() and weights.is_file():
        return weights
    return chkpt_path


def load_checkpoint(
    chkpt_path: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
) -> tuple[Params, ModelConfig, TokenizerLike]:
    """Load params, model config, and tokenizer from a pickle checkpoint."""
    chkpt_path = resolve_checkpoint_path(Path(chkpt_path))
    with chkpt_path.open("rb") as f:
        chkpt = pickle.load(f)

    model_raw = dict(chkpt["config"]["model"])
    # * Older checkpoints omit Nord-style training knobs; ModelConfig defaults apply.
    model_cfg = ModelConfig(**model_raw)
    params: Params = jax_tree_to_jnp(chkpt["params"])

    tok_path = Path(tokenizer_path or chkpt.get("tokenizer_path", ""))
    if tok_path.is_file():
        tok: TokenizerLike = load_tokenizer(tok_path)
    elif chkpt.get("tokenizer_kind") == "bpe" or "merges" in chkpt:
        tok = ByteBPETokenizer()
        tok.merges = [tuple(pair) for pair in chkpt["merges"]]
    elif "chars" in chkpt:
        tok = CharTokenizer(list(chkpt["chars"]))
    else:
        raise FileNotFoundError(
            f"Tokenizer not found: {tok_path}. Pass --tokenizer-path or "
            "ensure the checkpoint stores chars/merges."
        )
    return params, model_cfg, tok


def jax_tree_to_jnp(tree: Params) -> Params:
    """Convert numpy leaves in a pytree to JAX arrays on the default device."""
    return jax.tree.map(
        lambda x: jax.device_put(jnp.asarray(x)) if x is not None else None,
        tree,
    )


def apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Mask logits below the top-k threshold to ``-inf``."""
    k = min(top_k, logits.shape[-1])
    thresh = np.partition(logits, -k)[-k]
    return np.where(logits < thresh, -np.inf, logits)


def _sample_token(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int | None,
    rng: np.random.Generator,
) -> int:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        logits = apply_top_k(logits, top_k)
    logits = logits - np.nanmax(logits)
    probs = np.exp(np.where(np.isfinite(logits), logits, -1e10))
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(
    params: Params,
    tok: TokenizerLike,
    model_cfg: ModelConfig,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    seed: int = 0,
) -> str:
    """Autoregressive sampling with KV cache + LSTM carry.

    Prefills the prompt once, then decodes one token at a time without
    rescanning the full window (until the cache fills and must slide).
    """
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt) if prompt else tok.encode("\n")
    if not ids:
        ids = tok.encode("\n")
    block_size = model_cfg.block_size
    prefill_fn, decode_fn = bind_generate_fns(model_cfg)

    def _prefill_from_ids(token_ids: list[int]) -> tuple[np.ndarray, Any, int]:
        padded, length = right_pad_block(token_ids, block_size)
        tokens = jax.device_put(jnp.asarray(padded[None, :], dtype=jnp.int32))
        logits, cache = prefill_fn(params, tokens, int(length))
        return np.asarray(logits[0]), cache, length - 1

    logits, cache, t = _prefill_from_ids(ids)
    for _ in range(max_tokens):
        next_id = _sample_token(
            logits, temperature=temperature, top_k=top_k, rng=rng
        )
        ids.append(next_id)
        if t + 1 < block_size:
            t = t + 1
            logits_j, cache = decode_fn(
                params,
                jax.device_put(jnp.asarray([next_id], dtype=jnp.int32)),
                cache,
                jnp.asarray(t, dtype=jnp.int32),
            )
            logits = np.asarray(logits_j[0])
        else:
            # * Sliding window: rebuild cache from the last block_size tokens.
            logits, cache, t = _prefill_from_ids(ids)

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
