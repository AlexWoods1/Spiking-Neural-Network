"""Shakespeare dataset and batching for SpikedLM (char or byte-BPE)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from spiking_neural_network.LLM_spiked.tokenizer import ByteBPETokenizer

__all__ = [
    "ByteBPETokenizer",
    "CharDataset",
    "CharTokenizer",
    "TokenizerProtocol",
    "load_tokenizer",
]


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Minimal encode/decode interface used by train and generate."""

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def save(self, path: Path | str) -> None: ...


class CharTokenizer:
    """Character-level tokenizer with a fixed stoi/itos vocabulary."""

    def __init__(self, chars: list[str]) -> None:
        if len(chars) < 2:
            raise ValueError("chars must contain at least 2 characters")
        if len(set(chars)) != len(chars):
            raise ValueError("chars must be unique")
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids.

        Args:
            text: Input string.

        Returns:
            List of integer token ids.

        Raises:
            KeyError: If ``text`` contains a character outside the vocabulary.
        """
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token ids to a string."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path | str) -> None:
        """Persist vocabulary as JSON ``{"chars": [...]}``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> CharTokenizer:
        """Load a tokenizer from JSON."""
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "chars" not in raw:
            raise ValueError(f"Invalid tokenizer file: {path}")
        return cls(list(raw["chars"]))

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Build a tokenizer from the sorted unique characters in ``text``."""
        chars = sorted(set(text))
        return cls(chars)


def load_tokenizer(path: Path | str) -> CharTokenizer | ByteBPETokenizer:
    """Load char or byte-BPE tokenizer by inspecting JSON keys."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid tokenizer file: {path}")
    if "merges" in raw:
        return ByteBPETokenizer.load(path)
    if "chars" in raw:
        return CharTokenizer.load(path)
    raise ValueError(
        f"Invalid tokenizer file (expected 'merges' or 'chars'): {path}"
    )


def _bin_path(txt_path: Path) -> Path:
    return txt_path.with_suffix(".bin")


def _cache_is_fresh(txt_path: Path, bin_path: Path, tokenizer_path: Path) -> bool:
    if not bin_path.is_file():
        return False
    bin_mtime = bin_path.stat().st_mtime
    return (
        bin_mtime >= txt_path.stat().st_mtime
        and bin_mtime >= tokenizer_path.stat().st_mtime
    )


def _encode_and_cache(
    txt_path: Path,
    bin_path: Path,
    tokenizer: TokenizerProtocol,
) -> np.ndarray:
    """Encode text to token ids and write a uint16 ``.bin`` cache."""
    text = txt_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    if not ids:
        raise ValueError(f"Encoded empty sequence from {txt_path}")
    if max(ids) > np.iinfo(np.uint16).max:
        raise ValueError(f"Token id {max(ids)} exceeds uint16; increase cache dtype")
    arr = np.array(ids, dtype=np.uint16)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(bin_path)
    print(f"Wrote {bin_path} ({len(arr):,} tokens)")
    return arr.astype(np.int32)


def _load_or_encode(
    txt_path: Path,
    tokenizer_path: Path,
    tokenizer: TokenizerProtocol,
) -> np.ndarray:
    if not txt_path.is_file():
        raise FileNotFoundError(f"Missing text file: {txt_path}")
    bin_path = _bin_path(txt_path)
    if _cache_is_fresh(txt_path, bin_path, tokenizer_path):
        arr = np.fromfile(bin_path, dtype=np.uint16)
        print(f"Loaded {bin_path} ({len(arr):,} tokens)")
        return arr.astype(np.int32)
    return _encode_and_cache(txt_path, bin_path, tokenizer)


class CharDataset:
    """Random-window next-token batches over encoded Shakespeare splits."""

    def __init__(
        self,
        data_dir: Path | str,
        tokenizer_path: Path | str,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        tokenizer_path = Path(tokenizer_path)
        self.tokenizer: CharTokenizer | ByteBPETokenizer = load_tokenizer(
            tokenizer_path
        )
        self.tokenizer_path = tokenizer_path
        self.train = _load_or_encode(
            data_dir / "train.txt", tokenizer_path, self.tokenizer
        )
        self.val = _load_or_encode(
            data_dir / "val.txt", tokenizer_path, self.tokenizer
        )
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_batch(
        self, split: str, batch_size: int, block_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a batch of ``(inputs, targets)`` with shape ``(B, T)``.

        Args:
            split: ``"train"`` or ``"val"``.
            batch_size: Number of sequences.
            block_size: Context length ``T``.

        Returns:
            Integer arrays ``x`` and ``y`` where ``y`` is ``x`` shifted by one.
        """
        data = self.train if split == "train" else self.val
        # * Random starts; need block_size+1 tokens so y can shift by one.
        high = len(data) - block_size
        if high <= 0:
            raise ValueError(
                f"Split '{split}' has {len(data)} tokens; need > {block_size}"
            )
        starts = self.rng.integers(0, high, size=batch_size)
        x = np.stack([data[s : s + block_size] for s in starts])
        y = np.stack([data[s + 1 : s + 1 + block_size] for s in starts])
        return x.astype(np.int32), y.astype(np.int32)
