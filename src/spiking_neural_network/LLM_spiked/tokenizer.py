"""Byte-level BPE tokenizer (STLM-compatible) for SpikedLM."""

from __future__ import annotations

import json
from pathlib import Path


class ByteBPETokenizer:
    """Trainable byte-level BPE (ids 0..255 are raw UTF-8 bytes)."""

    def __init__(self) -> None:
        # * Ordered merges learned in train(): (id_a, id_b) -> next new id.
        self.merges: list[tuple[int, int]] = []

    @staticmethod
    def _count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        a, b = pair
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @property
    def vocab_size(self) -> int:
        # * Base 256 bytes + one id per merge.
        return 256 + len(self.merges)

    def train(self, text: str, vocab_size: int) -> None:
        """Learn merges from ``text`` until ``vocab_size`` ids exist."""
        if vocab_size < 256:
            raise ValueError("Vocab size must be >= 256")
        self.merges = []
        ids = list(text.encode("utf-8"))
        while len(self.merges) < vocab_size - 256:
            n_pairs = self._count_pairs(ids)
            if not n_pairs:
                break
            pair = max(n_pairs, key=n_pairs.get)  # type: ignore[arg-type]
            new_id = 256 + len(self.merges)
            self.merges.append(pair)
            ids = self._merge(ids, pair, new_id)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids via successive merge application."""
        ids = list(text.encode("utf-8"))
        for i, pair in enumerate(self.merges):
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
        return ids

    def _build_vocab(self) -> list[bytes]:
        # * vocab[i] = byte string for token id i
        vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            vocab.append(vocab[a] + vocab[b])
        return vocab

    def decode(self, ids: list[int]) -> str:
        """Decode token ids to a UTF-8 string (invalid bytes replaced)."""
        vocab = self._build_vocab()
        raw = b"".join(vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: Path | str) -> None:
        """Persist merges as JSON ``{"merges": [[a, b], ...]}``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"merges": [list(pair) for pair in self.merges]}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> ByteBPETokenizer:
        """Load a tokenizer from JSON."""
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "merges" not in raw:
            raise ValueError(f"Invalid BPE tokenizer file: {path}")
        tok = cls()
        tok.merges = [tuple(pair) for pair in raw["merges"]]
        return tok
