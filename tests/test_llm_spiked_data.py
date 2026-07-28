"""Tests for char tokenizer and dataset batching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spiking_neural_network.LLM_spiked.data import CharDataset, CharTokenizer


def _write_tiny_corpus(root: Path) -> CharTokenizer:
    train = "abcdefghi\n" * 40
    val = "abcdefghi\n" * 10
    (root / "train.txt").write_text(train, encoding="utf-8")
    (root / "val.txt").write_text(val, encoding="utf-8")
    tok = CharTokenizer.from_text(train + val)
    tok.save(root / "tokenizer.json")
    return tok


def test_char_tokenizer_roundtrip(tmp_path: Path) -> None:
    tok = CharTokenizer.from_text("abca")
    assert tok.vocab_size == 3
    ids = tok.encode("caba")
    assert tok.decode(ids) == "caba"
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.chars == tok.chars


def test_char_tokenizer_too_small() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        CharTokenizer(["a"])


def test_char_tokenizer_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        CharTokenizer(["a", "a"])


def test_get_batch_too_short(tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_text("ab", encoding="utf-8")
    (tmp_path / "val.txt").write_text("ab", encoding="utf-8")
    tok = CharTokenizer.from_text("ab")
    tok.save(tmp_path / "tokenizer.json")
    data = CharDataset(tmp_path, tmp_path / "tokenizer.json")
    with pytest.raises(ValueError, match="need >"):
        data.get_batch("train", batch_size=1, block_size=8)


def test_invalid_tokenizer_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"nope": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid tokenizer"):
        CharTokenizer.load(path)


def test_missing_text_file(tmp_path: Path) -> None:
    tok = CharTokenizer(["a", "b"])
    tok.save(tmp_path / "tokenizer.json")
    with pytest.raises(FileNotFoundError):
        CharDataset(tmp_path, tmp_path / "tokenizer.json")


def test_get_batch_shapes(tmp_path: Path) -> None:
    tok = _write_tiny_corpus(tmp_path)
    data = CharDataset(tmp_path, tmp_path / "tokenizer.json", rng=np.random.default_rng(0))
    assert data.tokenizer.vocab_size == tok.vocab_size
    x, y = data.get_batch("train", batch_size=4, block_size=8)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # * Targets are inputs shifted by one within each window.
    assert np.all(y[:, :-1] == x[:, 1:])


def test_bin_cache_reuse(tmp_path: Path) -> None:
    _write_tiny_corpus(tmp_path)
    CharDataset(tmp_path, tmp_path / "tokenizer.json")
    assert (tmp_path / "train.bin").is_file()
    CharDataset(tmp_path, tmp_path / "tokenizer.json")
