"""Download tinyshakespeare and write train/val splits plus a char tokenizer."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from spiking_neural_network.LLM_spiked.data import CharTokenizer

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download_shakespeare() -> str:
    """Fetch the Karpathy tinyshakespeare corpus as UTF-8 text."""
    with urllib.request.urlopen(SHAKESPEARE_URL) as resp:
        return resp.read().decode("utf-8")


def split_train_val(text: str, train_frac: float) -> tuple[str, str]:
    """Split text on a newline near ``train_frac`` of the corpus length."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    # * Split on a newline boundary near the cut so we don't bisect a line.
    cut = int(len(text) * train_frac)
    cut = text.rfind("\n", 0, cut)
    if cut <= 0:
        cut = int(len(text) * train_frac)
    return text[:cut], text[cut:]


def write_splits(out_dir: Path, train: str, val: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text(train, encoding="utf-8")
    (out_dir / "val.txt").write_text(val, encoding="utf-8")
    print(f"Wrote {out_dir / 'train.txt'} ({len(train):,} chars)")
    print(f"Wrote {out_dir / 'val.txt'} ({len(val):,} chars)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Shakespeare char LM data.")
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--train-frac", type=float, default=0.9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = download_shakespeare()
    train, val = split_train_val(text, args.train_frac)
    write_splits(args.out, train, val)
    # * Vocab from the full corpus so train/val share the same stoi map.
    tok = CharTokenizer.from_text(text)
    tok_path = args.out / "tokenizer.json"
    tok.save(tok_path)
    print(f"Wrote {tok_path} (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()
