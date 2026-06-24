"""Preview MNIST spike batches from the pipeline ``DataModule``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from spiking_neural_network.pipeline import (
    DataModule,
    DataModuleConfig,
    MNISTDataConfig,
    SampleBatch,
)
from spiking_neural_network.datasets import DatasetError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mnist"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview MNIST spike dataloaders.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--t-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.data_dir.is_dir():
        print(
            f"Error: MNIST not found at {args.data_dir}. "
            "Place raw IDX files under MNIST/raw/.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        module = DataModule.from_mnist(
            MNISTDataConfig(
                data_dir=args.data_dir,
                t_steps=args.t_steps,
                seed=args.seed,
                train_limit=args.limit if args.split == "train" else None,
                val_limit=args.limit if args.split == "val" else None,
                test_limit=args.limit if args.split == "test" else None,
            ),
            DataModuleConfig(
                batch_size=args.batch_size,
                shuffle=not args.no_shuffle,
                seed=args.seed,
            ),
        )
    except DatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    loaders: dict[str, Callable[[], Iterator[SampleBatch]]] = {
        "train": module.train_dataloader,
        "val": module.val_dataloader,
        "test": module.test_dataloader,
    }
    sources = {
        "train": module.train,
        "val": module.val,
        "test": module.test,
    }
    split = args.split
    source = sources[split]
    if source is None:
        raise RuntimeError(f"{split} split was not initialized")
    print(f"split={split} samples={len(source)}")

    for batch_index, (batch_spikes, batch_labels) in enumerate(loaders[split]()):
        print(
            f"batch {batch_index}: spikes={batch_spikes.shape} "
            f"labels={batch_labels.tolist()} mean_rate={batch_spikes.mean():.4f}"
        )


if __name__ == "__main__":
    main()
