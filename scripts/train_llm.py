"""CLI entrypoint for SpikedLM training."""

from __future__ import annotations

import argparse
from pathlib import Path

from spiking_neural_network.LLM_spiked.train import train


def main() -> None:
    p = argparse.ArgumentParser(description="Train SpikedLM on Shakespeare.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/llm_smoke.yaml"),
        help="Path to YAML config",
    )
    args = p.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
