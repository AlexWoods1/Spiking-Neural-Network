"""Train an AdaLi network on MNIST with train/val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from spiking_neural_network.builder import ENCODING_SEED_TEST_OFFSET, Trainer, TrainingConfig, derived_seed
from spiking_neural_network.datasets import load_mnist
from spiking_neural_network.evaluation import (
    classify_image,
    collect_predictions,
    print_prediction_summary,
)
from spiking_neural_network.pipeline import (
    build_adali_model,
    build_mnist_data_module,
    default_split_limits,
)
from spiking_neural_network.plotting import (
    plot_classified_image,
    plot_classified_sample_grid,
    plot_evaluation_figures,
    plot_training_history,
)
from spiking_neural_network.training_logs import export_training_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mnist"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def build_parser() -> argparse.ArgumentParser:
    train_size, val_size, test_size = default_split_limits()
    parser = argparse.ArgumentParser(description="Train AdaLi on MNIST.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--t-steps", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument(
        "--lr-final",
        type=float,
        default=0.01,
        help="Cosine-anneal learning rate to this value by the final epoch",
    )
    parser.add_argument("--weight-scale", type=float, default=0.2)
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal-loss focusing parameter (0 reduces to softmax CE)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Optional focal-loss class balance weight in (0, 1]",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=train_size,
        help="Cap training samples (default: all official MNIST train holdout)",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=val_size,
        help="Cap validation samples (default: all official MNIST val holdout)",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=test_size,
        help="Cap test samples (default: all official MNIST test split)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots")
    parser.add_argument(
        "--preview-count",
        type=int,
        default=0,
        help="Number of test images to show with classifications (0 to skip)",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "jax"],
        default="jax",
        help="Training backend: jax (batched, fast) or numpy (reference)",
    )
    parser.add_argument(
        "--eval-val-every",
        type=int,
        default=1,
        help="Run validation accuracy every N epochs (always runs on final epoch)",
    )
    parser.add_argument(
        "--eval-test-every",
        type=int,
        default=0,
        help="Run test accuracy every N epochs during training (0 disables in-loop test eval)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip plots and in-loop test eval; validate only on the final epoch",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for history.csv, run.json, and class_summary.csv",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip writing training metrics to JSON/CSV",
    )
    encoding_group = parser.add_mutually_exclusive_group()
    encoding_group.add_argument(
        "--preencode",
        action="store_true",
        help="Bulk-encode all spikes up front (may use a lot of RAM)",
    )
    encoding_group.add_argument(
        "--lazy-encode",
        action="store_true",
        help="Encode spikes on each batch (slower, low memory)",
    )
    return parser


def _limited_test_images(data_dir: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    images, labels = load_mnist(data_dir, train=False)
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    return images, labels


def _plot_classification_previews(args: argparse.Namespace, model, test_images, test_labels) -> None:
    if args.preview_count <= 0 or test_images.shape[0] == 0:
        return

    preview_rng = np.random.default_rng(derived_seed(args.seed, ENCODING_SEED_TEST_OFFSET))
    predicted, probabilities = classify_image(
        model,
        test_images[0],
        t_steps=args.t_steps,
        rng=preview_rng,
    )
    plot_classified_image(test_images[0], int(test_labels[0]), predicted, probabilities)
    plot_classified_sample_grid(
        model,
        test_images,
        test_labels,
        t_steps=args.t_steps,
        seed=args.seed,
        count=args.preview_count,
    )


def _resolve_eval_schedule(args: argparse.Namespace) -> tuple[int, int]:
    if args.fast:
        return args.epochs, 0
    return args.eval_val_every, args.eval_test_every


def _run_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "t_steps": args.t_steps,
        "hidden": args.hidden,
        "learning_rate": args.learning_rate,
        "lr_final": args.lr_final,
        "weight_scale": args.weight_scale,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "backend": args.backend,
        "eval_val_every": args.eval_val_every,
        "eval_test_every": args.eval_test_every,
        "fast": args.fast,
    }


def _export_training_metrics(
    args: argparse.Namespace,
    *,
    history: list[dict[str, float | int]],
    final_accuracies: dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Path]:
    export_paths = export_training_run(
        args.output_dir,
        history=history,
        final_accuracies=final_accuracies,
        run_config=_run_config(args),
        y_true=y_true,
        y_pred=y_pred,
        num_classes=y_score.shape[1],
    )
    print(f"Saved training metrics to {args.output_dir.resolve()}")
    for name, path in export_paths.items():
        print(f"  {name}: {path.name}")
    return export_paths


def main() -> None:
    args = build_parser().parse_args()
    if not args.data_dir.is_dir():
        print(f"Error: MNIST not found at {args.data_dir}", file=sys.stderr)
        raise SystemExit(1)

    eval_val_every, eval_test_every = _resolve_eval_schedule(args)
    skip_plots = args.no_plot or args.fast
    if args.preencode:
        preencode: bool | None = True
    elif args.lazy_encode:
        preencode = False
    else:
        preencode = None

    data_module = build_mnist_data_module(
        data_dir=args.data_dir,
        t_steps=args.t_steps,
        seed=args.seed,
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        preencode=preencode,
    )
    model = build_adali_model(
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        lr_final=args.lr_final,
        weight_scale=args.weight_scale,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        seed=args.seed,
        backend=args.backend,
    )
    trainer = Trainer(
        model,
        TrainingConfig(train_name="adali", total_epochs=args.epochs),
    )
    history, final_accuracies = trainer.fit(
        data_module,
        evaluate_val=True,
        evaluate_test=eval_test_every > 0,
        eval_val_every=eval_val_every,
        eval_test_every=eval_test_every,
        show_progress=not args.no_progress,
    )

    y_true, y_pred, y_score = collect_predictions(model, data_module.test_dataloader())

    if not args.no_export:
        _export_training_metrics(
            args,
            history=history,
            final_accuracies=final_accuracies,
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
        )

    print_prediction_summary(y_true, y_pred, y_score.shape[1])

    if skip_plots:
        return

    test_images, test_labels = _limited_test_images(args.data_dir, args.test_limit)
    _plot_classification_previews(args, model, test_images, test_labels)
    plot_training_history(history)
    plot_evaluation_figures(y_true, y_pred, y_score)


if __name__ == "__main__":
    main()
