"""Persist training metrics to JSON and CSV for offline plotting."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

HistoryRecord = dict[str, float | int]


def _history_fieldnames(history: list[HistoryRecord]) -> list[str]:
    fieldnames = ["epoch", "train_loss", "learning_rate"]
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(str(key))
    return fieldnames


def save_history_csv(history: list[HistoryRecord], path: Path) -> None:
    """Write per-epoch metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _history_fieldnames(history)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def build_class_summary_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> list[dict[str, float | int]]:
    """Return per-class counts and recall/precision for tabular export."""
    from spiking_neural_network.evaluation import build_confusion_matrix

    confusion = build_confusion_matrix(y_true, y_pred, num_classes)
    true_counts = confusion.sum(axis=1)
    pred_counts = confusion.sum(axis=0)
    rows: list[dict[str, float | int]] = []
    for digit in range(num_classes):
        true_count = int(true_counts[digit])
        pred_count = int(pred_counts[digit])
        correct = int(confusion[digit, digit])
        recall = float(correct / true_count) if true_count else 0.0
        precision = float(correct / pred_count) if pred_count else 0.0
        rows.append(
            {
                "digit": digit,
                "true_count": true_count,
                "pred_count": pred_count,
                "correct": correct,
                "recall": recall,
                "precision": precision,
            }
        )
    return rows


def save_class_summary_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    """Write per-class prediction summary to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["digit", "true_count", "pred_count", "correct", "recall", "precision"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_run_json(
    path: Path,
    *,
    history: list[HistoryRecord],
    final_accuracies: dict[str, float],
    run_config: dict[str, Any],
    class_summary: list[dict[str, float | int]] | None = None,
) -> None:
    """Write a full training run snapshot to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": run_config,
        "history": history,
        "final_accuracies": final_accuracies,
    }
    if class_summary is not None:
        payload["class_summary"] = class_summary
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_training_run(
    output_dir: Path,
    *,
    history: list[HistoryRecord],
    final_accuracies: dict[str, float],
    run_config: dict[str, Any],
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    num_classes: int | None = None,
) -> dict[str, Path]:
    """Export history CSV, run JSON, and optional class summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "history_csv": output_dir / "history.csv",
        "run_json": output_dir / "run.json",
    }
    save_history_csv(history, paths["history_csv"])
    class_summary = None
    if y_true is not None and y_pred is not None:
        if num_classes is None:
            num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        class_summary = build_class_summary_rows(y_true, y_pred, num_classes)
        paths["class_summary_csv"] = output_dir / "class_summary.csv"
        save_class_summary_csv(class_summary, paths["class_summary_csv"])
    save_run_json(
        paths["run_json"],
        history=history,
        final_accuracies=final_accuracies,
        run_config=run_config,
        class_summary=class_summary,
    )
    return paths


def load_training_run(path: Path) -> dict[str, Any]:
    """Load a training run snapshot from a directory or ``run.json`` file."""
    run_json = path if path.name == "run.json" else path / "run.json"
    if not run_json.is_file():
        raise FileNotFoundError(f"Training run JSON not found: {run_json}")
    return json.loads(run_json.read_text(encoding="utf-8"))
