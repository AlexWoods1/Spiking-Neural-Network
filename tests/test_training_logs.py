"""Tests for training metric export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from spiking_neural_network.training_logs import (
    build_class_summary_rows,
    export_training_run,
    save_history_csv,
)


def test_save_history_csv_writes_expected_columns(tmp_path: Path) -> None:
    history = [
        {"epoch": 1, "train_loss": 1.9, "learning_rate": 0.25, "val_acc": 0.647},
        {"epoch": 2, "train_loss": 1.7, "learning_rate": 0.21, "val_acc": 0.69, "test_acc": 0.68},
    ]
    path = tmp_path / "history.csv"
    save_history_csv(history, path)

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[0]["val_acc"] == "0.647"
    assert rows[1]["test_acc"] == "0.68"


def test_build_class_summary_rows_reports_dead_classes() -> None:
    y_true = np.array([1, 1, 8, 8, 0])
    y_pred = np.array([0, 0, 0, 2, 0])
    rows = build_class_summary_rows(y_true, y_pred, num_classes=10)

    by_digit = {int(row["digit"]): row for row in rows}
    assert by_digit[1]["pred_count"] == 0
    assert by_digit[1]["recall"] == 0.0
    assert by_digit[8]["pred_count"] == 0
    assert by_digit[0]["pred_count"] == 4


def test_export_training_run_writes_json_and_csv(tmp_path: Path) -> None:
    history = [{"epoch": 1, "train_loss": 1.5, "learning_rate": 0.1, "val_acc": 0.7}]
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 0, 2])

    paths = export_training_run(
        tmp_path,
        history=history,
        final_accuracies={"train_acc": 0.7, "val_acc": 0.7, "test_acc": 0.67},
        run_config={"epochs": 1, "t_steps": 4},
        y_true=y_true,
        y_pred=y_pred,
        num_classes=3,
    )

    assert paths["history_csv"].exists()
    assert paths["run_json"].exists()
    assert paths["class_summary_csv"].exists()

    payload = json.loads(paths["run_json"].read_text(encoding="utf-8"))
    assert payload["config"]["epochs"] == 1
    assert payload["final_accuracies"]["test_acc"] == pytest.approx(0.67)
    assert len(payload["class_summary"]) == 3
