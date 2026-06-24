import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from spiking_neural_network.encoding import SpikeEncoding
from spiking_neural_network.adali.model import AdaLi
from spiking_neural_network.config import AdaLiConfig
from spiking_neural_network.plotting import (
    plot_classified_image,
    plot_classified_sample_grid,
    plot_evaluation_figures,
    plot_membrane_potential,
    plot_run_report,
    plot_spike_encoding,
    plot_spikes,
    plot_training_history,
)


def test_plot_spike_encoding_saves_png(tmp_path) -> None:
    rates = np.full((4, 4), 0.25)
    encoding = SpikeEncoding.from_rates(rates=rates, t=5, rng=np.random.default_rng(0))
    image = np.full((4, 4), 128, dtype=np.uint8)
    output = tmp_path / "spike_plot.png"

    plot_spike_encoding(
        encoding,
        image=image,
        show=False,
        save_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_spike_encoding_without_image(tmp_path) -> None:
    rates = np.full((3, 3), 0.5)
    encoding = SpikeEncoding.from_rates(rates=rates, t=4, rng=np.random.default_rng(1))
    output = tmp_path / "no_image_plot.png"

    plot_spike_encoding(
        encoding,
        title="Test encoding",
        show=False,
        save_path=output,
    )

    assert output.is_file()


def test_plot_spike_encoding_show_calls_pyplot() -> None:
    rates = np.full((2, 2), 0.2)
    encoding = SpikeEncoding.from_rates(rates=rates, t=3, rng=np.random.default_rng(2))

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_spike_encoding(encoding, show=True)

    mock_show.assert_called_once()


def test_plot_spikes_saves_png(tmp_path) -> None:
    spikes = np.zeros((5, 4, 4))
    spikes[0, 0, 0] = 1
    spikes[2, 1, 2] = 1
    output = tmp_path / "spikes.png"

    plot_spikes(spikes, title="Test spikes", show=False, save_path=output)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_spikes_show_calls_pyplot() -> None:
    spikes = np.zeros((3, 2, 2))
    spikes[1, 0, 1] = 1

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_spikes(spikes, show=True)

    mock_show.assert_called_once()


def test_plot_spikes_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="spikes must have shape"):
        plot_spikes(np.ones((4, 4)))


def test_plot_membrane_potential_saves_png(tmp_path) -> None:
    membrane = np.linspace(0, 1.2, 10).reshape(10, 1, 1)
    input_spikes = np.zeros((10, 1, 1))
    input_spikes[2, 0, 0] = 1
    output = tmp_path / "membrane.png"

    plot_membrane_potential(
        membrane,
        threshold=1.0,
        input_spikes=input_spikes,
        trace_at=(0, 0),
        show=False,
        save_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_membrane_potential_show_calls_pyplot() -> None:
    membrane = np.zeros((5, 2, 2))

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_membrane_potential(membrane, threshold=1.0, show=True)

    mock_show.assert_called_once()


def test_plot_membrane_potential_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="membrane must have shape"):
        plot_membrane_potential(np.ones((4, 4)), threshold=1.0)


def test_plot_membrane_potential_accepts_title(tmp_path) -> None:
    membrane = np.zeros((5, 2, 2))
    output = tmp_path / "membrane_title.png"

    plot_membrane_potential(
        membrane,
        threshold=1.0,
        title="Membrane trace",
        show=False,
        save_path=output,
    )

    assert output.is_file()


def test_plot_classified_image_show_calls_pyplot() -> None:
    probabilities = np.array([0.1, 0.2, 0.7])

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_classified_image(
            np.full((28, 28), 128, dtype=np.uint8),
            true_label=2,
            predicted=2,
            probabilities=probabilities,
            show=True,
        )

    mock_show.assert_called_once()


def test_plot_classified_sample_grid_rejects_empty_images() -> None:
    model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3))

    with pytest.raises(ValueError, match="at least one sample"):
        plot_classified_sample_grid(
            model,
            np.zeros((0, 28, 28), dtype=np.uint8),
            np.array([], dtype=int),
            t_steps=4,
            seed=1,
        )


def test_plot_classified_sample_grid_show_calls_pyplot() -> None:
    model = AdaLi(AdaLiConfig(hidden_dims=(8,), output_dim=3))

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_classified_sample_grid(
            model,
            np.full((2, 28, 28), 200, dtype=np.uint8),
            np.array([0, 1], dtype=int),
            t_steps=4,
            seed=3,
            count=2,
            show=True,
        )

    mock_show.assert_called_once()


def test_plot_evaluation_figures_runs_without_display() -> None:
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1],
        ]
    )

    plot_evaluation_figures(y_true, y_pred, y_score, show=False)


def test_plot_evaluation_figures_rejects_invalid_y_score() -> None:
    with pytest.raises(ValueError, match="y_score must be a 2D array"):
        plot_evaluation_figures(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0.5, 0.5]),
        )


def test_plot_training_history_runs_without_display() -> None:
    history = [
        {"epoch": 1, "val_acc": 0.5, "train_loss": 0.4, "learning_rate": 0.1},
        {"epoch": 2, "val_acc": 0.7, "train_loss": 0.2, "learning_rate": 0.05},
    ]

    plot_training_history(history, show=False)


def test_plot_training_history_skips_missing_val_accuracy() -> None:
    history = [
        {"epoch": 1, "train_loss": 0.4, "learning_rate": 0.1},
        {"epoch": 2, "train_loss": 0.2, "learning_rate": 0.05, "val_acc": 0.7},
    ]

    plot_training_history(history, show=False)


def test_plot_training_history_writes_png(tmp_path: Path) -> None:
    history = [
        {"epoch": 1, "train_loss": 1.8, "learning_rate": 0.25},
        {"epoch": 2, "train_loss": 1.6, "learning_rate": 0.1, "val_acc": 0.89},
    ]
    save_path = tmp_path / "training_history.png"

    plot_training_history(history, show=False, save_path=save_path)

    assert save_path.is_file()
    assert save_path.stat().st_size > 0


def test_plot_run_report_writes_png(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_dir.joinpath("run.json").write_text(
        """{
  "config": {
    "epochs": 2,
    "t_steps": 8,
    "hidden": 128,
    "learning_rate": 0.25
  },
  "history": [
    {"epoch": 1, "train_loss": 1.8, "learning_rate": 0.25},
    {"epoch": 2, "train_loss": 1.6, "learning_rate": 0.01, "val_acc": 0.81}
  ],
  "final_accuracies": {
    "train_acc": 0.8,
    "val_acc": 0.81,
    "test_acc": 0.79
  },
  "class_summary": [
    {
      "digit": 0,
      "true_count": 10,
      "pred_count": 9,
      "correct": 8,
      "recall": 0.8,
      "precision": 0.89
    },
    {
      "digit": 1,
      "true_count": 12,
      "pred_count": 11,
      "correct": 10,
      "recall": 0.83,
      "precision": 0.91
    }
  ]
}""",
        encoding="utf-8",
    )
    save_path = run_dir / "run_report.png"

    plot_run_report(run_dir, show=False, save_path=save_path)

    assert save_path.is_file()


def test_plot_evaluation_figures_show_calls_pyplot() -> None:
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_score = np.array([[0.9, 0.1], [0.2, 0.8]])

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_evaluation_figures(y_true, y_pred, y_score, show=True)

    mock_show.assert_called_once()


def test_plot_training_history_show_calls_pyplot() -> None:
    history = [{"epoch": 1, "val_acc": 0.5, "train_loss": 0.4}]

    with (
        patch("spiking_neural_network.plotting.plt.show") as mock_show,
        patch("spiking_neural_network.plotting.plt.close"),
    ):
        plot_training_history(history, show=True)

    mock_show.assert_called_once()
