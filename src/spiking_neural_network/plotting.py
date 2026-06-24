from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spiking_neural_network.encoding import SpikeEncoding


def plot_spike_encoding(
    encoding: SpikeEncoding,
    *,
    image: np.ndarray | None = None,
    title: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Plot input image, rates, spike raster, and population activity.

    Args:
        encoding: Encoded spike trains with shape ``(T, H, W)``.
        image: Optional grayscale image shown beside the rate map.
        title: Optional figure title.
        show: Whether to open an interactive window.
        save_path: Optional path to save the figure as PNG.
    """
    t_steps, height, width = encoding.samples.shape
    spike_events = encoding.samples > 0
    times, rows, cols = np.where(spike_events)
    neuron_ids = rows * width + cols
    population_activity = encoding.samples.sum(axis=(1, 2))

    if image is None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
        rate_ax = axes[0]
        raster_ax = axes[1]
        activity_ax = axes[2]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        image_ax = axes[0, 0]
        rate_ax = axes[0, 1]
        raster_ax = axes[1, 0]
        activity_ax = axes[1, 1]

        image_ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        image_ax.set_title("Input")
        image_ax.axis("off")

    rate_ax.imshow(encoding.rates, cmap="viridis", vmin=0, vmax=1)
    rate_ax.set_title("Rate map")
    rate_ax.axis("off")

    raster_ax.scatter(times, neuron_ids, s=1, c="black", marker="|", linewidths=0.5)
    raster_ax.set_xlim(-0.5, t_steps - 0.5)
    raster_ax.set_ylim(-0.5, height * width - 0.5)
    raster_ax.set_xlabel("Time step")
    raster_ax.set_ylabel("Neuron (flattened pixel)")
    raster_ax.set_title("Spike raster")

    activity_ax.plot(np.arange(t_steps), population_activity, color="black")
    activity_ax.set_xlim(0, t_steps - 1)
    activity_ax.set_xlabel("Time step")
    activity_ax.set_ylabel("Spike count")
    activity_ax.set_title("Population activity")

    if title is not None:
        fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_spikes(
    spikes: np.ndarray,
    *,
    title: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Plot a spike raster and population activity from ``(T, H, W)`` spikes.

    Args:
        spikes: Spike tensor with shape ``(T, H, W)``.
        title: Optional figure title.
        show: Whether to open an interactive window.
        save_path: Optional path to save the figure as PNG.
    """
    if spikes.ndim != 3:
        raise ValueError(f"spikes must have shape (T, H, W); got {spikes.shape}")

    t_steps, height, width = spikes.shape
    spike_events = spikes > 0
    times, rows, cols = np.where(spike_events)
    neuron_ids = rows * width + cols
    population_activity = spikes.sum(axis=(1, 2))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    raster_ax, activity_ax = axes

    raster_ax.scatter(times, neuron_ids, s=1, c="black", marker="|", linewidths=0.5)
    raster_ax.set_xlim(-0.5, t_steps - 0.5)
    raster_ax.set_ylim(-0.5, height * width - 0.5)
    raster_ax.set_xlabel("Time step")
    raster_ax.set_ylabel("Neuron (flattened pixel)")
    raster_ax.set_title("Spike raster")

    activity_ax.plot(np.arange(t_steps), population_activity, color="black")
    activity_ax.set_xlim(0, t_steps - 1)
    activity_ax.set_xlabel("Time step")
    activity_ax.set_ylabel("Spike count")
    activity_ax.set_title("Population activity")

    if title is not None:
        fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_membrane_potential(
    membrane: np.ndarray,
    *,
    threshold: float,
    input_spikes: np.ndarray | None = None,
    trace_at: tuple[int, int] | None = None,
    title: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Plot membrane potential over time for one pixel.

    Args:
        membrane: Membrane potentials with shape ``(T, H, W)``.
        threshold: Spike threshold shown as a horizontal reference line.
        input_spikes: Optional input spikes with the same shape for overlay markers.
        trace_at: ``(row, col)`` pixel to plot. Defaults to the image center.
        title: Optional figure title.
        show: Whether to open an interactive window.
        save_path: Optional path to save the figure as PNG.
    """
    if membrane.ndim != 3:
        raise ValueError(f"membrane must have shape (T, H, W); got {membrane.shape}")

    t_steps, height, width = membrane.shape
    if trace_at is None:
        row, col = height // 2, width // 2
    else:
        row, col = trace_at

    trace = membrane[:, row, col]
    time_axis = np.arange(t_steps)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time_axis, trace, color="tab:blue", label="Membrane potential")
    ax.axhline(threshold, color="tab:red", linestyle="--", label="Threshold")

    if input_spikes is not None:
        input_trace = input_spikes[:, row, col] > 0
        input_times = time_axis[input_trace]
        if input_times.size > 0:
            ax.scatter(
                input_times,
                np.full(input_times.shape, threshold * 0.1),
                s=12,
                c="black",
                marker="|",
                label="Input spikes",
            )

    ax.set_xlim(0, t_steps - 1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Membrane potential")
    ax.set_title(f"Membrane trace at pixel ({row}, {col})")
    ax.legend(loc="upper right")

    if title is not None:
        fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_classified_image(
    image: np.ndarray,
    true_label: int,
    predicted: int,
    probabilities: np.ndarray,
    *,
    show: bool = True,
) -> None:
    """Show one digit image beside its predicted class probabilities."""
    fig, (ax_image, ax_probs) = plt.subplots(1, 2, figsize=(8, 4))

    ax_image.imshow(image, cmap="gray", vmin=0, vmax=255)
    outcome = "correct" if true_label == predicted else "incorrect"
    confidence = float(probabilities[predicted])
    ax_image.set_title(
        f"true={true_label}  predicted={predicted}  ({outcome}, {confidence:.0%})"
    )
    ax_image.axis("off")

    digits = np.arange(probabilities.shape[0])
    colors = ["tab:orange" if digit == predicted else "tab:gray" for digit in digits]
    ax_probs.bar(digits, probabilities, color=colors)
    ax_probs.set_xticks(digits)
    ax_probs.set_ylim(0.0, 1.0)
    ax_probs.set_xlabel("digit")
    ax_probs.set_ylabel("probability")
    ax_probs.set_title("classification probabilities")

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_classified_sample_grid(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    t_steps: int,
    seed: int,
    count: int = 6,
    show: bool = True,
) -> None:
    """Show a grid of test digits with true and predicted labels."""
    from spiking_neural_network.evaluation import classify_image
    from spiking_neural_network.seeds import ENCODING_SEED_TEST_OFFSET, derived_seed

    sample_count = min(count, int(images.shape[0]))
    if sample_count == 0:
        raise ValueError("images must contain at least one sample")

    columns = min(3, sample_count)
    rows = (sample_count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(derived_seed(seed, ENCODING_SEED_TEST_OFFSET))

    for index in range(sample_count):
        predicted, probabilities = classify_image(
            model,
            images[index],
            t_steps=t_steps,
            rng=rng,
        )
        axis = axes_flat[index]
        axis.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        true_label = int(labels[index])
        outcome = "ok" if predicted == true_label else "miss"
        confidence = float(probabilities[predicted])
        axis.set_title(
            f"true {true_label} -> {predicted} ({confidence:.0%}, {outcome})"
        )
        axis.axis("off")

    for axis in axes_flat[sample_count:]:
        axis.axis("off")

    fig.suptitle("Test sample classifications", y=1.02)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_confusion_matrix_on_axis(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    display_labels: list[str],
    values_format: str,
) -> None:
    """Render a confusion matrix heatmap with per-cell annotations."""
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(display_labels)))
    axis.set_yticks(range(len(display_labels)))
    axis.set_xticklabels(display_labels)
    axis.set_yticklabels(display_labels)
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = (
                format(value, values_format)
                if values_format != "d"
                else str(int(value))
            )
            axis.text(
                col, row, text, ha="center", va="center", color="black", fontsize=9
            )


def plot_evaluation_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    show: bool = True,
) -> None:
    """Plot count and recall confusion matrices."""
    from spiking_neural_network.evaluation import build_confusion_matrix

    if y_score.ndim != 2:
        raise ValueError("y_score must be a 2D array of class probabilities")

    num_classes = y_score.shape[1]
    display_labels = [str(label) for label in range(num_classes)]
    counts = build_confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    recall = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    matrix_configs = (
        ("Confusion Matrix (counts)", counts, "d"),
        ("Confusion Matrix (recall by true label)", recall, ".2f"),
    )
    for axis, (title, matrix, values_format) in zip(axes, matrix_configs, strict=True):
        _plot_confusion_matrix_on_axis(
            axis,
            matrix,
            title=title,
            display_labels=display_labels,
            values_format=values_format,
        )

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_history(
    history: list[dict[str, float | int]],
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Plot validation accuracy and mean training loss by epoch."""
    epochs = [int(record["epoch"]) for record in history]
    train_loss = [float(record["train_loss"]) for record in history]
    val_epochs = [int(record["epoch"]) for record in history if "val_acc" in record]
    val_accs = [float(record["val_acc"]) for record in history if "val_acc" in record]

    fig, loss_axis = plt.subplots(figsize=(8, 5))
    loss_axis.plot(epochs, train_loss, marker="o", color="#4c78a8", label="Mean Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Train loss", color="#4c78a8")
    loss_axis.tick_params(axis="y", labelcolor="#4c78a8")
    loss_axis.grid(True, alpha=0.3)

    if val_epochs:
        accuracy_axis = loss_axis.twinx()
        accuracy_axis.plot(
            val_epochs,
            val_accs,
            marker="s",
            color="#f58518",
            linestyle="--",
            label="Validation Accuracy",
        )
        accuracy_axis.set_ylabel("Validation accuracy", color="#f58518")
        accuracy_axis.tick_params(axis="y", labelcolor="#f58518")
        accuracy_axis.set_ylim(0.0, 1.0)

    fig.suptitle("Training history")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _format_run_config_summary(config: dict[str, object]) -> str:
    keys = (
        "epochs",
        "t_steps",
        "hidden",
        "learning_rate",
        "lr_final",
        "weight_scale",
        "batch_size",
        "backend",
        "seed",
        "fast",
    )
    lines = [f"{key}: {config[key]}" for key in keys if key in config]
    return "\n".join(lines)


def plot_run_report(
    run_path: str | Path,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Plot training history, final metrics, and per-class summary from a run export."""
    from matplotlib.gridspec import GridSpec

    from spiking_neural_network.training_logs import load_training_run

    payload = load_training_run(Path(run_path))
    history = payload["history"]
    config = payload["config"]
    final_accuracies = payload["final_accuracies"]
    class_summary = payload.get("class_summary", [])

    epochs = [int(record["epoch"]) for record in history]
    train_loss = [float(record["train_loss"]) for record in history]
    learning_rates = [float(record["learning_rate"]) for record in history]
    val_epochs = [int(record["epoch"]) for record in history if "val_acc" in record]
    val_accs = [float(record["val_acc"]) for record in history if "val_acc" in record]

    digits = [int(row["digit"]) for row in class_summary]
    recalls = [float(row["recall"]) for row in class_summary]
    precisions = [float(row["precision"]) for row in class_summary]
    true_counts = [int(row["true_count"]) for row in class_summary]
    pred_counts = [int(row["pred_count"]) for row in class_summary]
    correct_counts = [int(row["correct"]) for row in class_summary]

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = GridSpec(3, 2, figure=fig)

    config_ax = fig.add_subplot(grid[0, 0])
    config_ax.axis("off")
    config_ax.set_title("Run configuration", loc="left", fontweight="bold")
    config_ax.text(
        0.0,
        1.0,
        _format_run_config_summary(config),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    accuracy_ax = fig.add_subplot(grid[0, 1])
    split_names = list(final_accuracies.keys())
    split_values = [float(final_accuracies[name]) for name in split_names]
    bars = accuracy_ax.bar(
        split_names, split_values, color=["#4c78a8", "#72b7b2", "#f58518"]
    )
    accuracy_ax.set_ylim(0.0, 1.0)
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.set_title("Final split accuracy", fontweight="bold")
    for bar, value in zip(bars, split_values, strict=True):
        accuracy_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    loss_ax = fig.add_subplot(grid[1, 0])
    loss_ax.plot(epochs, train_loss, marker="o", color="#4c78a8")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Train loss")
    loss_ax.set_title("Training loss", fontweight="bold")
    loss_ax.grid(True, alpha=0.3)

    lr_ax = fig.add_subplot(grid[1, 1])
    lr_ax.plot(epochs, learning_rates, marker="o", color="#e45756")
    lr_ax.set_xlabel("Epoch")
    lr_ax.set_ylabel("Learning rate")
    lr_ax.set_title("Learning-rate schedule", fontweight="bold")
    lr_ax.grid(True, alpha=0.3)

    metrics_ax = fig.add_subplot(grid[2, 0])
    width = 0.35
    x = np.arange(len(digits))
    metrics_ax.bar(x - width / 2, recalls, width, label="Recall", color="#54a24b")
    metrics_ax.bar(x + width / 2, precisions, width, label="Precision", color="#b279a2")
    metrics_ax.set_xticks(x)
    metrics_ax.set_xticklabels([str(digit) for digit in digits])
    metrics_ax.set_xlabel("Digit")
    metrics_ax.set_ylabel("Score")
    metrics_ax.set_ylim(0.0, 1.05)
    metrics_ax.set_title("Per-class recall and precision (test)", fontweight="bold")
    metrics_ax.legend()
    metrics_ax.grid(True, axis="y", alpha=0.3)

    counts_ax = fig.add_subplot(grid[2, 1])
    counts_ax.bar(
        x - width / 2, true_counts, width, label="True count", color="#9ecae9"
    )
    counts_ax.bar(
        x + width / 2, pred_counts, width, label="Pred count", color="#fdae6b"
    )
    counts_ax.plot(
        x,
        correct_counts,
        color="#222222",
        marker="o",
        linewidth=1.5,
        label="Correct",
    )
    counts_ax.set_xticks(x)
    counts_ax.set_xticklabels([str(digit) for digit in digits])
    counts_ax.set_xlabel("Digit")
    counts_ax.set_ylabel("Count")
    counts_ax.set_title("Per-class counts (test)", fontweight="bold")
    counts_ax.legend()
    counts_ax.grid(True, axis="y", alpha=0.3)

    title_bits = [
        f"hidden={config.get('hidden')}",
        f"t_steps={config.get('t_steps')}",
        f"backend={config.get('backend')}",
    ]
    if val_epochs:
        title_bits.append(f"val_acc={val_accs[-1]:.1%}")
    if "test_acc" in final_accuracies:
        title_bits.append(f"test_acc={float(final_accuracies['test_acc']):.1%}")
    fig.suptitle(
        "Training run report — " + ", ".join(str(bit) for bit in title_bits),
        fontsize=13,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
