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
