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
    rate_ax.set_title("Firing rates")
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
