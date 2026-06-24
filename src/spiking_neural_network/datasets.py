"""MNIST loading and spike-train dataloaders."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from spiking_neural_network.encoding import SpikeEncoding
from spiking_neural_network.lif import flatten_spikes

_MNIST_IMAGE_MAGIC = 2051
_MNIST_LABEL_MAGIC = 2049


class DatasetError(Exception):
    """Raised when a dataset file is missing or invalid."""


class Split(str, Enum):
    """MNIST data partition."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for MNIST spike dataloaders."""

    data_dir: Path
    t_steps: int = 8
    seed: int | None = 42
    shuffle: bool = True
    batch_size: int = 32
    val_size: int = 10_000

    def __post_init__(self) -> None:
        if self.t_steps < 1:
            raise ValueError(f"t_steps must be at least 1: {self.t_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1: {self.batch_size}")
        if self.val_size < 1:
            raise ValueError(f"val_size must be at least 1: {self.val_size}")


def _read_idx(path: Path, expected_magic: int) -> np.ndarray:
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")

    with path.open("rb") as handle:
        magic = int.from_bytes(handle.read(4), byteorder="big")
        if magic != expected_magic:
            raise DatasetError(f"Unexpected magic number in {path}: {magic}")

        if expected_magic == _MNIST_IMAGE_MAGIC:
            _count = int.from_bytes(handle.read(4), byteorder="big")
            rows = int.from_bytes(handle.read(4), byteorder="big")
            cols = int.from_bytes(handle.read(4), byteorder="big")
            buffer = np.frombuffer(handle.read(), dtype=np.uint8)
            return buffer.reshape(-1, rows, cols)

        _count = int.from_bytes(handle.read(4), byteorder="big")
        return np.frombuffer(handle.read(), dtype=np.uint8)


def load_mnist(
    data_dir: Path,
    *,
    train: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST images and labels from raw IDX files.

    Args:
        data_dir: Root directory containing ``MNIST/raw/``.
        train: Load training split when ``True``, test split otherwise.

    Returns:
        Tuple of ``(images, labels)`` with shapes ``(N, 28, 28)`` and ``(N,)``.
    """
    split = "train" if train else "t10k"
    raw_dir = data_dir / "MNIST" / "raw"
    images = _read_idx(raw_dir / f"{split}-images-idx3-ubyte", _MNIST_IMAGE_MAGIC)
    labels = _read_idx(raw_dir / f"{split}-labels-idx1-ubyte", _MNIST_LABEL_MAGIC)
    if images.shape[0] != labels.shape[0]:
        raise DatasetError(
            f"Image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}"
        )
    return images, labels


def load_mnist_bundle(
    data_dir: Path,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load official MNIST train and test splits as ``((train_x, train_y), (test_x, test_y))``."""
    return load_mnist(data_dir, train=True), load_mnist(data_dir, train=False)


def split_official_train_val(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    val_size: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Split official MNIST train into train and validation holdout.

    Validation is the last ``val_size`` samples, matching ``iter_mnist_samples``.
    """
    count = images.shape[0]
    if val_size >= count:
        raise DatasetError(f"val_size {val_size} must be smaller than train count {count}")

    val_start = count - val_size
    return (images[:val_start], labels[:val_start]), (images[val_start:], labels[val_start:])


def _split_indices(
    count: int,
    *,
    split: Split,
    val_size: int,
) -> np.ndarray:
    if val_size >= count:
        raise DatasetError(f"val_size {val_size} must be smaller than train count {count}")

    val_start = count - val_size
    if split is Split.TRAIN:
        return np.arange(val_start)
    if split is Split.VAL:
        return np.arange(val_start, count)
    raise DatasetError(f"unsupported split for index partitioning: {split}")


def _mnist_arrays_for_split(
    data_dir: Path,
    split: Split,
    val_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split is Split.TEST:
        images, labels = load_mnist(data_dir, train=False)
        indices = np.arange(labels.shape[0])
    else:
        images, labels = load_mnist(data_dir, train=True)
        indices = _split_indices(labels.shape[0], split=split, val_size=val_size)
    return images, labels, indices


def _prepare_indices(
    indices: np.ndarray,
    *,
    shuffle: bool,
    seed: int | None,
    limit: int | None,
) -> np.ndarray:
    if shuffle:
        shuffled = indices.copy()
        np.random.default_rng(seed).shuffle(shuffled)
        indices = shuffled
    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1: {limit}")
        indices = indices[:limit]
    return indices


def encode_image_spikes(
    image: np.ndarray,
    t_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Encode one MNIST image as a flattened spike train ``(T, n_input)``."""
    if image.ndim != 2:
        raise DatasetError(f"Expected 2D image, got shape {image.shape}")

    rates = image.astype(np.float64) / 255.0
    encoding = SpikeEncoding.from_rates(rates, t_steps, rng)
    spikes = (encoding.samples > 0).astype(np.float32, copy=False)
    flat, _ = flatten_spikes(spikes)
    return flat


def preencode_mnist_split(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    t_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode ``indices`` into ``(N, T, F)`` spikes and ``(N,)`` labels."""
    if indices.size == 0:
        raise DatasetError("indices must contain at least one sample")
    if images.ndim != 3:
        raise DatasetError(f"Expected image stack (N, H, W), got shape {images.shape}")

    spikes = np.stack(
        [encode_image_spikes(images[int(index)], t_steps, rng) for index in indices],
        axis=0,
    ).astype(np.float32, copy=False)
    return spikes, labels[indices].astype(np.int64)


def iter_mnist_samples(
    config: DataLoaderConfig,
    *,
    split: Split = Split.TRAIN,
    limit: int | None = None,
) -> Iterator[tuple[np.ndarray, int]]:
    """Yield ``(spike_train, label)`` pairs from train, val, or test split."""
    images, labels, indices = _mnist_arrays_for_split(
        config.data_dir,
        split,
        config.val_size,
    )
    indices = _prepare_indices(
        indices,
        shuffle=config.shuffle,
        seed=config.seed,
        limit=limit,
    )
    rng = np.random.default_rng(config.seed)
    for index in indices:
        yield encode_image_spikes(images[int(index)], config.t_steps, rng), int(labels[int(index)])


def iter_mnist_batches(
    config: DataLoaderConfig,
    *,
    split: Split = Split.TRAIN,
    limit: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(batch_spikes, batch_labels)`` from train, val, or test split."""
    batch_spikes: list[np.ndarray] = []
    batch_labels: list[int] = []

    for spikes, label in iter_mnist_samples(config, split=split, limit=limit):
        batch_spikes.append(spikes)
        batch_labels.append(label)
        if len(batch_spikes) == config.batch_size:
            yield np.stack(batch_spikes), np.asarray(batch_labels, dtype=int)
            batch_spikes = []
            batch_labels = []

    if batch_spikes:
        yield np.stack(batch_spikes), np.asarray(batch_labels, dtype=int)
