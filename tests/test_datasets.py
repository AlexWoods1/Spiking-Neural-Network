from pathlib import Path

import numpy as np
import pytest

from spiking_neural_network.datasets import (
    DatasetError,
    DataLoaderConfig,
    Split,
    _read_idx,
    _split_indices,
    encode_image_spikes,
    iter_mnist_batches,
    iter_mnist_samples,
    load_mnist,
    load_mnist_bundle,
    preencode_mnist_split,
    split_official_train_val,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"


def test_load_mnist_train(mnist_available: Path) -> None:
    images, labels = load_mnist(mnist_available, train=True)
    assert images.shape == (60000, 28, 28)
    assert labels.shape == (60000,)
    assert labels.dtype == np.uint8


def test_load_mnist_test(mnist_available: Path) -> None:
    images, labels = load_mnist(mnist_available, train=False)
    assert images.shape == (10000, 28, 28)
    assert labels.shape == (10000,)


def test_load_mnist_bundle(mnist_available: Path) -> None:
    (train_images, train_labels), (test_images, test_labels) = load_mnist_bundle(
        mnist_available
    )
    assert train_images.shape == (60000, 28, 28)
    assert test_images.shape == (10000, 28, 28)
    assert train_labels.shape == (60000,)
    assert test_labels.shape == (10000,)


def test_split_official_train_val_sizes() -> None:
    images = np.zeros((12, 28, 28), dtype=np.uint8)
    labels = np.arange(12, dtype=np.uint8)
    (train_images, train_labels), (val_images, val_labels) = split_official_train_val(
        images,
        labels,
        val_size=3,
    )
    assert train_images.shape == (9, 28, 28)
    assert val_images.shape == (3, 28, 28)
    assert train_labels.shape == (9,)
    assert val_labels.shape == (3,)
    np.testing.assert_array_equal(train_labels, np.arange(9))
    np.testing.assert_array_equal(val_labels, np.array([9, 10, 11]))


def test_encode_image_spikes_shape() -> None:
    rng = np.random.default_rng(0)
    image = np.full((28, 28), 128, dtype=np.uint8)
    spikes = encode_image_spikes(image, t_steps=5, rng=rng)
    assert spikes.shape == (5, 28 * 28)


def test_preencode_mnist_split_shape() -> None:
    images = np.full((4, 28, 28), 200, dtype=np.uint8)
    labels = np.arange(4, dtype=np.uint8)
    indices = np.array([0, 2, 3], dtype=int)
    rng = np.random.default_rng(1)

    spikes, split_labels = preencode_mnist_split(
        images,
        labels,
        indices,
        t_steps=5,
        rng=rng,
    )

    assert spikes.shape == (3, 5, 28 * 28)
    np.testing.assert_array_equal(split_labels, np.array([0, 2, 3]))


def test_preencode_mnist_split_rejects_empty_indices() -> None:
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    labels = np.array([0], dtype=np.uint8)
    with pytest.raises(DatasetError, match="at least one sample"):
        preencode_mnist_split(
            images,
            labels,
            np.array([], dtype=int),
            t_steps=4,
            rng=np.random.default_rng(0),
        )


def test_iter_mnist_samples(mnist_available: Path) -> None:
    config = DataLoaderConfig(
        data_dir=mnist_available,
        t_steps=3,
        seed=1,
        shuffle=False,
        batch_size=4,
    )
    samples = list(iter_mnist_samples(config, split=Split.TRAIN, limit=2))
    assert len(samples) == 2
    spikes, label = samples[0]
    assert spikes.shape == (3, 28 * 28)
    assert 0 <= label <= 9


def test_iter_mnist_batches(mnist_available: Path) -> None:
    config = DataLoaderConfig(
        data_dir=mnist_available,
        t_steps=2,
        seed=2,
        shuffle=False,
        batch_size=3,
    )
    batches = list(iter_mnist_batches(config, split=Split.TRAIN, limit=7))
    assert len(batches) == 3
    batch_spikes, batch_labels = batches[0]
    assert batch_spikes.shape == (3, 2, 28 * 28)
    assert batch_labels.shape == (3,)


def test_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_mnist(tmp_path, train=True)


def test_dataloader_config_rejects_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="t_steps must be at least 1"):
        DataLoaderConfig(data_dir=tmp_path, t_steps=0)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        DataLoaderConfig(data_dir=tmp_path, batch_size=0)
    with pytest.raises(ValueError, match="val_size must be at least 1"):
        DataLoaderConfig(data_dir=tmp_path, val_size=0)


def test_split_official_train_val_rejects_too_large_val_size() -> None:
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    labels = np.arange(4, dtype=np.uint8)

    with pytest.raises(DatasetError, match="val_size 4 must be smaller"):
        split_official_train_val(images, labels, val_size=4)


def test_split_indices_rejects_too_large_val_size() -> None:
    with pytest.raises(DatasetError, match="val_size 3 must be smaller"):
        _split_indices(3, split=Split.TRAIN, val_size=3)


def test_split_indices_rejects_unsupported_split() -> None:
    with pytest.raises(DatasetError, match="unsupported split for index partitioning"):
        _split_indices(10, split=Split.TEST, val_size=2)


def test_encode_image_spikes_rejects_non_2d_image() -> None:
    with pytest.raises(DatasetError, match="Expected 2D image"):
        encode_image_spikes(np.zeros((4, 28, 28)), t_steps=4, rng=np.random.default_rng(0))


def test_preencode_mnist_split_rejects_invalid_image_stack() -> None:
    with pytest.raises(DatasetError, match="Expected image stack"):
        preencode_mnist_split(
            np.zeros((4, 28), dtype=np.uint8),
            np.arange(4, dtype=np.uint8),
            np.array([0, 1], dtype=int),
            t_steps=4,
            rng=np.random.default_rng(0),
        )


def test_iter_mnist_samples_rejects_invalid_limit(mnist_available: Path) -> None:
    config = DataLoaderConfig(data_dir=mnist_available, shuffle=False)
    with pytest.raises(ValueError, match="limit must be at least 1"):
        list(iter_mnist_samples(config, split=Split.TRAIN, limit=0))


def test_iter_mnist_samples_supports_shuffle(mnist_available: Path) -> None:
    config = DataLoaderConfig(data_dir=mnist_available, shuffle=True, seed=11, batch_size=4)
    samples = list(iter_mnist_samples(config, split=Split.TRAIN, limit=8))
    assert len(samples) == 8


def test_read_idx_rejects_bad_magic(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad-images-idx3-ubyte"
    bad_file.write_bytes(b"\x00\x00\x00\x00")

    with pytest.raises(DatasetError, match="Unexpected magic number"):
        _read_idx(bad_file, expected_magic=2051)


def test_load_mnist_rejects_image_label_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "MNIST" / "raw"
    raw_dir.mkdir(parents=True)
    image_path = raw_dir / "train-images-idx3-ubyte"
    label_path = raw_dir / "train-labels-idx1-ubyte"
    image_path.write_bytes(
        b"\x00\x00\x08\x03"
        + (2).to_bytes(4, "big")
        + (28).to_bytes(4, "big")
        + (28).to_bytes(4, "big")
        + bytes(2 * 28 * 28)
    )
    label_path.write_bytes(b"\x00\x00\x08\x01" + (1).to_bytes(4, "big") + b"\x00")

    with pytest.raises(DatasetError, match="Image/label count mismatch"):
        load_mnist(tmp_path, train=True)


def test_split_sizes(mnist_available: Path) -> None:
    config = DataLoaderConfig(
        data_dir=mnist_available,
        t_steps=2,
        seed=3,
        shuffle=False,
        batch_size=4,
        val_size=10_000,
    )
    train_count = sum(1 for _ in iter_mnist_samples(config, split=Split.TRAIN))
    val_count = sum(1 for _ in iter_mnist_samples(config, split=Split.VAL))
    test_count = sum(1 for _ in iter_mnist_samples(config, split=Split.TEST, limit=100))
    assert train_count == 50_000
    assert val_count == 10_000
    assert test_count == 100
