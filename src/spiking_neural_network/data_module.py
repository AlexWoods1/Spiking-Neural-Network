"""Dataset sample sources and batching for model training."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spiking_neural_network.config import DataModuleConfig
from spiking_neural_network.datasets import (
    encode_image_spikes,
    load_mnist_bundle,
    preencode_mnist_split,
    split_official_train_val,
)
from spiking_neural_network.exceptions import ParameterError
from spiking_neural_network.seeds import (
    ENCODING_SEED_TEST_OFFSET,
    ENCODING_SEED_TRAIN_OFFSET,
    ENCODING_SEED_VAL_OFFSET,
    SHUFFLE_SEED_OFFSET,
    derived_seed,
)

SampleBatch = tuple[np.ndarray, np.ndarray]

MNIST_FEATURE_DIM = 784
DEFAULT_PREENCODE_MAX_BYTES = 2 * 1024 * 1024 * 1024
PREENCODE_DTYPE = np.dtype(np.float32)


def estimate_preencode_bytes(
    num_samples: int,
    t_steps: int,
    *,
    feature_dim: int = MNIST_FEATURE_DIM,
    dtype: np.dtype = PREENCODE_DTYPE,
) -> int:
    """Return the RAM needed to store ``(N, T, F)`` spike tensors."""
    return num_samples * t_steps * feature_dim * dtype.itemsize


def should_preencode_split(
    num_samples: int,
    t_steps: int,
    *,
    max_bytes: int = DEFAULT_PREENCODE_MAX_BYTES,
) -> bool:
    """Return whether bulk pre-encoding stays within the memory budget."""
    return estimate_preencode_bytes(num_samples, t_steps) <= max_bytes


@dataclass(frozen=True)
class MNISTDataConfig:
    """MNIST-specific loading and spike-encoding configuration."""

    data_dir: Path
    t_steps: int = 8
    val_size: int = 10_000
    seed: int = 42
    preencode: bool | None = None
    preencode_max_bytes: int = DEFAULT_PREENCODE_MAX_BYTES
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None

    def __post_init__(self) -> None:
        if self.t_steps < 1:
            raise ParameterError("t_steps must be at least 1")
        if self.val_size < 1:
            raise ParameterError("val_size must be at least 1")
        if self.seed <= 0:
            raise ParameterError("seed must be positive")
        for name, limit in (
            ("train_limit", self.train_limit),
            ("val_limit", self.val_limit),
            ("test_limit", self.test_limit),
        ):
            if limit is not None and limit < 1:
                raise ParameterError(f"{name} must be at least 1 when set")
        if self.preencode_max_bytes < 1:
            raise ParameterError("preencode_max_bytes must be at least 1")


class SampleSource(ABC):
    """Lazily materialize one encoded sample and label by index."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in this split."""

    @abstractmethod
    def sample(self, index: int) -> tuple[np.ndarray, int]:
        """Return ``(features, label)`` for the sample at ``index``."""


class ArraySampleSource(SampleSource):
    """Pre-encoded samples stored as ``(N, ...)`` feature arrays."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError("feature and label counts must match")
        self._features = features
        self._labels = labels

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def sample(self, index: int) -> tuple[np.ndarray, int]:
        return self._features[index], int(self._labels[index])


class MNISTSampleSource(SampleSource):
    """Lazily encode MNIST images into spike trains on ``sample()``."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        t_steps: int,
        rng: np.random.Generator,
    ) -> None:
        self._images = images
        self._labels = labels
        self._indices = indices
        self._t_steps = t_steps
        self._rng = rng

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def sample(self, index: int) -> tuple[np.ndarray, int]:
        image_index = int(self._indices[index])
        spikes = encode_image_spikes(
            self._images[image_index],
            self._t_steps,
            self._rng,
        )
        return spikes, int(self._labels[image_index])


def preencode_mnist_source(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    t_steps: int,
    rng: np.random.Generator,
) -> ArraySampleSource:
    """Bulk-encode MNIST images into an ``ArraySampleSource``."""
    features, split_labels = preencode_mnist_split(
        images,
        labels,
        indices,
        t_steps=t_steps,
        rng=rng,
    )
    return ArraySampleSource(features, split_labels)


class DataProvider(ABC):
    """Build train, validation, and test sample sources for a dataset."""

    @abstractmethod
    def build_splits(self) -> tuple[SampleSource, SampleSource, SampleSource]:
        """Return lazy sample sources for train, validation, and test."""


class MNISTDataProvider(DataProvider):
    """MNIST provider with official train/val/test partitioning."""

    def __init__(self, config: MNISTDataConfig) -> None:
        self.config = config

    def build_splits(self) -> tuple[SampleSource, SampleSource, SampleSource]:
        (train_images, train_labels), (test_images, test_labels) = load_mnist_bundle(
            self.config.data_dir
        )
        (train_images, train_labels), (val_images, val_labels) = (
            split_official_train_val(
                train_images,
                train_labels,
                val_size=self.config.val_size,
            )
        )
        train_indices = self._limit_indices(
            np.arange(train_images.shape[0]), self.config.train_limit
        )
        val_indices = self._limit_indices(
            np.arange(val_images.shape[0]), self.config.val_limit
        )
        test_indices = self._limit_indices(
            np.arange(test_images.shape[0]), self.config.test_limit
        )
        train_rng = np.random.default_rng(
            derived_seed(self.config.seed, ENCODING_SEED_TRAIN_OFFSET)
        )
        val_rng = np.random.default_rng(
            derived_seed(self.config.seed, ENCODING_SEED_VAL_OFFSET)
        )
        test_rng = np.random.default_rng(
            derived_seed(self.config.seed, ENCODING_SEED_TEST_OFFSET)
        )
        return (
            self._build_split_source(
                train_images,
                train_labels,
                train_indices,
                rng=train_rng,
            ),
            self._build_split_source(
                val_images,
                val_labels,
                val_indices,
                rng=val_rng,
            ),
            self._build_split_source(
                test_images,
                test_labels,
                test_indices,
                rng=test_rng,
            ),
        )

    def _resolve_preencode(self, num_samples: int) -> bool:
        """Decide whether this split should be bulk pre-encoded."""
        within_budget = should_preencode_split(
            num_samples,
            self.config.t_steps,
            max_bytes=self.config.preencode_max_bytes,
        )
        if self.config.preencode is False:
            return False
        if self.config.preencode is True and not within_budget:
            needed_gib = estimate_preencode_bytes(num_samples, self.config.t_steps) / (
                1024**3
            )
            budget_gib = self.config.preencode_max_bytes / (1024**3)
            warnings.warn(
                "Bulk pre-encoding would need "
                f"{needed_gib:.1f} GiB for {num_samples} samples at "
                f"t_steps={self.config.t_steps}; budget is {budget_gib:.1f} GiB. "
                "Using lazy on-the-fly encoding instead.",
                stacklevel=3,
            )
            return False
        if self.config.preencode is None and not within_budget:
            return False
        return True

    def _build_split_source(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> SampleSource:
        if self._resolve_preencode(indices.size):
            return preencode_mnist_source(
                images,
                labels,
                indices,
                t_steps=self.config.t_steps,
                rng=rng,
            )
        return MNISTSampleSource(
            images,
            labels,
            indices,
            t_steps=self.config.t_steps,
            rng=rng,
        )

    @staticmethod
    def _limit_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
        if limit is None:
            return indices
        return indices[:limit]


class DataModule:
    """Batch and iterate over dataset-agnostic lazy sample sources."""

    def __init__(self, config: DataModuleConfig) -> None:
        self.config = config
        self.train: SampleSource | None = None
        self.val: SampleSource | None = None
        self.test: SampleSource | None = None

    @classmethod
    def from_provider(
        cls,
        module_config: DataModuleConfig,
        provider: DataProvider,
    ) -> DataModule:
        """Create a module and populate splits from a dataset provider."""
        module = cls(module_config)
        module.setup(provider)
        return module

    @classmethod
    def from_mnist(
        cls,
        mnist_config: MNISTDataConfig,
        module_config: DataModuleConfig | None = None,
    ) -> DataModule:
        """Convenience factory for MNIST-backed modules."""
        return cls.from_provider(
            module_config or DataModuleConfig(seed=mnist_config.seed),
            MNISTDataProvider(mnist_config),
        )

    def setup(self, provider: DataProvider) -> None:
        """Load dataset metadata and build lazy train/val/test sources."""
        self.train, self.val, self.test = provider.build_splits()

    def _iter_batches(
        self,
        source: SampleSource,
        *,
        shuffle: bool | None = None,
        epoch: int | None = None,
    ) -> Iterator[SampleBatch]:
        """Encode and batch samples from one lazy split."""
        if shuffle is None:
            shuffle = self.config.shuffle
        indices = np.arange(len(source))
        if shuffle:
            shuffle_seed = derived_seed(self.config.seed, SHUFFLE_SEED_OFFSET)
            seed = shuffle_seed if epoch is None else shuffle_seed + epoch
            indices = np.random.default_rng(seed).permutation(indices)

        for start in range(0, len(indices), self.config.batch_size):
            batch_indices = indices[start : start + self.config.batch_size]
            batch_features: list[np.ndarray] = []
            batch_labels: list[int] = []
            for index in batch_indices:
                features, label = source.sample(int(index))
                batch_features.append(features)
                batch_labels.append(label)
            yield np.stack(batch_features), np.asarray(batch_labels, dtype=int)

    def train_dataloader(self, epoch: int | None = None) -> Iterator[SampleBatch]:
        """Return a fresh training batch iterator for one epoch."""
        if self.train is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return self._iter_batches(self.train, shuffle=self.config.shuffle, epoch=epoch)

    def val_dataloader(self) -> Iterator[SampleBatch]:
        """Return a fresh validation batch iterator."""
        if self.val is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return self._iter_batches(self.val, shuffle=False)

    def test_dataloader(self) -> Iterator[SampleBatch]:
        """Return a fresh test batch iterator."""
        if self.test is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return self._iter_batches(self.test, shuffle=False)

    def num_batches(self, source: SampleSource | None) -> int:
        """Return the number of batches for a sample source."""
        if source is None:
            raise RuntimeError("Call setup() before requesting batch counts")
        sample_count = len(source)
        if sample_count == 0:
            return 0
        return (sample_count + self.config.batch_size - 1) // self.config.batch_size
