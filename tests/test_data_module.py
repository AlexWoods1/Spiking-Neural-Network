"""Tests for dataset sample sources and batching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spiking_neural_network.config import DataModuleConfig
from spiking_neural_network.data_module import (
    ArraySampleSource,
    DataModule,
    MNISTDataConfig,
    MNISTDataProvider,
    MNISTSampleSource,
    estimate_preencode_bytes,
    preencode_mnist_source,
    should_preencode_split,
)
from spiking_neural_network.seeds import (
    ENCODING_SEED_TRAIN_OFFSET,
    ENCODING_SEED_VAL_OFFSET,
    derived_seed,
)
from tests.helpers import array_source


class TestMNISTSampleSource:
    def test_lazy_encode_shape(self) -> None:
        images = np.full((4, 28, 28), 200, dtype=np.uint8)
        labels = np.arange(4, dtype=np.uint8)
        source = MNISTSampleSource(
            images,
            labels,
            np.arange(3),
            t_steps=5,
            rng=np.random.default_rng(1),
        )

        spikes, label = source.sample(0)

        assert spikes.shape == (5, 784)
        assert label == 0
        assert len(source) == 3

    def test_split_encoding_rngs_are_independent(self) -> None:
        images = np.full((2, 28, 28), 200, dtype=np.uint8)
        labels = np.array([0, 1], dtype=np.uint8)
        val_source = MNISTSampleSource(
            images,
            labels,
            np.array([0]),
            t_steps=4,
            rng=np.random.default_rng(derived_seed(7, ENCODING_SEED_VAL_OFFSET)),
        )
        train_source = MNISTSampleSource(
            images,
            labels,
            np.array([1]),
            t_steps=4,
            rng=np.random.default_rng(derived_seed(7, ENCODING_SEED_TRAIN_OFFSET)),
        )
        expected_val_spikes, _ = val_source.sample(0)

        train_source.sample(0)
        fresh_val_source = MNISTSampleSource(
            images,
            labels,
            np.array([0]),
            t_steps=4,
            rng=np.random.default_rng(derived_seed(7, ENCODING_SEED_VAL_OFFSET)),
        )
        val_spikes, _ = fresh_val_source.sample(0)

        np.testing.assert_array_equal(val_spikes, expected_val_spikes)


class TestMNISTDataProvider:
    def test_limit_indices(self) -> None:
        limited = MNISTDataProvider._limit_indices(np.arange(10), 3)
        np.testing.assert_array_equal(limited, np.arange(3))

    def test_limit_indices_none_returns_all(self) -> None:
        indices = np.arange(10)
        np.testing.assert_array_equal(
            MNISTDataProvider._limit_indices(indices, None),
            indices,
        )

    def test_preencode_matches_lazy_source(self) -> None:
        images = np.full((4, 28, 28), 200, dtype=np.uint8)
        labels = np.arange(4, dtype=np.uint8)
        indices = np.array([0, 2, 1], dtype=int)
        lazy = MNISTSampleSource(
            images,
            labels,
            indices,
            t_steps=4,
            rng=np.random.default_rng(9),
        )
        preencoded = preencode_mnist_source(
            images,
            labels,
            indices,
            t_steps=4,
            rng=np.random.default_rng(9),
        )

        assert isinstance(preencoded, ArraySampleSource)
        for index in range(len(indices)):
            lazy_spikes, lazy_label = lazy.sample(index)
            pre_spikes, pre_label = preencoded.sample(index)
            np.testing.assert_array_equal(lazy_spikes, pre_spikes)
            assert lazy_label == pre_label

    def test_build_split_source_respects_preencode_flag(self) -> None:
        images = np.full((4, 28, 28), 100, dtype=np.uint8)
        labels = np.arange(4, dtype=np.uint8)
        indices = np.arange(3)
        rng = np.random.default_rng(3)
        preconfigured = MNISTDataProvider(
            MNISTDataConfig(data_dir=Path("data"), preencode=True),
        )
        lazy_configured = MNISTDataProvider(
            MNISTDataConfig(data_dir=Path("data"), preencode=False),
        )

        preencoded = preconfigured._build_split_source(
            images,
            labels,
            indices,
            rng=rng,
        )
        lazy = lazy_configured._build_split_source(
            images,
            labels,
            indices,
            rng=np.random.default_rng(3),
        )

        assert isinstance(preencoded, ArraySampleSource)
        assert isinstance(lazy, MNISTSampleSource)

    def test_resolve_preencode_auto_disables_large_splits(self) -> None:
        provider = MNISTDataProvider(
            MNISTDataConfig(data_dir=Path("data"), t_steps=50, preencode=None),
        )

        assert provider._resolve_preencode(50_000) is False
        assert provider._resolve_preencode(100) is True

    def test_resolve_preencode_explicit_true_warns_and_falls_back(self) -> None:
        provider = MNISTDataProvider(
            MNISTDataConfig(data_dir=Path("data"), t_steps=50, preencode=True),
        )

        with pytest.warns(UserWarning, match="lazy on-the-fly encoding"):
            assert provider._resolve_preencode(50_000) is False

    def test_should_preencode_split_uses_float32_budget(self) -> None:
        num_samples = 100
        t_steps = 4
        assert should_preencode_split(
            num_samples,
            t_steps,
            max_bytes=estimate_preencode_bytes(num_samples, t_steps),
        )


class TestDataModule:
    def test_iter_batches_uses_default_shuffle_from_config(self) -> None:
        module = DataModule(DataModuleConfig(batch_size=2, shuffle=True, seed=3))
        module.train = array_source(4)

        batches = list(module._iter_batches(module.train))

        assert len(batches) == 2
        assert batches[0][0].shape[0] == 2

    def test_test_dataloader_requires_setup(self) -> None:
        module = DataModule(DataModuleConfig())
        with pytest.raises(RuntimeError, match="setup\\(\\)"):
            list(module.test_dataloader())

    def test_iter_batches_respects_batch_size(self) -> None:
        module = DataModule(DataModuleConfig(batch_size=3, shuffle=False))
        module.train = array_source(7)

        batches = list(module._iter_batches(module.train, shuffle=False))

        assert len(batches) == 3
        assert batches[0][0].shape == (3, 4, 784)
        assert batches[-1][0].shape == (1, 4, 784)

    def test_train_dataloader_changes_order_by_epoch(self) -> None:
        module = DataModule(DataModuleConfig(batch_size=10, shuffle=True, seed=7))
        module.train = array_source(20)

        first_epoch = [
            labels.tolist() for _, labels in module.train_dataloader(epoch=1)
        ]
        second_epoch = [
            labels.tolist() for _, labels in module.train_dataloader(epoch=2)
        ]

        assert first_epoch != second_epoch

    def test_val_and_test_dataloaders_do_not_shuffle(self) -> None:
        module = DataModule(DataModuleConfig(batch_size=10))
        module.val = array_source(5)
        module.test = array_source(5)

        val_labels = [labels.tolist() for _, labels in module.val_dataloader()]
        test_labels = [labels.tolist() for _, labels in module.test_dataloader()]

        assert val_labels == [list(range(5))]
        assert test_labels == [list(range(5))]

    def test_dataloader_requires_setup(self) -> None:
        module = DataModule(DataModuleConfig())
        with pytest.raises(RuntimeError, match="setup\\(\\)"):
            list(module.train_dataloader())

    def test_num_batches_counts_partial_final_batch(self) -> None:
        module = DataModule(DataModuleConfig(batch_size=3))
        module.train = array_source(7)

        assert module.num_batches(module.train) == 3

    def test_num_batches_requires_setup(self) -> None:
        module = DataModule(DataModuleConfig())
        with pytest.raises(RuntimeError, match="setup\\(\\)"):
            module.num_batches(module.train)

    def test_from_provider_populates_splits(self) -> None:
        class _Provider:
            def build_splits(self):
                return array_source(3), array_source(2), array_source(1)

        module = DataModule.from_provider(DataModuleConfig(), _Provider())

        assert len(module.train) == 3
        assert len(module.val) == 2
        assert len(module.test) == 1

    def test_array_sample_source_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="feature and label counts must match"):
            ArraySampleSource(np.zeros((2, 4, 784)), np.array([0]))

    def test_from_mnist_builds_splits(self, mnist_available: Path) -> None:
        from spiking_neural_network.validation import data_partitions

        train_limit, val_limit, test_limit = data_partitions(8, 0.5, 0.25, 0.25)
        module = DataModule.from_mnist(
            MNISTDataConfig(
                data_dir=mnist_available,
                train_limit=train_limit,
                val_limit=val_limit,
                test_limit=test_limit,
            ),
        )

        assert len(module.train) == train_limit
        assert len(module.val) == val_limit
        assert len(module.test) == test_limit

    def test_val_dataloader_requires_setup(self) -> None:
        module = DataModule(DataModuleConfig())
        with pytest.raises(RuntimeError, match="setup\\(\\)"):
            list(module.val_dataloader())
