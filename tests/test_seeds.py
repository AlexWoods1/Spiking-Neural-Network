"""Tests for deterministic experiment seed offsets."""

from spiking_neural_network.seeds import (
    ENCODING_SEED_TEST_OFFSET,
    ENCODING_SEED_TRAIN_OFFSET,
    ENCODING_SEED_VAL_OFFSET,
    MODEL_SEED_OFFSET,
    SHUFFLE_SEED_OFFSET,
    derived_seed,
)


def test_derived_seed_adds_offset() -> None:
    assert derived_seed(42, MODEL_SEED_OFFSET) == 42
    assert derived_seed(42, ENCODING_SEED_TRAIN_OFFSET) == 43
    assert derived_seed(42, ENCODING_SEED_VAL_OFFSET) == 44
    assert derived_seed(42, ENCODING_SEED_TEST_OFFSET) == 45
    assert derived_seed(42, SHUFFLE_SEED_OFFSET) == 46
