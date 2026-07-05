import pytest

from spiking_neural_network.exceptions import ParameterError
from spiking_neural_network.validation import (
    data_partitions,
    relative_error,
    require_at_least,
    require_in_range,
    require_non_negative,
    require_positive,
)


@pytest.mark.parametrize(
    ("expected", "actual", "result"),
    [
        (10.0, 10.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 5.0, float("inf")),
        (100.0, 90.0, 0.1),
        (-100.0, -90.0, 0.1),
    ],
)
def test_relative_error(expected: float, actual: float, result: float) -> None:
    assert relative_error(expected, actual) == result


def test_data_partitions_splits_60_20_20() -> None:
    train_size, val_size, test_size = data_partitions(10_000, 0.6, 0.2, 0.2)

    assert (train_size, val_size, test_size) == (6_000, 2_000, 2_000)
    assert train_size + val_size + test_size == 10_000


def test_data_partitions_assigns_rounding_remainder_to_test() -> None:
    train_size, val_size, test_size = data_partitions(10, 0.33, 0.33, 0.34)

    assert train_size == 3
    assert val_size == 3
    assert test_size == 4
    assert train_size + val_size + test_size == 10


@pytest.mark.parametrize(
    ("data_size", "percentages", "match"),
    [
        (-1, (0.6, 0.2, 0.2), "data_size must be non-negative"),
        (100, (-0.1, 0.6, 0.5), "training_percentage must be between 0 and 1"),
        (100, (0.6, 1.1, 0.2), "validation_percentage must be between 0 and 1"),
        (100, (0.6, 0.2, 1.1), "test_percentage must be between 0 and 1"),
        (100, (0.5, 0.3, 0.3), "partition percentages must sum to 1"),
    ],
)
def test_data_partitions_rejects_invalid_input(
    data_size: int,
    percentages: tuple[float, float, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        data_partitions(data_size, *percentages)


def test_data_partitions_accepts_floating_point_sum_near_one() -> None:
    train_size, val_size, test_size = data_partitions(
        100,
        1 / 3,
        1 / 3,
        1 - (2 / 3),
    )

    assert train_size + val_size + test_size == 100
    assert test_size >= 0


def test_require_positive_rejects_non_positive() -> None:
    with pytest.raises(ParameterError, match="x must be positive"):
        require_positive("x", 0)


def test_require_at_least_rejects_below_minimum() -> None:
    with pytest.raises(ParameterError, match="batch_size must be at least 1"):
        require_at_least("batch_size", 0)


def test_require_non_negative_rejects_negative() -> None:
    with pytest.raises(ParameterError, match="gamma must be non-negative"):
        require_non_negative("gamma", -1.0)


def test_require_in_range_rejects_outside_open_interval() -> None:
    with pytest.raises(ParameterError, match="decay must be between 0 and 1"):
        require_in_range("decay", 1.0, 0.0, 1.0)


def test_require_in_range_supports_half_open_interval() -> None:
    with pytest.raises(
        ParameterError, match="focal_alpha must be in \\(0, 1\\] when set"
    ):
        require_in_range(
            "focal_alpha",
            0.0,
            0.0,
            1.0,
            high_inclusive=True,
            message_suffix=" when set",
        )
