import pytest

from spiking_neural_network.validation import data_partitions, relative_error


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
