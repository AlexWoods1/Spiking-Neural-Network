"""Small validation helpers for metrics and dataset sizing."""

from __future__ import annotations

import math


def relative_error(expected: float, actual: float) -> float:
    """Return the relative error between expected and actual values.

    Args:
        expected: Reference value.
        actual: Observed value.

    Returns:
        ``0.0`` when both values are zero, ``inf`` when only ``expected`` is
        zero, otherwise ``abs(expected - actual) / abs(expected)``.
    """
    if expected == 0:
        return 0.0 if actual == 0 else float("inf")
    return abs(expected - actual) / abs(expected)


def data_partitions(
    data_size: int,
    training_percentage: float,
    validation_percentage: float,
    test_percentage: float,
) -> tuple[int, int, int]:
    """Return sample counts for train, validation, and test partitions.

    Train and validation sizes use floored percentages. Test receives all
    remaining samples so the three counts always sum to ``data_size``.

    Args:
        data_size: Total number of samples to partition.
        training_percentage: Fraction allocated to training.
        validation_percentage: Fraction allocated to validation.
        test_percentage: Fraction allocated to test; must satisfy
            ``training + validation + test == 1``.

    Returns:
        Tuple ``(training_size, validation_size, test_size)``.

    Raises:
        ValueError: If ``data_size`` is negative, any percentage is outside
            ``[0, 1]``, or the percentages do not sum to ``1``.
    """
    if data_size < 0:
        raise ValueError(f"data_size must be non-negative: {data_size}")

    for name, percentage in (
        ("training_percentage", training_percentage),
        ("validation_percentage", validation_percentage),
        ("test_percentage", test_percentage),
    ):
        if percentage < 0.0 or percentage > 1.0:
            raise ValueError(f"{name} must be between 0 and 1: {percentage}")

    total_fraction = training_percentage + validation_percentage + test_percentage
    if not math.isclose(total_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "partition percentages must sum to 1: " f"got {total_fraction}"
        )

    training_size = int(data_size * training_percentage)
    validation_size = int(data_size * validation_percentage)
    test_size = data_size - training_size - validation_size
    return training_size, validation_size, test_size
