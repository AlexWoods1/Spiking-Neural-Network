"""Small validation helpers for metrics, dataset sizing, and config fields."""

from __future__ import annotations

import math

from spiking_neural_network.exceptions import ParameterError


def require_positive(name: str, value: float | int) -> None:
    """Raise ``ParameterError`` when ``value`` is not strictly positive."""
    if value <= 0:
        raise ParameterError(f"{name} must be positive")


def require_at_least(name: str, value: int, minimum: int = 1) -> None:
    """Raise ``ParameterError`` when ``value`` is below ``minimum``."""
    if value < minimum:
        raise ParameterError(f"{name} must be at least {minimum}")


def require_non_negative(name: str, value: float | int) -> None:
    """Raise ``ParameterError`` when ``value`` is negative."""
    if value < 0:
        raise ParameterError(f"{name} must be non-negative")


def require_in_range(
    name: str,
    value: float,
    low: float,
    high: float,
    *,
    low_inclusive: bool = False,
    high_inclusive: bool = False,
    message_suffix: str = "",
) -> None:
    """Raise ``ParameterError`` when ``value`` falls outside ``(low, high)``."""
    below = value < low if low_inclusive else value <= low
    above = value > high if high_inclusive else value >= high
    if below or above:
        raise ParameterError(
            _range_message(
                name,
                low,
                high,
                low_inclusive=low_inclusive,
                high_inclusive=high_inclusive,
                message_suffix=message_suffix,
            )
        )


def _range_message(
    name: str,
    low: float,
    high: float,
    *,
    low_inclusive: bool,
    high_inclusive: bool,
    message_suffix: str,
) -> str:
    if not low_inclusive and high_inclusive:
        return f"{name} must be in ({low:g}, {high:g}]{message_suffix}"
    if low_inclusive and not high_inclusive:
        return f"{name} must be in [{low:g}, {high:g}){message_suffix}"
    if low_inclusive and high_inclusive:
        return f"{name} must be between {low:g} and {high:g} inclusive{message_suffix}"
    return f"{name} must be between {low:g} and {high:g}{message_suffix}"


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
