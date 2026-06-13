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
    return abs(expected - actual) / expected
