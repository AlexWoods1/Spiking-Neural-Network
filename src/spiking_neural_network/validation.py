def relative_error(expected: float, actual: float) -> float:
    if expected == 0:
        return 0.0 if actual == 0 else float("inf")
    return abs(expected - actual) / expected
