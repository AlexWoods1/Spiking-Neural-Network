import numpy as np
import pytest

from spiking_neural_network.validation import relative_error


@pytest.mark.parametrize(
    ("expected", "actual", "result"),
    [
        (10.0, 10.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 5.0, float("inf")),
        (100.0, 90.0, 0.1),
    ],
)
def test_relative_error(expected: float, actual: float, result: float) -> None:
    assert relative_error(expected, actual) == result
