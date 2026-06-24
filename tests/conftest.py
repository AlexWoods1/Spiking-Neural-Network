import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"


@pytest.fixture
def mnist_available() -> Path:
    if not (DATA_DIR / "MNIST" / "raw" / "train-images-idx3-ubyte").is_file():
        pytest.skip("MNIST raw files not present")
    return DATA_DIR
