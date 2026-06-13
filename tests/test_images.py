from pathlib import Path

import cv2
import numpy as np
import pytest

from spiking_neural_network.images import (
    ImageError,
    intensity_normalize,
    load_grayscale,
    resize_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = PROJECT_ROOT / "Images" / "Screenshot 2025-05-02 185435.png"


def test_intensity_normalize_scales_to_unit_range() -> None:
    image = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    normalized = intensity_normalize(image)

    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)


def test_intensity_normalize_flat_image_returns_zeros() -> None:
    image = np.full((4, 4), 7, dtype=np.uint8)
    normalized = intensity_normalize(image)

    assert normalized.shape == image.shape
    assert np.all(normalized == 0.0)


def test_intensity_normalize_rejects_non_2d_input() -> None:
    with pytest.raises(ImageError, match="Image must be 2D grayscale"):
        intensity_normalize(np.ones((2, 2, 3)))


def test_intensity_normalize_rejects_non_finite_values() -> None:
    image = np.array([[0.0, np.nan], [1.0, 2.0]])
    with pytest.raises(ImageError, match="non-finite"):
        intensity_normalize(image)


def test_resize_image_changes_shape() -> None:
    image = np.zeros((16, 24), dtype=np.uint8)
    resized = resize_image(image, (32, 32))

    assert resized.shape == (32, 32)


def test_load_grayscale_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(ImageError, match="Image file not found"):
        load_grayscale(missing)


def test_load_grayscale_rejects_unsupported_format(tmp_path: Path) -> None:
    unsupported = tmp_path / "image.txt"
    unsupported.write_text("not an image", encoding="utf-8")
    with pytest.raises(ImageError, match="Unsupported image format"):
        load_grayscale(unsupported)


def test_load_grayscale_reads_png_file(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), np.arange(16, dtype=np.uint8).reshape(4, 4))

    loaded = load_grayscale(image_path)

    assert loaded.ndim == 2
    assert loaded.dtype == np.uint8
    assert loaded.shape == (4, 4)


@pytest.mark.skipif(not SAMPLE_IMAGE.is_file(), reason="Sample project image missing")
def test_load_grayscale_reads_project_sample_image() -> None:
    image = load_grayscale(SAMPLE_IMAGE)

    assert image.ndim == 2
    assert image.dtype == np.uint8
    assert image.size > 0
