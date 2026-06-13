import cv2
import numpy as np
from pathlib import Path

SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}


class ImageError(Exception):
    """Raised when an image cannot be loaded or used in the pipeline."""


def show(img: np.ndarray) -> None:
    """Display an image in an OpenCV window until a key is pressed."""
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load a grayscale image from disk.

    Args:
        path: Path to a supported image file.

    Returns:
        A 2D ``uint8`` array with shape ``(height, width)``.

    Raises:
        ImageError: If the path is missing, unsupported, or cannot be decoded.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageError(f"Image file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise ImageError(f"Unsupported image format: {path.suffix}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ImageError(f"Failed to load image: {path}")
    return image


def intensity_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image luminance to the range [0, 1].

    Args:
        image: 2D grayscale array.

    Returns:
        A float array with the same shape as ``image``.

    Raises:
        ImageError: If ``image`` is not 2D or contains non-finite values.
    """
    if image.ndim != 2:
        raise ImageError(f"Image must be 2D grayscale: {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ImageError("Image contains non-finite values")

    img_min = image.min()
    img_max = image.max()
    if img_min == img_max:
        return np.zeros_like(image, dtype=float)
    return (image - img_min) / (img_max - img_min)
