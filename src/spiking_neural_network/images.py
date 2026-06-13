import cv2
import numpy as np
from pathlib import Path

def show(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}


class ImageError(Exception):
    """Generic error for image-related issues."""


def load_grayscale(path: str | Path) -> np.ndarray:
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
    """Normalize the luminance of the image to be between 0 and 1."""
    if image.ndim != 2:
        raise ImageError(f"Image must be grayscale: {image.shape}")
    img_min = image.min()
    img_max = image.max()
    if img_min == img_max:
        return np.zeros_like(image, dtype=float)
    return (image - img_min) / (img_max - img_min)