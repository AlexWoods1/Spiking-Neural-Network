import cv2
import numpy as np


def show(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def intensity_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize the luminance of the image to be between 0 and 1."""
    img_min = image.min()
    img_max = image.max()
    if img_min == img_max:
        return np.zeros_like(image, dtype=float)
    return (image - img_min) / (img_max - img_min)
