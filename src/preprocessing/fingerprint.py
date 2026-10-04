"""
Fingerprint and Iris Preprocessing and Crossing Number (CN) Minutiae Detection.
Directly reproduces the preprocessing functions from the source-of-truth notebook.
"""
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from skimage.morphology import skeletonize

from src.utils.io import get_logger

logger = get_logger(__name__)

def preprocess_fingerprint(
    image_path_or_img: Union[str, np.ndarray]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fingerprint / Iris Preprocessing from notebook:
    1. Grayscale Normalization & Gaussian Filter
    2. Binarization (Otsu adaptive threshold, THRESH_BINARY_INV + THRESH_OTSU)
    3. Thinning / Skeletonization (Zhang-Suen / Lee morphological skeletonize)

    Returns:
        img: original grayscale image (uint8)
        binary: thresholded binary image (uint8, 0 or 255)
        skeleton_uint8: skeletonized binary image (uint8, 0 or 255)
    """
    if isinstance(image_path_or_img, str):
        img = cv2.imread(image_path_or_img, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_path_or_img, np.ndarray):
        if len(image_path_or_img.shape) == 3:
            img = cv2.cvtColor(image_path_or_img, cv2.COLOR_BGR2GRAY)
        else:
            img = image_path_or_img.copy()
    else:
        img = None

    if img is None or img.size == 0:
        return None, None, None

    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    filtered = cv2.GaussianBlur(img_norm, (5, 5), 0)
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    skeleton = skeletonize(binary > 0)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    return img, binary, skeleton_uint8

# Alias matching preprocess_image in notebook
preprocess_image = preprocess_fingerprint

def detect_minutiae(skeleton_img: np.ndarray) -> Tuple[int, int]:
    """
    Crossing Number (CN) Minutiae Detection:
    CN = 1 -> Ridge Ending
    CN = 3 -> Ridge Bifurcation

    Returns:
        (endings, bifurcations)
    """
    if skeleton_img is None or skeleton_img.size == 0:
        return 0, 0

    skel = (skeleton_img > 0).astype(np.uint8)
    h, w = skel.shape
    endings = 0
    bifurcations = 0

    # Iterating over internal pixels (1 to h-2, 1 to w-2)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 1:
                neighbors = [
                    skel[y - 1, x],
                    skel[y - 1, x + 1],
                    skel[y, x + 1],
                    skel[y + 1, x + 1],
                    skel[y + 1, x],
                    skel[y + 1, x - 1],
                    skel[y, x - 1],
                    skel[y - 1, x - 1]
                ]
                cn = 0.5 * np.sum(np.abs(np.array(neighbors) - np.roll(neighbors, -1)))
                if cn == 1:
                    endings += 1
                elif cn == 3:
                    bifurcations += 1

    return endings, bifurcations
