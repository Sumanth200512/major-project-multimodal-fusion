"""
Feature extraction modules for:
1. Alteration Detection (295D: 7 base features + 288 HOG)
2. Gender Classification (297D per modality: 9 base features + 288 HOG)
"""
from typing import Optional, Union, Dict, Any, Tuple
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.morphology import skeletonize

from src.preprocessing.fingerprint import preprocess_fingerprint, detect_minutiae
from src.utils.io import get_logger

logger = get_logger(__name__)

def extract_alteration_features(
    image_path_or_img: Union[str, np.ndarray]
) -> Optional[np.ndarray]:
    """
    Feature Extractor for Alteration Detection (295 dimensions):
    - Minutiae: endings, minutiae_density
    - GLCM Texture: contrast, dissimilarity, homogeneity
    - LBP Variance: uniform LBP variance
    - Sobel Edge Variance: gradient magnitude variance
    - HOG: 288-dimensional spatial gradient representation (96x96, orientations=8, (16,16) cells)

    Returns:
        np.ndarray of shape (295,) or None on failure.
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
        return None

    # Preprocess
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    filtered = cv2.GaussianBlur(img_norm, (5, 5), 0)
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = (skeletonize(binary > 0) * 255).astype(np.uint8)

    # 1. Minutiae Count
    h, w = img.shape
    total_area = float(h * w)
    skel = (skeleton > 0).astype(np.uint8)
    endings = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 1:
                nbrs = [
                    skel[y - 1, x], skel[y - 1, x + 1], skel[y, x + 1], skel[y + 1, x + 1],
                    skel[y + 1, x], skel[y + 1, x - 1], skel[y, x - 1], skel[y - 1, x - 1]
                ]
                if 0.5 * np.sum(np.abs(np.array(nbrs) - np.roll(nbrs, -1))) == 1:
                    endings += 1
    minutiae_density = endings / total_area

    # 2. GLCM Texture Features
    img_q = (img // 16).astype(np.uint8)
    glcm = graycomatrix(img_q, distances=[1, 3], angles=[0, np.pi / 4], levels=16, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    # 3. LBP Variance & Sobel Edge Variance
    lbp = local_binary_pattern(img, 8, 1, method='uniform')
    lbp_var = float(np.var(lbp))

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_var = float(np.var(np.sqrt(sobelx**2 + sobely**2)))

    # 4. HOG Features (Histogram of Oriented Gradients)
    img_resized = cv2.resize(img, (96, 96))
    hog_feats = hog(
        img_resized,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=False
    )

    base_feats = np.array([
        endings, minutiae_density, contrast, dissimilarity,
        homogeneity, lbp_var, grad_var
    ], dtype=np.float64)

    return np.hstack((base_feats, hog_feats))

def extract_fingerprint_features(
    img: np.ndarray,
    binary_img: np.ndarray,
    skeleton_img: np.ndarray
) -> np.ndarray:
    """
    Features for Gender Classification (297 dimensions):
    - Minutiae: endings, bifurcations, total_minutiae, minutiae_density
    - Ridge Density: skeleton pixels / total area
    - GLCM Texture: contrast, homogeneity, energy, correlation
    - HOG: 288-dimensional spatial gradient representation (96x96, orientations=8, (16,16) cells)

    Used for BOTH Iris and Fingerprint modalities in the multimodal classification pipeline.
    """
    if img is None or binary_img is None or skeleton_img is None:
        return np.zeros(297, dtype=np.float64)

    h, w = img.shape
    total_area = float(h * w)

    endings, bifurcations = detect_minutiae(skeleton_img)
    total_minutiae = endings + bifurcations
    minutiae_density = total_minutiae / total_area
    ridge_density = np.sum(skeleton_img > 0) / total_area

    # GLCM Features
    img_q = (img // 16).astype(np.uint8)
    glcm = graycomatrix(img_q, distances=[1, 3], angles=[0, np.pi / 4], levels=16, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    base_feats = np.array([
        endings, bifurcations, total_minutiae, minutiae_density,
        ridge_density, contrast, homogeneity, energy, correlation
    ], dtype=np.float64)

    # HOG Feature extraction
    img_resized = cv2.resize(img, (96, 96))
    hog_feats = hog(
        img_resized,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=False
    )

    return np.hstack((base_feats, hog_feats))

def extract_gender_features_from_image(
    image_path_or_img: Union[str, np.ndarray]
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Convenience wrapper that runs preprocessing + 297D feature extraction for gender classification.
    Also returns intermediate explanation metrics for UI transparency.
    """
    img, binary, skeleton = preprocess_fingerprint(image_path_or_img)
    if img is None:
        return None, {}

    feats = extract_fingerprint_features(img, binary, skeleton)
    endings, bifurcations = detect_minutiae(skeleton)
    h, w = img.shape
    total_area = float(h * w)

    explanation = {
        "dimensions": f"{w}x{h}",
        "endings": int(endings),
        "bifurcations": int(bifurcations),
        "total_minutiae": int(endings + bifurcations),
        "minutiae_density": float((endings + bifurcations) / total_area),
        "ridge_density": float(np.sum(skeleton > 0) / total_area),
        "glcm_contrast": float(feats[5]),
        "glcm_homogeneity": float(feats[6]),
        "glcm_energy": float(feats[7]),
        "glcm_correlation": float(feats[8]),
        "hog_dim": 288,
        "total_dim": int(len(feats))
    }

    return feats, explanation
