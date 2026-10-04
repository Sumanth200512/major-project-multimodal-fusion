"""
Integration and End-to-End Tests for Biometric Security Dashboard.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

from src.config import (
    ALTERATION_FEATURE_DIM, GENDER_FEATURE_DIM, FUSION_FEATURE_DIM,
    LABEL_ALTERED, LABEL_REAL, LABEL_FEMALE, LABEL_MALE,
    METRICS_DIR, MODELS_DIR
)
from src.preprocessing.fingerprint import preprocess_fingerprint, detect_minutiae
from src.preprocessing.features import (
    extract_alteration_features,
    extract_fingerprint_features,
    extract_gender_features_from_image
)
from src.models.late_fusion import LateFusionStackingClassifier
from src.models.classifiers import build_alteration_detector, build_benchmark_classifiers
from src.inference.pipeline import BiometricInferencePipeline
from src.utils.validation import convert_uploaded_file_to_cv2, validate_feature_dimension
from src.utils.io import load_artifact, load_json

def test_feature_extraction_synthetic():
    """Verify feature extractor output dimensionality and structure."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 180

    feat_alt = extract_alteration_features(img)
    assert feat_alt is not None
    assert len(feat_alt) == ALTERATION_FEATURE_DIM  # 295

    img_pre, binary, skeleton = preprocess_fingerprint(img)
    assert img_pre is not None
    assert binary is not None
    assert skeleton is not None

    feat_gender = extract_fingerprint_features(img_pre, binary, skeleton)
    assert feat_gender is not None
    assert len(feat_gender) == GENDER_FEATURE_DIM  # 297

def test_image_conversion_and_robustness():
    """Verify image conversion handles invalid, empty, and corrupted payloads."""
    # Empty
    res_empty, err_empty = convert_uploaded_file_to_cv2(b"")
    assert res_empty is None
    assert "empty" in err_empty

    # Corrupt
    res_corrupt, err_corrupt = convert_uploaded_file_to_cv2(b"corrupt_random_bytes_not_img")
    assert res_corrupt is None
    assert "Corrupted" in err_corrupt or "failed" in err_corrupt

    # Valid in-memory PNG
    dummy_img = np.zeros((50, 50), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", dummy_img)
    res_valid, err_valid = convert_uploaded_file_to_cv2(encoded.tobytes())
    assert err_valid is None
    assert res_valid is not None
    assert res_valid.shape == (50, 50)

def test_late_fusion_stacking_classifier():
    """Verify late fusion stacking classifier logic."""
    np.random.seed(42)
    n_samples = 40
    X_fusion = np.random.rand(n_samples, FUSION_FEATURE_DIM)
    y = np.random.randint(0, 2, size=n_samples)

    clf = LateFusionStackingClassifier(n_features_single=GENDER_FEATURE_DIM, random_state=42)
    clf.fit(X_fusion, y)

    preds = clf.predict(X_fusion)
    probs = clf.predict_proba(X_fusion)

    assert len(preds) == n_samples
    assert probs.shape == (n_samples, 2)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_feature_dimension_validation():
    """Verify feature dimension safety checks."""
    feat_valid = np.zeros(295)
    valid, _ = validate_feature_dimension(feat_valid, 295, "Test")
    assert valid is True

    feat_invalid = np.zeros(200)
    valid_inv, msg = validate_feature_dimension(feat_invalid, 295, "Test")
    assert valid_inv is False
    assert "dimension mismatch" in msg

def test_trained_pipeline_end_to_end():
    """
    Test end-to-end inference flow using saved production models:
    1. Altered fingerprint triggers ALTERED and pipeline stop.
    2. Real fingerprint triggers REAL and proceeds.
    3. Multimodal iris + fingerprint predicts gender with confidence.
    """
    pipeline = BiometricInferencePipeline()
    loaded, status = pipeline.load_models()
    assert loaded is True, f"Saved models failed to load: {status}"

    sample_dir = Path("assets/samples")
    assert (sample_dir / "sample_altered_fingerprint.bmp").exists()
    assert (sample_dir / "sample_real_fingerprint.bmp").exists()
    assert (sample_dir / "sample_iris.tiff").exists()

    # 1. Altered test
    alt_img = cv2.imread(str(sample_dir / "sample_altered_fingerprint.bmp"), cv2.IMREAD_GRAYSCALE)
    alt_res = pipeline.verify_fingerprint(alt_img)
    assert alt_res["success"] is True
    assert alt_res["is_real"] is False
    assert alt_res["status"] == "ALTERED"
    assert alt_res["confidence"] > 0.50

    # 2. Real test
    real_img = cv2.imread(str(sample_dir / "sample_real_fingerprint.bmp"), cv2.IMREAD_GRAYSCALE)
    real_res = pipeline.verify_fingerprint(real_img)
    assert real_res["success"] is True
    assert real_res["is_real"] is True
    assert real_res["status"] == "REAL"
    assert real_res["confidence"] > 0.50

    # 3. Multimodal gender prediction test
    iris_img = cv2.imread(str(sample_dir / "sample_iris.tiff"), cv2.IMREAD_GRAYSCALE)
    gender_res = pipeline.predict_multimodal_gender(real_img, iris_img)
    assert gender_res["success"] is True
    assert gender_res["gender"] in ["Male", "Female"]
    assert 0.0 <= gender_res["confidence"] <= 1.0
    assert gender_res["iris_feature_dim"] == 297
    assert gender_res["fp_feature_dim"] == 297
    assert gender_res["fusion_feature_dim"] == 594

def test_results_and_metadata_persistence():
    """Verify that experiment results and metadata files exist and have valid structure."""
    res_csv = METRICS_DIR / "experiment_results.csv"
    res_json = METRICS_DIR / "experiment_results.json"
    cms_json = METRICS_DIR / "confusion_matrices.json"
    rocs_json = METRICS_DIR / "roc_curves.json"
    metadata_json = MODELS_DIR / "metadata.json"

    assert res_csv.exists()
    assert res_json.exists()
    assert cms_json.exists()
    assert rocs_json.exists()
    assert metadata_json.exists()

    metadata = load_json(metadata_json)
    assert "feature_dimensions" in metadata
    assert metadata["feature_dimensions"]["alteration"] == 295
    assert metadata["feature_dimensions"]["gender_single"] == 297
    assert metadata["feature_dimensions"]["gender_fusion"] == 594
