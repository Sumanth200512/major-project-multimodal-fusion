"""
Inference Pipeline:
Fingerprint Verification (Alteration Detection)
-> HALT if Altered
-> Iris Processing
-> Multimodal Feature Fusion
-> Late Fusion Stacking Gender Prediction
"""
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from src.config import (
    ALTERATION_MODELS_DIR, CLASSIFICATION_MODELS_DIR, SCALERS_DIR,
    ALTERATION_FEATURE_DIM, GENDER_FEATURE_DIM, FUSION_FEATURE_DIM,
    LABEL_ALTERED, LABEL_REAL, LABEL_MALE, LABEL_FEMALE
)
from src.preprocessing.features import (
    extract_alteration_features,
    extract_fingerprint_features,
    preprocess_fingerprint
)
from src.utils.io import load_artifact, get_logger
from src.utils.validation import validate_feature_dimension

logger = get_logger(__name__)

class BiometricInferencePipeline:
    """
    Manages loading of trained models and executes two-stage inference:
    Stage 1: Fingerprint Alteration Detection
    Stage 2: Multimodal Gender Classification
    """
    def __init__(self):
        self.clf_alt = None
        self.scaler_alt = None
        self.clf_late_fusion = None
        self.scaler_iris = None
        self.scaler_fp = None
        self.is_loaded = False

    def load_models(self) -> Tuple[bool, Dict[str, str]]:
        """
        Load all persisted model and scaler artifacts.
        Returns (success_flag, status_dictionary)
        """
        status = {}
        try:
            alt_path = ALTERATION_MODELS_DIR / "alteration_detector.joblib"
            alt_sc_path = SCALERS_DIR / "alteration_scaler.joblib"
            fusion_path = CLASSIFICATION_MODELS_DIR / "late_fusion_stacking.joblib"
            iris_sc_path = SCALERS_DIR / "iris_scaler.joblib"
            fp_sc_path = SCALERS_DIR / "fingerprint_scaler.joblib"

            missing = []
            for p, name in [
                (alt_path, "Alteration Detector"),
                (alt_sc_path, "Alteration Scaler"),
                (fusion_path, "Late Fusion Stacking"),
                (iris_sc_path, "Iris Scaler"),
                (fp_sc_path, "Fingerprint Scaler")
            ]:
                if not p.exists():
                    missing.append(name)
                    status[name] = "Missing"
                else:
                    status[name] = "Available"

            if missing:
                self.is_loaded = False
                logger.warning(f"Missing models: {missing}")
                return False, status

            self.clf_alt = load_artifact(alt_path)
            self.scaler_alt = load_artifact(alt_sc_path)
            self.clf_late_fusion = load_artifact(fusion_path)
            self.scaler_iris = load_artifact(iris_sc_path)
            self.scaler_fp = load_artifact(fp_sc_path)
            self.is_loaded = True
            logger.info("All model artifacts loaded successfully for inference.")
            return True, status

        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            self.is_loaded = False
            status["error"] = str(e)
            return False, status

    def set_active_classifier(self, model_filename_or_key: str):
        """
        Dynamically select an active multimodal classifier model.
        Supports Late Fusion Stacking, SVM, Random Forest, Hist Gradient Boosting.
        """
        model_path = CLASSIFICATION_MODELS_DIR / model_filename_or_key
        if not model_path.exists():
            # Try appending .joblib
            model_path = CLASSIFICATION_MODELS_DIR / f"{model_filename_or_key}.joblib"
        if model_path.exists():
            self.clf_late_fusion = load_artifact(model_path)
            logger.info(f"Active classifier switched to: {model_path.name}")
            return True
        else:
            logger.warning(f"Requested classifier file not found: {model_filename_or_key}")
            return False

    def verify_fingerprint(
        self,
        fp_img_gray: np.ndarray
    ) -> Dict[str, Any]:
        """
        Stage 1: Fingerprint Alteration Detection.
        Extracts 295D features, scales using alteration scaler, and predicts.

        Returns:
            {
                "success": bool,
                "is_real": bool,
                "status": "REAL" | "ALTERED",
                "label": int (1=Real, 0=Altered),
                "confidence": float (percentage 0.0 - 1.0),
                "prob_altered": float,
                "prob_real": float,
                "feature_dim": int,
                "features_raw": np.ndarray,
                "error": str or None
            }
        """
        if not self.is_loaded:
            loaded, _ = self.load_models()
            if not loaded:
                return {
                    "success": False,
                    "error": "Required models are not loaded. Train or load existing models first."
                }

        feat_alt = extract_alteration_features(fp_img_gray)
        if feat_alt is None:
            return {"success": False, "error": "Failed to extract alteration features from fingerprint."}

        valid, err = validate_feature_dimension(feat_alt, ALTERATION_FEATURE_DIM, "Alteration Feature")
        if not valid:
            return {"success": False, "error": err}

        feat_alt_scaled = self.scaler_alt.transform([feat_alt])
        pred = int(self.clf_alt.predict(feat_alt_scaled)[0])
        prob = self.clf_alt.predict_proba(feat_alt_scaled)[0]

        is_real = (pred == LABEL_REAL)
        confidence = float(prob[pred])

        return {
            "success": True,
            "is_real": is_real,
            "status": "REAL" if is_real else "ALTERED",
            "label": pred,
            "confidence": confidence,
            "prob_altered": float(prob[LABEL_ALTERED]),
            "prob_real": float(prob[LABEL_REAL]),
            "feature_dim": len(feat_alt),
            "features_raw": feat_alt,
            "error": None
        }

    def predict_multimodal_gender(
        self,
        fp_img_gray: np.ndarray,
        iris_img_gray: np.ndarray
    ) -> Dict[str, Any]:
        """
        Stage 2: Multimodal Gender Classification.
        Preprocesses both modalities, extracts 297D features each, scales with
        respective scalers, concatenates into 594D vector, and runs Late Fusion Stacking.

        Returns:
            {
                "success": bool,
                "gender": "Male" | "Female",
                "gender_label": int (1=Male, 0=Female),
                "confidence": float,
                "prob_female": float,
                "prob_male": float,
                "iris_feature_dim": int,
                "fp_feature_dim": int,
                "fusion_feature_dim": int,
                "iris_features_raw": np.ndarray,
                "fp_features_raw": np.ndarray,
                "error": str or None
            }
        """
        if not self.is_loaded:
            loaded, _ = self.load_models()
            if not loaded:
                return {
                    "success": False,
                    "error": "Required models are not loaded. Train or load existing models first."
                }

        # Preprocess both images
        i_img, i_bin, i_skel = preprocess_fingerprint(iris_img_gray)
        f_img, f_bin, f_skel = preprocess_fingerprint(fp_img_gray)

        if i_img is None or f_img is None:
            return {"success": False, "error": "Unable to preprocess biometric images."}

        feat_iris = extract_fingerprint_features(i_img, i_bin, i_skel)
        feat_fp = extract_fingerprint_features(f_img, f_bin, f_skel)

        valid_i, err_i = validate_feature_dimension(feat_iris, GENDER_FEATURE_DIM, "Iris Feature")
        valid_f, err_f = validate_feature_dimension(feat_fp, GENDER_FEATURE_DIM, "Fingerprint Feature")

        if not valid_i:
            return {"success": False, "error": err_i}
        if not valid_f:
            return {"success": False, "error": err_f}

        # Scale features using fitted scalers
        feat_iris_s = self.scaler_iris.transform([feat_iris])
        feat_fp_s = self.scaler_fp.transform([feat_fp])

        # Multimodal feature fusion [F_iris | F_fp]
        feat_fusion = np.hstack((feat_iris_s, feat_fp_s))
        valid_fuse, err_fuse = validate_feature_dimension(feat_fusion[0], FUSION_FEATURE_DIM, "Fusion Feature")
        if not valid_fuse:
            return {"success": False, "error": err_fuse}

        # Predict using Late Fusion Stacking classifier
        gender_pred = int(self.clf_late_fusion.predict(feat_fusion)[0])
        gender_prob = self.clf_late_fusion.predict_proba(feat_fusion)[0]

        gender_str = "Male" if gender_pred == LABEL_MALE else "Female"
        confidence = float(gender_prob[gender_pred])

        return {
            "success": True,
            "gender": gender_str,
            "gender_label": gender_pred,
            "confidence": confidence,
            "prob_female": float(gender_prob[LABEL_FEMALE]),
            "prob_male": float(gender_prob[LABEL_MALE]),
            "iris_feature_dim": len(feat_iris),
            "fp_feature_dim": len(feat_fp),
            "fusion_feature_dim": feat_fusion.shape[1],
            "iris_features_raw": feat_iris,
            "fp_features_raw": feat_fp,
            "error": None
        }
