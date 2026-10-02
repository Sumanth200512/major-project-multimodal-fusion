"""
Classifier definitions and model builders.
"""
from typing import Dict, Any
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from src.config import (
    SVM_PARAMS, RF_PARAMS, HGB_PARAMS, LATE_FUSION_STACKING_PARAMS,
    ALTERATION_HGB_PARAMS
)
from src.models.late_fusion import LateFusionStackingClassifier

def build_alteration_detector() -> HistGradientBoostingClassifier:
    """Build the alteration detection classifier."""
    return HistGradientBoostingClassifier(**ALTERATION_HGB_PARAMS)

def build_benchmark_classifiers() -> Dict[str, Any]:
    """
    Build the exact 4 benchmark classifiers used in the experiment:
    1. SVM (RBF Kernel)
    2. Random Forest
    3. Hist Gradient Boosting
    4. Late Fusion Stacking
    """
    return {
        "SVM (RBF Kernel)": SVC(**SVM_PARAMS),
        "Random Forest": RandomForestClassifier(**RF_PARAMS),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(**HGB_PARAMS),
        "Late Fusion Stacking": LateFusionStackingClassifier(**LATE_FUSION_STACKING_PARAMS)
    }
