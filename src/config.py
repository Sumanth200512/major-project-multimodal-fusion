"""
Centralized Configuration for Biometric Security Dashboard.
All paths, label encodings, random seeds, and model hyperparameters are defined here.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
ALTERATION_MODELS_DIR = MODELS_DIR / "alteration"
CLASSIFICATION_MODELS_DIR = MODELS_DIR / "classification"
SCALERS_DIR = MODELS_DIR / "scalers"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Ensure runtime directories exist
for p in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR,
    ALTERATION_MODELS_DIR, CLASSIFICATION_MODELS_DIR, SCALERS_DIR,
    METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, ASSETS_DIR
]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for Exact Reproducibility
RANDOM_STATE = 42

# Label Mappings (Source of Truth from Notebook)
# Alteration Detection: Label 0 = Altered / Spoofed, Label 1 = Real / Genuine
LABEL_ALTERED = 0
LABEL_REAL = 1
ALTERATION_LABELS = {LABEL_ALTERED: "Altered", LABEL_REAL: "Real"}

# Multimodal Gender Classification: Label 0 = Female, Label 1 = Male
LABEL_FEMALE = 0
LABEL_MALE = 1
GENDER_LABELS = {LABEL_FEMALE: "Female", LABEL_MALE: "Male"}

# Feature Dimensions
ALTERATION_BASE_FEAT_DIM = 7
HOG_FEAT_DIM = 288
ALTERATION_FEATURE_DIM = ALTERATION_BASE_FEAT_DIM + HOG_FEAT_DIM  # 295

GENDER_BASE_FEAT_DIM = 9
GENDER_FEATURE_DIM = GENDER_BASE_FEAT_DIM + HOG_FEAT_DIM  # 297
FUSION_FEATURE_DIM = GENDER_FEATURE_DIM * 2  # 594

# Dataset Settings
TEST_SPLIT_SIZE = 0.20
ALTERED_LIMIT = 6000
GENDER_COHORT_LIMIT = 1000

# Classifier Configurations (from notebook)
ALTERATION_HGB_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": 8,
    "l2_regularization": 0.1,
    "random_state": RANDOM_STATE
}

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 10.0,
    "gamma": "scale",
    "probability": True,
    "random_state": RANDOM_STATE
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "random_state": RANDOM_STATE
}

HGB_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": 8,
    "random_state": RANDOM_STATE
}

LATE_FUSION_STACKING_PARAMS = {
    "n_features_single": GENDER_FEATURE_DIM,
    "random_state": RANDOM_STATE
}
