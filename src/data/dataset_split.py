"""
Dataset Splitting and Cohort Pairing with strict Data Leakage Prevention.
Ensures deterministic stratified 80/20 train/test splits.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE, TEST_SPLIT_SIZE,
    ALTERED_LIMIT, GENDER_COHORT_LIMIT
)
from src.utils.io import get_logger

logger = get_logger(__name__)

def split_alteration_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SPLIT_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split alteration dataset into train and test with stratification.
    Validation data is strictly held out.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Alteration Split: Train={len(y_tr)}, Test={len(y_te)}")
    return X_tr, X_te, y_tr, y_te

def split_multimodal_dataset(
    X_iris: np.ndarray,
    X_fp: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SPLIT_SIZE,
    random_state: int = RANDOM_STATE
) -> Dict[str, np.ndarray]:
    """
    Split multimodal dataset using stratified indices so Iris and Fingerprint
    modalities maintain identical row alignment.
    """
    indices = np.arange(len(y))
    idx_tr, idx_te, y_train, y_test = train_test_split(
        indices, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Multimodal Split: Train={len(idx_tr)}, Test={len(idx_te)}")
    return {
        "idx_tr": idx_tr,
        "idx_te": idx_te,
        "X_iris_tr_raw": X_iris[idx_tr],
        "X_iris_te_raw": X_iris[idx_te],
        "X_fp_tr_raw": X_fp[idx_tr],
        "X_fp_te_raw": X_fp[idx_te],
        "y_train": y_train,
        "y_test": y_test
    }
