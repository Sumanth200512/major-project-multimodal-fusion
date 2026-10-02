"""
Model Evaluator for generating exact classification metrics:
Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC AUC.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

from src.utils.io import get_logger

logger = get_logger(__name__)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate full evaluation metrics matching the notebook's evaluation:
    Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC": roc_auc,
        "confusion_matrix": cm,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }

def compile_results_dataframe(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format results into a clean evaluation dataframe."""
    df = pd.DataFrame(results_list)
    return df
