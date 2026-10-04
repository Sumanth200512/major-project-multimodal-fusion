"""
End-to-End Model Trainer and Artifact Persistence Pipeline.
Implements Workflow A:
1. Dataset preparation and validation.
2. Feature extraction (with caching to avoid recomputation).
3. Strict Train/Validation separation with zero data leakage.
4. Model training (Alteration Detector + 4 Benchmark Classifiers × 3 Modalities).
5. Evaluation, metric logging, and plot generation.
6. Complete artifact serialization (models, scalers, metadata, metrics).
"""
import datetime
import glob
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    RANDOM_STATE, TEST_SPLIT_SIZE,
    ALTERATION_FEATURE_DIM, GENDER_FEATURE_DIM, FUSION_FEATURE_DIM,
    MODELS_DIR, ALTERATION_MODELS_DIR, CLASSIFICATION_MODELS_DIR, SCALERS_DIR,
    PROCESSED_DATA_DIR, METRICS_DIR, FIGURES_DIR,
    ALTERED_LIMIT, GENDER_COHORT_LIMIT
)
from src.data.dataset_manager import DatasetManager
from src.data.dataset_split import split_alteration_dataset, split_multimodal_dataset
from src.preprocessing.features import (
    extract_alteration_features,
    extract_fingerprint_features,
    preprocess_fingerprint
)
from src.models.classifiers import build_alteration_detector, build_benchmark_classifiers
from src.evaluation.evaluator import evaluate_predictions
from src.evaluation.plots import (
    plot_accuracy_comparison,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metric_comparison,
    plot_metric_heatmap
)
from src.utils.io import (
    save_artifact, save_json, save_numpy, load_numpy, get_logger
)

logger = get_logger(__name__)

class ModelTrainer:
    """
    Orchestrates full ML pipeline training and evaluation.
    """
    def __init__(self, data_manager: Optional[DatasetManager] = None):
        self.dm = data_manager or DatasetManager()

    def prepare_alteration_features(
        self,
        force_recompute: bool = False,
        sample_limit: int = 3000,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract or load cached alteration features (295D).
        """
        cache_x = PROCESSED_DATA_DIR / f"alteration_X_{sample_limit}.npy"
        cache_y = PROCESSED_DATA_DIR / f"alteration_y_{sample_limit}.npy"

        if not force_recompute and cache_x.exists() and cache_y.exists():
            logger.info("Loading cached alteration features...")
            if progress_cb:
                progress_cb(0.2, "Loaded cached alteration features.")
            return load_numpy(cache_x), load_numpy(cache_y)

        soco_avail, soco_info = self.dm.check_socofing_available()
        if not soco_avail:
            raise FileNotFoundError("SOCOFing dataset not found. Download it first.")

        real_files = glob.glob(os.path.join(soco_info["real_dir"], "*.BMP"))
        altered_files = []
        for root, _, files in os.walk(soco_info["alt_dir"]):
            for f in files:
                if f.upper().endswith(".BMP"):
                    altered_files.append(os.path.join(root, f))

        rng = np.random.default_rng(RANDOM_STATE)
        if len(altered_files) > ALTERED_LIMIT:
            altered_files = rng.choice(altered_files, size=ALTERED_LIMIT, replace=False).tolist()

        n_samples = min(len(real_files), len(altered_files), sample_limit)
        logger.info(f"Extracting alteration features for {n_samples} Real and {n_samples} Altered...")

        X_list, y_list = [], []
        total = n_samples * 2
        count = 0

        for p in real_files[:n_samples]:
            feat = extract_alteration_features(p)
            if feat is not None:
                X_list.append(feat)
                y_list.append(1)  # Real = 1
            count += 1
            if progress_cb and count % 50 == 0:
                progress_cb(count / total * 0.4, f"Extracting alteration features ({count}/{total})...")

        for p in altered_files[:n_samples]:
            feat = extract_alteration_features(p)
            if feat is not None:
                X_list.append(feat)
                y_list.append(0)  # Altered = 0
            count += 1
            if progress_cb and count % 50 == 0:
                progress_cb(count / total * 0.4, f"Extracting alteration features ({count}/{total})...")

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int64)

        save_numpy(X, cache_x)
        save_numpy(y, cache_y)
        return X, y

    def prepare_multimodal_features(
        self,
        force_recompute: bool = False,
        cohort_limit: int = GENDER_COHORT_LIMIT,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract or load cached multimodal features (297D per modality).
        """
        cache_iris = PROCESSED_DATA_DIR / f"iris_X_{cohort_limit}.npy"
        cache_fp = PROCESSED_DATA_DIR / f"fingerprint_X_{cohort_limit}.npy"
        cache_y = PROCESSED_DATA_DIR / f"multimodal_y_{cohort_limit}.npy"

        if not force_recompute and cache_iris.exists() and cache_fp.exists() and cache_y.exists():
            logger.info("Loading cached multimodal features...")
            if progress_cb:
                progress_cb(0.4, "Loaded cached multimodal features.")
            return load_numpy(cache_iris), load_numpy(cache_fp), load_numpy(cache_y)

        iris_avail, iris_info = self.dm.check_iris_available()
        soco_avail, soco_info = self.dm.check_socofing_available()

        if not iris_avail or not soco_avail:
            raise FileNotFoundError("Iris or SOCOFing dataset missing. Download datasets first.")

        # Read iris labels
        iris_labels = {}
        with open(iris_info["label_file"], "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    iris_labels[parts[0]] = int(parts[1])

        iris_males = [f for f, l in iris_labels.items() if l == 1]
        iris_females = [f for f, l in iris_labels.items() if l == 0]

        socofing_files = glob.glob(os.path.join(soco_info["real_dir"], "*.BMP"))
        fp_males = [f for f in socofing_files if "__M_" in os.path.basename(f)]
        fp_females = [f for f in socofing_files if "__F_" in os.path.basename(f)]

        num_males = min(len(iris_males), len(fp_males), cohort_limit)
        num_females = min(len(iris_females), len(fp_females), cohort_limit)

        paired_males = list(zip(iris_males[:num_males], fp_males[:num_males]))
        paired_females = list(zip(iris_females[:num_females], fp_females[:num_females]))

        logger.info(f"Extracting multimodal features: {num_males} Males, {num_females} Females...")
        X_iris_list, X_fp_list, y_list = [], [], []

        total_pairs = len(paired_males) + len(paired_females)
        processed = 0

        for iris_f, fp_f in paired_males:
            i_path = os.path.join(iris_info["images_dir"], iris_f)
            i_img, i_bin, i_skel = preprocess_fingerprint(i_path)
            f_img, f_bin, f_skel = preprocess_fingerprint(fp_f)
            if i_img is not None and f_img is not None:
                X_iris_list.append(extract_fingerprint_features(i_img, i_bin, i_skel))
                X_fp_list.append(extract_fingerprint_features(f_img, f_bin, f_skel))
                y_list.append(1)  # Male = 1
            processed += 1
            if progress_cb and processed % 20 == 0:
                progress_cb(0.4 + (processed / total_pairs) * 0.3, f"Extracting multimodal features ({processed}/{total_pairs})...")

        for iris_f, fp_f in paired_females:
            i_path = os.path.join(iris_info["images_dir"], iris_f)
            i_img, i_bin, i_skel = preprocess_fingerprint(i_path)
            f_img, f_bin, f_skel = preprocess_fingerprint(fp_f)
            if i_img is not None and f_img is not None:
                X_iris_list.append(extract_fingerprint_features(i_img, i_bin, i_skel))
                X_fp_list.append(extract_fingerprint_features(f_img, f_bin, f_skel))
                y_list.append(0)  # Female = 0
            processed += 1
            if progress_cb and processed % 20 == 0:
                progress_cb(0.4 + (processed / total_pairs) * 0.3, f"Extracting multimodal features ({processed}/{total_pairs})...")

        X_iris = np.array(X_iris_list, dtype=np.float64)
        X_fp = np.array(X_fp_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int64)

        save_numpy(X_iris, cache_iris)
        save_numpy(X_fp, cache_fp)
        save_numpy(y, cache_y)

        return X_iris, X_fp, y

    def train_all_models(
        self,
        force_recompute_features: bool = False,
        alteration_samples: int = 3000,
        multimodal_cohort_samples: int = GENDER_COHORT_LIMIT,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Complete training workflow:
        1. Extract/Load alteration features
        2. Train & Evaluate Alteration Detector
        3. Extract/Load multimodal features
        4. Split data (80/20 train/test) with NO data leakage
        5. Fit scalers strictly on training data
        6. Train 4 benchmark classifiers across 3 modalities
        7. Evaluate, record metrics, generate plots, and persist all artifacts
        """
        logger.info("Starting complete training pipeline...")
        if progress_cb:
            progress_cb(0.05, "Initializing dataset check...")

        # 1. Alteration Pipeline
        if progress_cb:
            progress_cb(0.1, "Preparing alteration dataset...")
        X_alt, y_alt = self.prepare_alteration_features(
            force_recompute=force_recompute_features,
            sample_limit=alteration_samples,
            progress_cb=progress_cb
        )

        X_alt_tr, X_alt_te, y_alt_tr, y_alt_te = split_alteration_dataset(X_alt, y_alt)
        scaler_alt = MinMaxScaler()
        X_alt_tr_s = scaler_alt.fit_transform(X_alt_tr)
        X_alt_te_s = scaler_alt.transform(X_alt_te)

        if progress_cb:
            progress_cb(0.45, "Training HistGradientBoosting alteration detector...")
        clf_alt = build_alteration_detector()
        clf_alt.fit(X_alt_tr_s, y_alt_tr)

        y_alt_pred = clf_alt.predict(X_alt_te_s)
        y_alt_prob = clf_alt.predict_proba(X_alt_te_s)[:, 1]
        alt_metrics = evaluate_predictions(y_alt_te, y_alt_pred, y_alt_prob)
        logger.info(f"Alteration Detector Accuracy: {alt_metrics['Accuracy']:.2%}")

        # Save alteration artifacts
        save_artifact(clf_alt, ALTERATION_MODELS_DIR / "alteration_detector.joblib")
        save_artifact(scaler_alt, SCALERS_DIR / "alteration_scaler.joblib")
        save_json(alt_metrics, METRICS_DIR / "alteration_metrics.json")

        # 2. Multimodal Pipeline
        if progress_cb:
            progress_cb(0.5, "Preparing multimodal dataset...")
        X_iris, X_fp, y_multi = self.prepare_multimodal_features(
            force_recompute=force_recompute_features,
            cohort_limit=multimodal_cohort_samples,
            progress_cb=progress_cb
        )

        splits = split_multimodal_dataset(X_iris, X_fp, y_multi)
        idx_tr = splits["idx_tr"]
        idx_te = splits["idx_te"]
        y_train = splits["y_train"]
        y_test = splits["y_test"]

        # Strict Normalization: Fit on Train, Transform Test - Zero Data Leakage!
        scaler_iris = MinMaxScaler()
        X_iris_tr = scaler_iris.fit_transform(X_iris[idx_tr])
        X_iris_te = scaler_iris.transform(X_iris[idx_te])

        scaler_fp = MinMaxScaler()
        X_fp_tr = scaler_fp.fit_transform(X_fp[idx_tr])
        X_fp_te = scaler_fp.transform(X_fp[idx_te])

        # Concatenation Fusion: F_fusion = [F_iris | F_fp]
        X_fusion_tr = np.hstack((X_iris_tr, X_fp_tr))
        X_fusion_te = np.hstack((X_iris_te, X_fp_te))

        # Save Scalers
        save_artifact(scaler_iris, SCALERS_DIR / "iris_scaler.joblib")
        save_artifact(scaler_fp, SCALERS_DIR / "fingerprint_scaler.joblib")

        modalities = {
            "Model 1: Iris Only": (X_iris_tr, X_iris_te),
            "Model 2: Fingerprint Only": (X_fp_tr, X_fp_te),
            "Model 3: Iris + Fingerprint (Fusion)": (X_fusion_tr, X_fusion_te)
        }

        classifiers = build_benchmark_classifiers()

        results = []
        cms = {}
        rocs = {}
        trained_models = {}

        total_models = len(classifiers) * len(modalities)
        trained_count = 0

        for clf_name, clf in classifiers.items():
            for mod_name, (X_tr, X_te) in modalities.items():
                trained_count += 1
                if progress_cb:
                    pct = 0.55 + (trained_count / total_models) * 0.35
                    progress_cb(pct, f"Training {clf_name} on {mod_name}...")

                clf.fit(X_tr, y_train)
                y_pred = clf.predict(X_te)
                y_prob = clf.predict_proba(X_te)[:, 1]

                eval_metrics = evaluate_predictions(y_test, y_pred, y_prob)

                key = f"{clf_name} - {mod_name}"
                cms[key] = eval_metrics["confusion_matrix"]
                rocs[key] = {
                    "fpr": eval_metrics["fpr"],
                    "tpr": eval_metrics["tpr"],
                    "auc": eval_metrics["AUC"]
                }

                results.append({
                    "Classifier": clf_name,
                    "Modality": mod_name,
                    "Accuracy": eval_metrics["Accuracy"],
                    "Precision": eval_metrics["Precision"],
                    "Recall": eval_metrics["Recall"],
                    "F1 Score": eval_metrics["F1 Score"],
                    "AUC": eval_metrics["AUC"]
                })

                # Persist trained model artifact
                safe_name = clf_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                safe_mod = mod_name.split(":")[0].lower().replace(" ", "_")
                model_filename = f"{safe_name}_{safe_mod}.joblib"
                save_artifact(clf, CLASSIFICATION_MODELS_DIR / model_filename)

                # Track key production models
                if clf_name == "Late Fusion Stacking" and "Fusion" in mod_name:
                    save_artifact(clf, CLASSIFICATION_MODELS_DIR / "late_fusion_stacking.joblib")
                if clf_name == "SVM (RBF Kernel)":
                    if "Iris Only" in mod_name:
                        save_artifact(clf, CLASSIFICATION_MODELS_DIR / "iris_classifier.joblib")
                    elif "Fingerprint Only" in mod_name:
                        save_artifact(clf, CLASSIFICATION_MODELS_DIR / "fingerprint_classifier.joblib")
                    elif "Fusion" in mod_name:
                        save_artifact(clf, CLASSIFICATION_MODELS_DIR / "fusion_classifier.joblib")

        # 3. Compile and save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(METRICS_DIR / "experiment_results.csv", index=False)
        save_json(results, METRICS_DIR / "experiment_results.json")
        save_json(cms, METRICS_DIR / "confusion_matrices.json")
        save_json(rocs, METRICS_DIR / "roc_curves.json")

        # 4. Generate and save figure files (static + data)
        if progress_cb:
            progress_cb(0.92, "Generating and saving evaluation figures...")

        plot_accuracy_comparison(results_df, save_path=FIGURES_DIR / "accuracy_comparison.png", interactive=False)
        plot_confusion_matrices(cms, save_path=FIGURES_DIR / "confusion_matrices.png", interactive=False)
        plot_roc_curves(rocs, save_path=FIGURES_DIR / "roc_curves.png", interactive=False)
        plot_metric_comparison(results_df, "Precision", save_path=FIGURES_DIR / "precision_comparison.png", interactive=False)
        plot_metric_comparison(results_df, "Recall", save_path=FIGURES_DIR / "recall_comparison.png", interactive=False)
        plot_metric_comparison(results_df, "F1 Score", save_path=FIGURES_DIR / "f1_comparison.png", interactive=False)
        plot_metric_comparison(results_df, "AUC", save_path=FIGURES_DIR / "auc_comparison.png", interactive=False)
        plot_metric_heatmap(results_df, save_path=FIGURES_DIR / "metric_heatmap.png", interactive=False)

        # 5. Save metadata
        metadata = {
            "random_state": RANDOM_STATE,
            "feature_dimensions": {
                "alteration": ALTERATION_FEATURE_DIM,
                "gender_single": GENDER_FEATURE_DIM,
                "gender_fusion": FUSION_FEATURE_DIM
            },
            "training_dataset": {
                "alteration_train_samples": len(y_alt_tr),
                "alteration_test_samples": len(y_alt_te),
                "multimodal_train_samples": len(idx_tr),
                "multimodal_test_samples": len(idx_te),
                "test_split_ratio": TEST_SPLIT_SIZE
            },
            "model_versions": {
                "alteration_detector": "HistGradientBoostingClassifier",
                "classifiers": list(classifiers.keys())
            },
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pipeline_version": "1.0.0"
        }
        save_json(metadata, MODELS_DIR / "metadata.json")

        if progress_cb:
            progress_cb(1.0, "Training pipeline complete! All artifacts saved.")

        logger.info("Training and evaluation completed successfully.")
        return {
            "results_df": results_df,
            "alteration_metrics": alt_metrics,
            "metadata": metadata
        }
