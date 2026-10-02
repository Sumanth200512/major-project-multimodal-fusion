# Biometric Security Dashboard: Multimodal Fingerprint + Iris Gender Classification & Integrity Verification

A production-grade, research-faithful Streamlit biometric application demonstrating **pre-classification alteration/spoof detection** and **multimodal biometric fusion** for demographic gender classification.

---

## 📌 Project Overview

Traditional unimodal biometric systems (relying on a single trait like a fingerprint or iris scan) are vulnerable to spoofing, synthetic alterations, and noisy acquisition environments. This application models a two-tier biometric security architecture:

1. **Pre-Classification Integrity Verification (Alteration Detection)**:
   - Evaluates incoming fingerprints against synthetic mutilations, obliterative alterations, and central rotations from the SOCOFing benchmark.
   - Extracts structural ridge discontinuities via Crossing Number (CN) minutiae density, Gray-Level Co-occurrence Matrix (GLCM) second-order statistics, Local Binary Pattern (LBP) variance, Sobel edge gradient variance, and dense 288-dimensional Histogram of Oriented Gradients (HOG) — total **295 dimensions**.
   - Classifies sample integrity with a tuned `HistGradientBoostingClassifier`.
   - **Fail-Secure Gate**: If an altered fingerprint is detected, the execution **HALTS immediately** and prevents downstream biometric processing.

2. **Multimodal Biometric Classification (Gender Identification)**:
   - If genuine, the user uploads an Iris image.
   - Extracts a balanced **297-dimensional feature vector** for each modality (Minutiae metrics + Ridge Density + GLCM + 288D HOG).
   - Normalizes both feature representations using strictly fitted `MinMaxScaler` instances.
   - Performs multimodal feature fusion ($594$D concatenated representation) evaluated through a **Late Fusion Stacking Classifier** (RBF Support Vector Classifiers + 5-fold cross-validated probability meta-classifier using Logistic Regression).
   - Produces the final gender prediction (`Male` vs. `Female`) alongside calibrated probabilistic confidence scores.

3. **Empirical Model Comparison**:
   - Compares 4 benchmark classifiers (**SVM with RBF Kernel**, **Random Forest**, **Hist Gradient Boosting**, **Late Fusion Stacking**) across 3 biometric modalities (**Iris Only**, **Fingerprint Only**, **Iris + Fingerprint Fusion**).
   - Displays exact experimentally obtained metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC) without fabrication.
   - Reproduces the experimental visualizations: Accuracy comparison, Late Fusion Confusion Matrices, ROC curves, and additional metric heatmaps.

---

## 🏗️ Architecture

```text
                      [ Input Fingerprint Image ]
                                   │
                      ┌────────────┴────────────┐
                      ▼                         ▼
             Morphological Thinning       HOG & GLCM Analysis
             (CN Minutiae Analysis)      (Gradient & Texture)
                      │                         │
                      └────────────┬────────────┘
                                   ▼
               [ Alteration Classifier (HistGradientBoosting) ]
                                   │
                   ┌───────────────┴───────────────┐
                   │                               │
            [ Altered (0) ]                 [ Genuine (1) ]
                   │                               │
             🛑 HALT & LOG                         ▼
                                        [ Input Iris Image ]
                                                   │
                            ┌──────────────────────┴──────────────────────┐
                            ▼                                             ▼
                  Iris Feature Extraction                     Fingerprint Feature Extraction
                  (9 Base + 288 HOG = 297D)                     (9 Base + 288 HOG = 297D)
                            │                                             │
                            └──────────────────────┬──────────────────────┘
                                                   ▼
                                      [ Normalization (MinMax) ]
                                                   │
                            ┌──────────────────────┴──────────────────────┐
                            ▼                                             ▼
                    Early Fusion (594D)                           Late Fusion Stacking
               (Direct Vector Concatenation)                  (Out-of-fold Meta-Classifier)
                            │                                             │
                            └──────────────────────┬──────────────────────┘
                                                   ▼
                                  [ Final Gender Prediction & Metrics ]
```

---

## 📂 Project Structure

```text
biometric-security-dashboard/
│
├── app.py                         # Main Streamlit dashboard home & overview
│
├── pages/
│   ├── 1_Prediction_Pipeline.py   # Page 1: Interactive Fingerprint -> Alteration -> Iris -> Gender Flow
│   └── 2_Model_Comparison.py      # Page 2: Benchmark comparisons, tables, and ROC/CM charts
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Centralized configuration, seeds, label encodings, and paths
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_manager.py     # Local dataset discovery, validation & KaggleHub downloader
│   │   └── dataset_split.py       # Deterministic stratified 80/20 train/test splitting (zero leakage)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── fingerprint.py         # Grayscale normalization, Otsu binarization, Zhang-Suen skeletonization
│   │   └── features.py            # CN minutiae, GLCM textures, LBP, Sobel variance, and HOG features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifiers.py         # Model builders (SVM, Random Forest, HistGradientBoosting)
│   │   ├── late_fusion.py         # Out-of-fold LateFusionStackingClassifier
│   │   └── trainer.py             # End-to-end training pipeline and artifact serialization
│   ├── inference/
│   │   ├── __init__.py
│   │   └── pipeline.py            # Two-stage BiometricInferencePipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py           # Metric calculations (Acc, Prec, Rec, F1, ROC-AUC)
│   │   └── plots.py               # Plotly interactive & Matplotlib chart generation
│   └── utils/
│       ├── __init__.py
│       ├── io.py                  # Joblib and JSON artifact persistence & logging
│       ├── validation.py          # Image decoding, channel conversion, and feature dimension validation
│       └── ui_helpers.py          # Shared Streamlit sidebar and status widgets
│
├── models/                        # Persisted trained models and scalers
│   ├── alteration/                # alteration_detector.joblib
│   ├── classification/            # late_fusion_stacking.joblib, svm, rf, hgb
│   ├── scalers/                   # alteration_scaler.joblib, iris_scaler.joblib, fingerprint_scaler.joblib
│   └── metadata.json              # Model serialization metadata
│
├── data/
│   ├── raw/                       # Raw dataset root
│   ├── processed/                 # Cached feature arrays (.npy)
│   └── splits/                    # Deterministic split indices
│
├── results/
│   ├── metrics/                   # experiment_results.csv, confusion_matrices.json, roc_curves.json
│   ├── figures/                   # Accuracy, confusion matrix, and ROC curve figures
│   └── predictions/
│
├── assets/
│   └── samples/                   # Sample real/altered fingerprints and iris images for testing
│
├── tests/
│   └── test_pipeline.py           # Unit and integration test suite
│
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml                # Biometric security dark theme & server configuration
```

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11 - 3.13)
- `pip` or `uv` package manager

### 2. Environment Setup

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows (Command Prompt / PowerShell)
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launching the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the test suite to verify feature extraction, dimension guards, image conversion robustness, and end-to-end inference:

```bash
pytest tests/
```

All unit and integration tests validate that:
- Feature extraction produces exact dimensions ($295$D for alteration, $297$D for gender, $594$D for fusion).
- Corrupted, zero-byte, or invalid images fail safely with clear error messages.
- Altered fingerprints trigger the `ALTERED` alert and stop the pipeline.
- Genuine fingerprints pass to the iris stage and predict gender with probabilistic confidence.

---

## 📊 Dataset Setup & Management

The application interfaces with two standard research datasets from Kaggle:
1. **SOCOFing** (`ruizgara/socofing`): Real and synthetic altered fingerprints (Central rotation, Z-cut, Obliteration).
2. **GFI Iris Dataset** (`basnamhamedsalih/iris-datasetndgfi`): Left-eye iris scans with demographic gender labels.

### Automatic Discovery & KaggleHub Downloader
- The app checks for existing datasets in `data/raw/` or in the local KaggleHub cache (`~/.cache/kagglehub/datasets/`).
- If missing, a download trigger is directly accessible in the Streamlit sidebar under **Dataset Status**.

---

## ⚙️ Workflows: Using Existing Models vs. Retraining

### Workflow A: Using Existing Trained Models (Default)
The repository includes pre-trained production model weights in `models/` and experimental metrics in `results/metrics/`.
- Predictions and model comparisons are immediately available upon starting the application without retraining.
- Artifacts are loaded into memory and cached with `@st.cache_resource` for zero-latency inference.

### Workflow B: Train / Retrain Models
1. Expand the **Model Management** section in the sidebar.
2. Select **Train / Retrain Models**.
3. Choose the sample sizes (default: 1,000 alteration samples per class, 750 multimodal cohorts per gender) and click **Start Training Pipeline**.
4. The system executes:
   - Feature extraction with cached array reuse (`data/processed/`).
   - Stratified 80/20 train/test split.
   - Strictly fits `MinMaxScaler` on the training partition only (**zero data leakage**).
   - Trains the Alteration Detector (`HistGradientBoostingClassifier`).
   - Trains 4 benchmark classifiers across 3 modalities.
   - Fits the `LateFusionStackingClassifier` with 5-fold cross-validated out-of-fold probabilities.
   - Evaluates all models, exports metrics (`.csv`, `.json`), saves figures (`.png`), and serializes all models (`joblib`).

---

## 🔬 Experimental Methodology & Integrity Guarantees

- **No Data Leakage**: All scalers and transformations are fitted strictly on `X_train` and applied to `X_test` via `.transform()`.
- **Out-of-Fold Meta-Learning**: `LateFusionStackingClassifier` utilizes 5-fold cross-validation on `X_train` to generate out-of-fold probability distributions for training the meta-classifier, preventing meta-learner overfitting.
- **Dimensionality Safety**: Feature extraction outputs are strictly validated prior to model inference ($295$D for alteration, $297$D for unimodal, $594$D for early fusion). Mismatches fail immediately with descriptive errors.
- **Privacy by Design**: Biometric uploads are processed strictly in-memory (`io.BytesIO`) and are never written to permanent disk storage.

---

## 🌐 Streamlit Community Cloud Deployment

This repository is pre-configured for Streamlit Community Cloud:
1. Fork or push this repository to GitHub.
2. Connect your repository on [share.streamlit.io](https://share.streamlit.io).
3. Set the Main file path to `app.py`.
4. The deployment will automatically utilize `requirements.txt` (featuring `opencv-python-headless` for headless server compatibility).

---

## 📄 License & Attribution

This demonstration application is created for academic research, evaluation, and demonstration purposes. Datasets are sourced from the SOCOFing project (Sokoto Coventry Fingerprint Dataset) and the GFI Iris Dataset.
