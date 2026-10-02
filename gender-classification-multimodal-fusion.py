# -*- coding: utf-8 -*-
# %%
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
# %pip install -q kagglehub
import kagglehub
basnamhamedsalih_iris_datasetndgfi_path = kagglehub.dataset_download('basnamhamedsalih/iris-datasetndgfi')
ruizgara_socofing_path = kagglehub.dataset_download('ruizgara/socofing')

# %%
print('Data source import complete.')

# %%
import os
import glob
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.morphology import skeletonize

# %%
# 1. Feature Extractor for Alteration Detection (Updated with HOG)
from skimage.feature import hog

def extract_alteration_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

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
                nbrs = [skel[y-1, x], skel[y-1, x+1], skel[y, x+1], skel[y+1, x+1],
                        skel[y+1, x], skel[y+1, x-1], skel[y, x-1], skel[y-1, x-1]]
                if 0.5 * np.sum(np.abs(np.array(nbrs) - np.roll(nbrs, -1))) == 1:
                    endings += 1
    minutiae_density = endings / total_area

    # 2. GLCM Texture Features
    img_q = (img // 16).astype(np.uint8)
    glcm = graycomatrix(img_q, distances=[1, 3], angles=[0, np.pi/4], levels=16, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    # 3. LBP Variance & Sobel Edge Variance
    lbp = local_binary_pattern(img, 8, 1, method='uniform')
    lbp_var = np.var(lbp)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_var = np.var(np.sqrt(sobelx**2 + sobely**2))

    # 4. HOG Features (Histogram of Oriented Gradients)
    img_resized = cv2.resize(img, (96, 96))
    hog_feats = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False)

    base_feats = np.array([endings, minutiae_density, contrast, dissimilarity, homogeneity, lbp_var, grad_var])
    return np.hstack((base_feats, hog_feats))


# %%
# 2. Load Real vs. Altered Dataset from SOCOFing
socofing_root = os.path.join(ruizgara_socofing_path, "SOCOFing")
real_files = glob.glob(
    os.path.join(socofing_root, "Real", "*.BMP")
)

# %%
altered_files = []
for root, _, files in os.walk(os.path.join(socofing_root, "Altered")):
    for f in files:
        if f.upper().endswith(".BMP"):
            altered_files.append(os.path.join(root, f))


# %%
print(
    f"Found {len(real_files)} Real and "
    f"{len(altered_files)} Altered fingerprints."
)

# %%
# ------------------------------------------------------------
# Limit Altered fingerprints to 6,000
# ------------------------------------------------------------
rng = np.random.default_rng(42)

# %%
ALTERED_LIMIT = 6000

# %%
if len(altered_files) > ALTERED_LIMIT:
    altered_files = rng.choice(
        altered_files,
        size=ALTERED_LIMIT,
        replace=False
    ).tolist()

# %%
print(
    f"Using {len(real_files)} Real and "
    f"{len(altered_files)} Altered fingerprints."
)
X_list, y_list = [], []
n_samples = min(len(real_files), len(altered_files), 3000)

# %%
for p in real_files[:n_samples]:
    feat = extract_alteration_features(p)
    if feat is not None:
        X_list.append(feat)
        y_list.append(1) # Label 1 = Real

# %%
for p in altered_files[:n_samples]:
    feat = extract_alteration_features(p)
    if feat is not None:
        X_list.append(feat)
        y_list.append(0) # Label 0 = Altered

# %%
X, y = np.array(X_list), np.array(y_list)

# %%
# 3. Train & Evaluate Alteration Detector
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = MinMaxScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

clf_alt = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=8,
    l2_regularization=0.1,
    random_state=42
)
clf_alt.fit(X_tr_s, y_tr)
y_pred = clf_alt.predict(X_te_s)

# %%
print("\n=== FINGERPRINT ALTERATION DETECTION RESULTS ===")
print(f"Accuracy : {accuracy_score(y_te, y_pred):.2%}")
print(f"F1-Score : {f1_score(y_te, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_te, y_pred, target_names=["Altered", "Real"]))

# %%
import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.morphology import skeletonize

# %%
# Set seed for reproducibility
np.random.seed(42)
print("All libraries imported successfully!")

# %%
import os
import glob
import cv2
import numpy as np

# %%
from skimage.morphology import skeletonize
from skimage.feature import graycomatrix, graycoprops

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# %% [markdown]
# ============================================================
# PREPROCESSING
# ============================================================

# %%
def preprocess_image(image_path_or_img):
    """
    Preprocess either a fingerprint or iris image.

    Returns:
        img              : original grayscale image
        binary           : thresholded image
        skeleton_uint8   : skeletonized binary image
    """

    if isinstance(image_path_or_img, str):
        img = cv2.imread(
            image_path_or_img,
            cv2.IMREAD_GRAYSCALE
        )
    else:
        img = image_path_or_img.copy()

    if img is None:
        return None, None, None

    # Normalize intensity
    img_norm = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    # Reduce noise
    filtered = cv2.GaussianBlur(
        img_norm,
        (5, 5),
        0
    )

    # Binary image
    _, binary = cv2.threshold(
        filtered,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Skeletonization
    skeleton = skeletonize(binary > 0)

    skeleton_uint8 = (
        skeleton.astype(np.uint8) * 255
    )

    return img, binary, skeleton_uint8


# %% [markdown]
# ============================================================
# MINUTIAE DETECTION
# ============================================================

# %%
def detect_minutiae(skeleton_img):
    """
    Crossing Number (CN) minutiae detection.

    CN = 1 -> ridge ending
    CN = 3 -> ridge bifurcation
    """

    skel = (
        skeleton_img > 0
    ).astype(np.uint8)

    h, w = skel.shape

    endings = 0
    bifurcations = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):

            if skel[y, x] != 1:
                continue

            neighbors = [
                skel[y-1, x],
                skel[y-1, x+1],
                skel[y, x+1],
                skel[y+1, x+1],
                skel[y+1, x],
                skel[y+1, x-1],
                skel[y, x-1],
                skel[y-1, x-1]
            ]

            neighbors = np.array(neighbors)

            cn = 0.5 * np.sum(
                np.abs(
                    neighbors -
                    np.roll(neighbors, -1)
                )
            )

            if cn == 1:
                endings += 1

            elif cn == 3:
                bifurcations += 1

    return endings, bifurcations


# %% [markdown]
# ============================================================
# FEATURE EXTRACTION
# ============================================================

# %%
def extract_features(
    img,
    binary_img,
    skeleton_img
):
    """
    Extract 9 numerical features.

    Features:
        0  ridge endings
        1  bifurcations
        2  total minutiae
        3  minutiae density
        4  ridge density
        5  GLCM contrast
        6  GLCM homogeneity
        7  GLCM energy
        8  GLCM correlation
    """

    if (
        img is None or
        binary_img is None or
        skeleton_img is None
    ):
        return np.zeros(9, dtype=np.float32)

    h, w = img.shape

    total_area = float(h * w)

    # -------------------------
    # Minutiae
    # -------------------------

    endings, bifurcations = detect_minutiae(
        skeleton_img
    )

    total_minutiae = (
        endings + bifurcations
    )

    minutiae_density = (
        total_minutiae / total_area
    )

    # -------------------------
    # Ridge density
    # -------------------------

    ridge_density = (
        np.sum(skeleton_img > 0)
        / total_area
    )

    # -------------------------
    # GLCM
    # -------------------------

    # Quantize 256 levels -> 16 levels
    img_q = (
        img // 16
    ).astype(np.uint8)

    glcm = graycomatrix(
        img_q,
        distances=[1, 3],
        angles=[0, np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        "contrast"
    ).mean()

    homogeneity = graycoprops(
        glcm,
        "homogeneity"
    ).mean()

    energy = graycoprops(
        glcm,
        "energy"
    ).mean()

    correlation = graycoprops(
        glcm,
        "correlation"
    ).mean()

    return np.array([
        endings,
        bifurcations,
        total_minutiae,
        minutiae_density,
        ridge_density,
        contrast,
        homogeneity,
        energy,
        correlation
    ], dtype=np.float32)

# %%
# Fingerprint Preprocessing & Feature Extraction
def preprocess_fingerprint(image_path_or_img):
    """
    Fingerprint Preprocessing:
    1. Grayscale Normalization & Gaussian Filter
    2. Binarization (Otsu adaptive threshold)
    3. Thinning / Skeletonization
    """
    if isinstance(image_path_or_img, str):
        img = cv2.imread(image_path_or_img, cv2.IMREAD_GRAYSCALE)
    else:
        img = image_path_or_img.copy()

    if img is None:
        return None, None, None

    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    filtered = cv2.GaussianBlur(img_norm, (5, 5), 0)
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    skeleton = skeletonize(binary > 0)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    return img, binary, skeleton_uint8


# %%
def detect_minutiae(skeleton_img):
    """Crossing Number (CN) Minutiae Detection: CN=1 (Ending), CN=3 (Bifurcation)"""
    skel = (skeleton_img > 0).astype(np.uint8)
    h, w = skel.shape
    endings, bifurcations = 0, 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 1:
                neighbors = [skel[y-1, x], skel[y-1, x+1], skel[y, x+1], skel[y+1, x+1],
                             skel[y+1, x], skel[y+1, x-1], skel[y, x-1], skel[y-1, x-1]]
                cn = 0.5 * np.sum(np.abs(np.array(neighbors) - np.roll(neighbors, -1)))
                if cn == 1: endings += 1
                elif cn == 3: bifurcations += 1

    return endings, bifurcations


# %%
def extract_fingerprint_features(img, binary_img, skeleton_img):
    """
    Features:
    - Minutiae Counts: Endings, Bifurcations, Total density
    - Ridge Density: Skeleton pixels / Total area
    - GLCM Texture: Contrast, Homogeneity, Energy, Correlation
    - HOG Features: 288-dimensional spatial gradient representation
    """
    if img is None or binary_img is None or skeleton_img is None:
        return np.zeros(297)

    h, w = img.shape
    total_area = float(h * w)

    endings, bifurcations = detect_minutiae(skeleton_img)
    total_minutiae = endings + bifurcations
    minutiae_density = total_minutiae / total_area
    ridge_density = np.sum(skeleton_img > 0) / total_area

    # GLCM Features
    img_q = (img // 16).astype(np.uint8)
    glcm = graycomatrix(img_q, distances=[1, 3], angles=[0, np.pi/4], levels=16, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    base_feats = np.array([endings, bifurcations, total_minutiae, minutiae_density,
                           ridge_density, contrast, homogeneity, energy, correlation])

    # HOG Feature extraction
    from skimage.feature import hog
    img_resized = cv2.resize(img, (96, 96))
    hog_feats = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False)

    return np.hstack((base_feats, hog_feats))


# %%
# Feature Fusion Dataset Preparation
print("Preparing Gender Classification Dataset...")
iris_path = os.path.join(basnamhamedsalih_iris_datasetndgfi_path, "DataSet", "GFI")
socofing_real = os.path.join(ruizgara_socofing_path, "SOCOFing", "Real")

# %%
iris_labels = {}
with open(os.path.join(iris_path, "List_left_GFI.txt"), "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            iris_labels[parts[0]] = int(parts[1])

# %%
iris_males = [f for f, l in iris_labels.items() if l == 1]
iris_females = [f for f, l in iris_labels.items() if l == 0]

# %%
socofing_files = glob.glob(os.path.join(socofing_real, "*.BMP"))
fp_males = [f for f in socofing_files if "__M_" in os.path.basename(f)]
fp_females = [f for f in socofing_files if "__F_" in os.path.basename(f)]

# %%
num_males = min(len(iris_males), len(fp_males), 1000)
num_females = min(len(iris_females), len(fp_females), 1000)

# %%
paired_males = list(zip(iris_males[:num_males], fp_males[:num_males]))
paired_females = list(zip(iris_females[:num_females], fp_females[:num_females]))

# %%
X_iris_list, X_fp_list, y_list = [], [], []

# %%
for iris_f, fp_f in paired_males:
    i_path = os.path.join(iris_path, "NUND_left", "NUND_left", iris_f)
    i_img, i_bin, i_skel = preprocess_fingerprint(i_path)
    f_img, f_bin, f_skel = preprocess_fingerprint(fp_f)
    if i_img is not None and f_img is not None:
        X_iris_list.append(extract_fingerprint_features(i_img, i_bin, i_skel))
        X_fp_list.append(extract_fingerprint_features(f_img, f_bin, f_skel))
        y_list.append(1)

# %%
for iris_f, fp_f in paired_females:
    i_path = os.path.join(iris_path, "NUND_left", "NUND_left", iris_f)
    i_img, i_bin, i_skel = preprocess_fingerprint(i_path)
    f_img, f_bin, f_skel = preprocess_fingerprint(fp_f)
    if i_img is not None and f_img is not None:
        X_iris_list.append(extract_fingerprint_features(i_img, i_bin, i_skel))
        X_fp_list.append(extract_fingerprint_features(f_img, f_bin, f_skel))
        y_list.append(0)

# %%
X_iris = np.array(X_iris_list)
X_fp = np.array(X_fp_list)
y = np.array(y_list)

# %%
#Feature Fusion & Model Training
# 1. Train/Test Split (80/20)
indices = np.arange(len(y))
idx_tr, idx_te, y_train, y_test = train_test_split(indices, y, test_size=0.20, random_state=42, stratify=y)

# %%
# 2. Min-Max Normalization (Fit on Train, Transform Test - No Data Leakage!)
scaler_iris = MinMaxScaler()
X_iris_tr = scaler_iris.fit_transform(X_iris[idx_tr])
X_iris_te = scaler_iris.transform(X_iris[idx_te])

# %%
scaler_fp = MinMaxScaler()
X_fp_tr = scaler_fp.fit_transform(X_fp[idx_tr])
X_fp_te = scaler_fp.transform(X_fp[idx_te])

# %%
# 3. Concatenation Fusion: F_fusion = [F_iris | F_fp]
X_fusion_tr = np.hstack((X_iris_tr, X_fp_tr))
X_fusion_te = np.hstack((X_iris_te, X_fp_te))

# %%
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

class LateFusionStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features_single=297, random_state=42):
        self.n_features_single = n_features_single
        self.random_state = random_state
        self.clf_iris = None
        self.clf_fp = None
        self.meta_clf = None
        self.is_fusion = False
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        n_features = X.shape[1]
        if n_features == self.n_features_single * 2:
            self.is_fusion = True
            X_iris = X[:, :self.n_features_single]
            X_fp = X[:, self.n_features_single:]

            # Base models for Iris and Fingerprint
            self.clf_iris = SVC(kernel='rbf', C=10.0, probability=True, random_state=self.random_state)
            self.clf_fp = SVC(kernel='rbf', C=10.0, probability=True, random_state=self.random_state)

            # Out-of-fold predictions to prevent data leakage during meta-classifier training
            prob_iris_cv = cross_val_predict(self.clf_iris, X_iris, y, cv=5, method='predict_proba')[:, 1]
            prob_fp_cv = cross_val_predict(self.clf_fp, X_fp, y, cv=5, method='predict_proba')[:, 1]

            # Fit on entire training data
            self.clf_iris.fit(X_iris, y)
            self.clf_fp.fit(X_fp, y)

            # Fit logistic regression meta classifier on probability features
            X_meta = np.column_stack((prob_iris_cv, prob_fp_cv))
            self.meta_clf = LogisticRegression(random_state=self.random_state)
            self.meta_clf.fit(X_meta, y)
        else:
            self.is_fusion = False
            self.clf_iris = SVC(kernel='rbf', C=10.0, probability=True, random_state=self.random_state)
            self.clf_iris.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.is_fusion:
            X_iris = X[:, :self.n_features_single]
            X_fp = X[:, self.n_features_single:]
            prob_iris = self.clf_iris.predict_proba(X_iris)[:, 1]
            prob_fp = self.clf_fp.predict_proba(X_fp)[:, 1]
            X_meta = np.column_stack((prob_iris, prob_fp))
            return self.meta_clf.predict_proba(X_meta)
        else:
            return self.clf_iris.predict_proba(X)

    def predict(self, X):
        if self.is_fusion:
            prob = self.predict_proba(X)[:, 1]
            return (prob >= 0.5).astype(int)
        else:
            return self.clf_iris.predict(X)

classifiers = {
    "SVM (RBF Kernel)": SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=8, random_state=42),
    "Late Fusion Stacking": LateFusionStackingClassifier(n_features_single=297, random_state=42)
}

# %%
modalities = {
    "Model 1: Iris Only": (X_iris_tr, X_iris_te),
    "Model 2: Fingerprint Only": (X_fp_tr, X_fp_te),
    "Model 3: Iris + Fingerprint (Fusion)": (X_fusion_tr, X_fusion_te)
}

# %%
results = []
cms = {}
rocs = {}

# %%
for clf_name, clf in classifiers.items():
    for mod_name, (X_tr, X_te) in modalities.items():
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        key = f"{clf_name} - {mod_name}"
        cms[key], rocs[key] = cm, (fpr, tpr, roc_auc)
        results.append({"Classifier": clf_name, "Modality": mod_name, "Accuracy": acc,
                        "Precision": prec, "Recall": rec, "F1 Score": f1, "AUC": roc_auc})

# %%
results_df = pd.DataFrame(results)
print("=== EXPERIMENTAL RESULTS ===")
print(results_df.to_string(index=False))

# %%
#Evaluation & Comparative Graphs
sns.set_theme(style="whitegrid")

# %%
# 1. Accuracy Comparison Chart
plt.figure(figsize=(11, 5))
ax = sns.barplot(data=results_df, x="Modality", y="Accuracy", hue="Classifier", palette="viridis")
plt.title("Gender Classification: Single Modality vs. Multimodal Fusion", fontsize=14, fontweight='bold')
plt.ylim(0.5, 1.0)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
plt.tight_layout()
plt.savefig('accuracy_comparison.png')

# %%
# 2. Confusion Matrices (Late Fusion Stacking)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
target_keys = [k for k in cms.keys() if "Late Fusion Stacking" in k]
for idx, key in enumerate(target_keys):
    sns.heatmap(cms[key], annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
    axes[idx].set_title(key.split(" - ")[1], fontweight='bold')
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")
plt.suptitle("Confusion Matrices (Late Fusion Stacking Classifier)", fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('confusion_matrices.png')

# %%
# 3. ROC Curves
plt.figure(figsize=(8, 5))
for key, (fpr, tpr, roc_auc) in rocs.items():
    if "Late Fusion Stacking" in key:
        plt.plot(fpr, tpr, lw=2, label=f"{key.split(' - ')[1]} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Modality Comparison (Late Fusion Stacking)', fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curves.png')
print("Completed. Plots saved.")


# %% [markdown]
# ============================================================
# NATIVE FILE SELECTOR UPLOAD PIPELINE (WORKS IN ALL EDITORS)
# Step 1: Opens OS File Window to pick Fingerprint Image.
# Step 2: Checks Fingerprint Alteration. If Altered -> HALT.
# Step 3: If Real -> Opens OS File Window to pick Iris Image -> Gender Classification.
# ============================================================

# %%
# ============================================================
# NATIVE OS FILE PICKER DIALOG (NO WIDGET EXTENSIONS NEEDED)
# ============================================================
import os
import cv2
import numpy as np

def select_file_via_gui(title_prompt="Select Image File"):
    """Opens native OS File Picker Dialog window to select image file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()              # Hide main Tk window
        root.attributes('-topmost', True) # Bring dialog to front
        file_path = filedialog.askopenfilename(
            title=title_prompt,
            filetypes=[("Image Files (*.bmp;*.png;*.jpg;*.jpeg)", "*.bmp;*.png;*.jpg;*.jpeg"), ("All Files", "*.*")]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"⚠️ GUI Dialog error: {e}")
        return None

def run_interactive_biometric_pipeline():
    print("=" * 70)
    print("STEP 1: FINGERPRINT UPLOAD & ALTERATION CHECK")
    print("=" * 70)
    print("📢 Opening OS File Picker Window to select Fingerprint Image...")

    fp_path = select_file_via_gui("Step 1: Select Fingerprint Image (.bmp, .png, .jpg)")
    
    # Fallback to text prompt if GUI fails or returns empty
    if not fp_path:
        fp_path = input("Or enter Fingerprint Image path manually: ").strip().strip('"')
    
    if not fp_path or not os.path.exists(fp_path):
        print("⚠️ No valid fingerprint image selected. Exiting.")
        return

    print(f"--> Fingerprint Selected: '{fp_path}'")

    # Extract 295 alteration features
    feat_alt = extract_alteration_features(fp_path)
    if feat_alt is None:
        print("❌ ERROR: Failed to process fingerprint image.")
        return

    # Scale & predict alteration using trained clf_alt model
    feat_alt_scaled = scaler.transform([feat_alt])
    alt_pred = clf_alt.predict(feat_alt_scaled)[0]
    alt_prob = clf_alt.predict_proba(feat_alt_scaled)[0]

    # Label 0 = Altered, Label 1 = Real
    if alt_pred == 0:
        print("
" + "!" * 70)
        print(f"🚨 SECURITY ALERT: Fingerprint is ALTERED / SPOOFED!")
        print(f"   Altered Confidence: {alt_prob[0]:.2%}")
        print("🛑 EXECUTION HALTED: Access denied due to fingerprint alteration.")
        print("!" * 70)
        return

    print(f"✅ FINGERPRINT VERIFIED: REAL Fingerprint confirmed (Confidence: {alt_prob[1]:.2%}).")
    print("--> Proceeding to Step 2: Select Iris Image for Gender Classification...")

    # ------------------------------------------------------------
    # STEP 2: Iris File Selector & Multimodal Gender Classification
    # ------------------------------------------------------------
    print("
" + "=" * 70)
    print("STEP 2: IRIS UPLOAD & MULTIMODAL GENDER CLASSIFICATION")
    print("=" * 70)
    print("📢 Opening OS File Picker Window to select Iris Image...")

    iris_path = select_file_via_gui("Step 2: Select Iris Image (.bmp, .png, .jpg)")
    if not iris_path:
        iris_path = input("Or enter Iris Image path manually: ").strip().strip('"')

    if not iris_path or not os.path.exists(iris_path):
        print("⚠️ No valid iris image selected. Exiting.")
        return

    print(f"--> Iris Image Selected: '{iris_path}'")

    # Preprocess & extract 297 biometric features for both images
    i_img, i_bin, i_skel = preprocess_fingerprint(iris_path)
    f_img, f_bin, f_skel = preprocess_fingerprint(fp_path)

    if i_img is None or f_img is None:
        print("❌ ERROR: Unable to process Iris or Fingerprint image features.")
        return

    feat_iris = extract_fingerprint_features(i_img, i_bin, i_skel)
    feat_fp = extract_fingerprint_features(f_img, f_bin, f_skel)

    # Scale features
    feat_iris_s = scaler_iris.transform([feat_iris])
    feat_fp_s = scaler_fp.transform([feat_fp])

    # Multimodal feature fusion [F_iris | F_fp]
    feat_fusion = np.hstack((feat_iris_s, feat_fp_s))

    # Predict gender using Late Fusion Stacking model
    clf_fusion = classifiers["Late Fusion Stacking"]
    gender_pred = clf_fusion.predict(feat_fusion)[0]
    gender_prob = clf_fusion.predict_proba(feat_fusion)[0]

    gender_label = "Male 👦" if gender_pred == 1 else "Female 👧"
    confidence = gender_prob[gender_pred]

    print("
" + "*" * 70)
    print(f"🎉 MULTIMODAL GENDER CLASSIFICATION RESULT: {gender_label}")
    print(f"📊 Model Confidence: {confidence:.2%}")
    print("*" * 70)

# Run the interactive GUI file selector
run_interactive_biometric_pipeline()

