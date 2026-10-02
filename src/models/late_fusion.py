"""
Late Fusion Stacking Classifier Implementation.
Preserves the exact architecture, out-of-fold probability estimation (to prevent data leakage),
and inference logic from the source-of-truth notebook.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class LateFusionStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Late Fusion Stacking Classifier:
    - If X has n_features == n_features_single * 2 (e.g. 594):
      - First n_features_single are Iris features.
      - Next n_features_single are Fingerprint features.
      - Base models: SVC(kernel='rbf', C=10.0, probability=True) for Iris and Fingerprint.
      - 5-fold cross_val_predict to generate out-of-fold probability predictions on training set.
      - Meta-classifier: LogisticRegression trained on out-of-fold probability features [P(Iris), P(FP)].
    - If X is single modality:
      - Single base SVC(kernel='rbf', C=10.0, probability=True).
    """
    def __init__(self, n_features_single: int = 297, random_state: int = 42):
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
            self.clf_iris = SVC(
                kernel='rbf',
                C=10.0,
                probability=True,
                random_state=self.random_state
            )
            self.clf_fp = SVC(
                kernel='rbf',
                C=10.0,
                probability=True,
                random_state=self.random_state
            )

            # Out-of-fold predictions to prevent data leakage during meta-classifier training
            prob_iris_cv = cross_val_predict(
                self.clf_iris, X_iris, y, cv=5, method='predict_proba'
            )[:, 1]
            prob_fp_cv = cross_val_predict(
                self.clf_fp, X_fp, y, cv=5, method='predict_proba'
            )[:, 1]

            # Fit on entire training data
            self.clf_iris.fit(X_iris, y)
            self.clf_fp.fit(X_fp, y)

            # Fit logistic regression meta classifier on probability features
            X_meta = np.column_stack((prob_iris_cv, prob_fp_cv))
            self.meta_clf = LogisticRegression(random_state=self.random_state)
            self.meta_clf.fit(X_meta, y)
        else:
            self.is_fusion = False
            self.clf_iris = SVC(
                kernel='rbf',
                C=10.0,
                probability=True,
                random_state=self.random_state
            )
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
