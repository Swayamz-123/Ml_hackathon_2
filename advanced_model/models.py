"""
Ensemble classifier for hallucination detection — v2 (Anti-Overfitting).

Design for 95 training samples:
  - Only 3 complementary models (LightGBM, SVM, LogReg L2)
  - Extreme regularization on all models
  - NO feature selection (only ~30 features, all curated)
  - NO meta-learner (simple probability averaging — safer at N=95)
  - RepeatedStratifiedKFold for stable OOF estimates
  - Swap-consistency handled in detector.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class EnsembleManager:
    """
    Multi-model ensemble with RepeatedStratifiedKFold OOF evaluation.

    All models are heavily regularized for the 95-sample regime.
    Final prediction = simple average of base-model probabilities
    (no meta-learner — too few samples for stacking to help).
    """

    def __init__(self, random_state: int = 42, n_folds: int = 5, n_repeats: int = 3):
        self.random_state = random_state
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.base_models = {}
        self._model_names = []

    def _build_models(self, seed: int) -> dict:
        models = {}

        if HAS_LGB:
            models["lgb"] = lgb.LGBMClassifier(
                n_estimators=50,       # few trees — prevent memorization
                max_depth=2,           # very shallow
                num_leaves=4,          # 2^2
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.6,
                reg_alpha=5.0,         # strong L1
                reg_lambda=10.0,       # strong L2
                min_child_samples=15,  # ~16% of training data per leaf
                random_state=seed,
                verbose=-1,
            )

        # SVM: very low C = very strong regularization
        models["svm"] = Pipeline([
            ("sc", StandardScaler()),
            ("svm", SVC(C=0.05, kernel="rbf", probability=True,
                        class_weight="balanced", random_state=seed)),
        ])

        # L2 logistic: low C, balanced
        models["lr_l2"] = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(
                C=0.05, solver="lbfgs", penalty="l2",
                class_weight="balanced", max_iter=2000,
                random_state=seed,
            )),
        ])

        return models

    def fit_and_predict_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """
        Fit with RepeatedStratifiedKFold and return OOF predictions.

        Returns (oof_probs, info_dict).
        """
        print(f"    Features: {X.shape[1]}, Samples: {X.shape[0]}")

        # --- CV with repeated stratified K-fold for stability ---
        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_folds,
            n_repeats=self.n_repeats,
            random_state=self.random_state * 31 + 17,
        )

        model_names = list(self._build_models(self.random_state).keys())
        self._model_names = model_names

        # Accumulate OOF predictions across repeats
        oof_sum = np.zeros((len(y), len(model_names)), dtype=np.float64)
        oof_count = np.zeros(len(y), dtype=np.float64)

        for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X, y)):
            fold_models = self._build_models(self.random_state + fold_idx)
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr = y[tr_idx]

            for i, name in enumerate(model_names):
                fold_models[name].fit(Xtr, ytr)
                p = fold_models[name].predict_proba(Xva)
                oof_sum[va_idx, i] += p[:, 1]

            oof_count[va_idx] += 1

        # Average OOF predictions across repeats
        oof = oof_sum / np.maximum(oof_count[:, None], 1)

        # --- Fit full-data models for prediction ---
        full_models = self._build_models(self.random_state)
        for name in model_names:
            full_models[name].fit(X, y)
        self.base_models = full_models

        # --- OOF accuracy: simple average of base-model probabilities ---
        avg_oof = oof.mean(axis=1)
        oof_acc = accuracy_score(y, (avg_oof >= 0.5).astype(int))

        # Per-model OOF for diagnostics
        per_model = {
            name: accuracy_score(y, (oof[:, i] >= 0.5).astype(int))
            for i, name in enumerate(model_names)
        }

        return oof, {"oof_accuracy": oof_acc, "model_names": model_names,
                     "per_model_oof": per_model}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(A is fake) using simple average of base models."""
        preds = []
        for name in self._model_names:
            p = self.base_models[name].predict_proba(X)
            preds.append(p[:, 1])
        return np.mean(np.vstack(preds), axis=0)
