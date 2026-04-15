"""
Advanced Hallucination Detector — Main Pipeline v2 (Anti-Overfitting).

Architecture:
  Phase 1 (unsupervised, run once):
    1a. BPE Tokenizer        — subword vocab from scratch (for LM)
    1b. KN Language Model    — 3-gram trained on CLEAN texts only
    1c. TF-IDF vectorizer    — fit on ALL texts for cosine similarity

  Phase 2 (feature extraction):
    Tower 1: Handcrafted linguistic features (~28 features)
    Tower 2: KN-LM perplexity (4 features)
    Tower 3: TF-IDF cosine similarity (1 feature)
    Total: ~33 features

  Phase 3 (ensemble):
    3 regularized classifiers (LGB, SVM, LR-L2)
    RepeatedStratified 5-fold CV → honest OOF accuracy
    Simple average (no meta-learner)

  Phase 4 (swap consistency):
    Average P(A fake | original) with 1 - P(A fake | swapped)

  Phase 5 (pseudo-labeling, optional, conservative):
    1 round with 98% confidence threshold

Changes from v1:
  - Dropped skip-gram embeddings (too noisy on 190 texts)
  - Dropped TF-IDF SVD (202 features → 1 scalar cosine)
  - Reduced features from ~404 to ~33
  - Reduced seeds from 7 to 3
  - Simplified ensemble (3 models, no meta-learner)
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .tokenizer import BPETokenizer
from .features import build_pair_feature_matrix, TfidfCosineSimilarity
from .language_model import LanguageModelFeatureExtractor
from .models import EnsembleManager


class AdvancedHallucinationDetector:

    def __init__(
        self,
        seeds=None,
        use_swap_consistency: bool = True,
        use_language_model: bool = True,
        bpe_vocab_size: int = 3000,
        n_folds: int = 5,
        n_repeats: int = 3,
    ):
        self.seeds = seeds or [42, 7, 99]
        self.use_swap_consistency = use_swap_consistency
        self.use_language_model = use_language_model
        self.bpe_vocab_size = bpe_vocab_size
        self.n_folds = n_folds
        self.n_repeats = n_repeats

        # Populated during fit
        self.tokenizer = None
        self.lm_extractor = None
        self.tfidf_sim = None
        self.ensemble_models = []

        self.cv_accuracy_ = None
        self.individual_seed_accuracies_ = []
        self._fitted = False

    # -----------------------------------------------------------------------
    # Label helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def labels_to_binary(labels):
        """1=A fake → 1, 2=B fake → 0"""
        return (np.asarray(labels) == 1).astype(int)

    @staticmethod
    def binary_to_labels(binary):
        return np.where(np.asarray(binary) == 1, 1, 2)

    # -----------------------------------------------------------------------
    # Data helpers
    # -----------------------------------------------------------------------
    def _all_texts(self, df_train, df_test=None):
        texts = (list(df_train["summary_A"].fillna("")) +
                 list(df_train["summary_B"].fillna("")))
        if df_test is not None:
            texts += (list(df_test["summary_A"].fillna("")) +
                      list(df_test["summary_B"].fillna("")))
        return texts

    def _clean_texts(self, df_train):
        """Return the non-hallucinated text from each training pair."""
        clean = []
        for _, row in df_train.iterrows():
            if row.get("label") == 1:
                clean.append(str(row["summary_B"]))   # B is clean
            elif row.get("label") == 2:
                clean.append(str(row["summary_A"]))   # A is clean
        return clean

    # -----------------------------------------------------------------------
    # Phase 1: unsupervised components (run once, cached)
    # -----------------------------------------------------------------------
    def _fit_phase1(self, df_train, df_test=None):
        all_texts   = self._all_texts(df_train, df_test)
        clean_texts = self._clean_texts(df_train)

        print("\n[Phase 1] Unsupervised pre-training...")

        # 1a. BPE tokenizer (used by LM internally)
        t0 = time.time()
        self.tokenizer = BPETokenizer(vocab_size=self.bpe_vocab_size)
        self.tokenizer.fit(all_texts)
        print(f"  BPE tokenizer: vocab={self.tokenizer.vocab_len}, {time.time()-t0:.1f}s")

        # 1b. KN language model — CLEAN TEXTS ONLY
        if self.use_language_model:
            t0 = time.time()
            self.lm_extractor = LanguageModelFeatureExtractor(n=3)
            self.lm_extractor.fit(clean_texts=clean_texts)
            print(f"  3-gram KN-LM: {len(clean_texts)} clean texts, {time.time()-t0:.1f}s")

        # 1c. TF-IDF vectorizer — fit on ALL texts
        t0 = time.time()
        self.tfidf_sim = TfidfCosineSimilarity(max_features=5000)
        self.tfidf_sim.fit(all_texts)
        print(f"  TF-IDF vectorizer: fit on {len(all_texts)} texts, {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------
    def _build_X(self, text_a, text_b):
        """Build full feature matrix for a list of pairs."""
        t0 = time.time()
        X = build_pair_feature_matrix(text_a, text_b)
        print(f"    Tower 1 (linguistic): {X.shape[1]} feats, {time.time()-t0:.1f}s")

        # Tower 2: LM perplexity
        if self.use_language_model and self.lm_extractor is not None:
            t0 = time.time()
            lm_feats = np.vstack([
                self.lm_extractor.extract_features(a, b)
                for a, b in zip(text_a, text_b)
            ])
            X = np.hstack([X, lm_feats])
            print(f"    Tower 2 (LM perplexity): +{lm_feats.shape[1]} feats, {time.time()-t0:.1f}s")

        # Tower 3: TF-IDF cosine similarity (1 scalar)
        if self.tfidf_sim is not None:
            t0 = time.time()
            cos_sim = self.tfidf_sim.similarity(text_a, text_b).reshape(-1, 1)
            X = np.hstack([X, cos_sim])
            print(f"    Tower 3 (TF-IDF cosine): +1 feat, {time.time()-t0:.1f}s")

        print(f"    Total features: {X.shape[1]}")
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # -----------------------------------------------------------------------
    # Phase 2+3: feature extraction + ensemble training
    # -----------------------------------------------------------------------
    def _fit_phases2_and_3(self, df_train, df_test=None):
        text_a = df_train["summary_A"].fillna("").tolist()
        text_b = df_train["summary_B"].fillna("").tolist()
        y = self.labels_to_binary(df_train["label"].to_numpy())

        print("\n[Phase 2] Feature extraction...")
        X = self._build_X(text_a, text_b)

        print(f"\n[Phase 3] Training {len(self.seeds)}-seed ensemble "
              f"({self.n_folds}-fold × {self.n_repeats}-repeat CV)...")
        self.ensemble_models = []
        self.individual_seed_accuracies_ = []

        for si, seed in enumerate(self.seeds):
            t0 = time.time()
            em = EnsembleManager(
                random_state=seed,
                n_folds=self.n_folds,
                n_repeats=self.n_repeats,
            )
            _, info = em.fit_and_predict_oof(X, y)
            self.ensemble_models.append(em)
            self.individual_seed_accuracies_.append(info["oof_accuracy"])

            per = info["per_model_oof"]
            per_str = "  ".join(f"{k}={v:.3f}" for k, v in per.items())
            print(f"  Seed {si+1}/{len(self.seeds)} (seed={seed}): "
                  f"OOF={info['oof_accuracy']:.4f}  [{per_str}]  {time.time()-t0:.1f}s")

        self.cv_accuracy_ = float(np.mean(self.individual_seed_accuracies_))
        std = float(np.std(self.individual_seed_accuracies_))
        print(f"\n  Mean OOF accuracy: {self.cv_accuracy_:.4f} ± {std:.4f}")
        print(f"  Per-seed: {[f'{a:.4f}' for a in self.individual_seed_accuracies_]}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def fit(self, df_train, df_test=None):
        print("=" * 65)
        print("  HALLUCINATION DETECTOR v2 — TRAINING")
        print("=" * 65)
        self._fit_phase1(df_train, df_test)
        self._fit_phases2_and_3(df_train, df_test)
        self._fitted = True
        return self

    def predict_proba_a_fake(self, df_data):
        """P(text A is hallucinated) with swap-consistency enforcement."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        def _proba(df):
            ta = df["summary_A"].fillna("").tolist()
            tb = df["summary_B"].fillna("").tolist()
            X = self._build_X(ta, tb)
            preds = [em.predict_proba(X) for em in self.ensemble_models]
            return np.mean(np.vstack(preds), axis=0)

        p = _proba(df_data)

        if self.use_swap_consistency:
            df_sw = pd.DataFrame({
                "summary_A": df_data["summary_B"].fillna(""),
                "summary_B": df_data["summary_A"].fillna(""),
            })
            p_sw = _proba(df_sw)
            p = 0.5 * (p + (1.0 - p_sw))

        return p

    def predict_labels(self, df_data):
        p = self.predict_proba_a_fake(df_data)
        return self.binary_to_labels((p >= 0.5).astype(int))

    def pseudo_label_refit(
        self,
        df_train,
        df_unlabeled,
        n_rounds: int = 1,
        confidence_threshold: float = 0.98,
        max_per_round: int = 200,
    ) -> int:
        """
        Conservative semi-supervised pseudo-labeling.
        Only re-runs Phases 2+3 — Phase 1 is cached.
        """
        print(f"\n[Phase 5] Pseudo-labeling ({n_rounds} round(s), "
              f"threshold={confidence_threshold:.2f})...")
        total_added = 0
        current_train = df_train.copy()

        for rnd in range(n_rounds):
            thr_hi = confidence_threshold
            thr_lo = 1.0 - thr_hi
            print(f"\n  Round {rnd+1}/{n_rounds}: "
                  f"confidence threshold [{thr_lo:.2f}, {thr_hi:.2f}]")

            p = self.predict_proba_a_fake(df_unlabeled)
            mask = (p <= thr_lo) | (p >= thr_hi)
            idx = np.where(mask)[0]

            if len(idx) == 0:
                print("    No confident predictions — stopping.")
                break

            conf = np.abs(p[idx] - 0.5)
            keep = idx[np.argsort(-conf)[:max_per_round]]

            df_pseudo = df_unlabeled.iloc[keep][["summary_A", "summary_B"]].copy()
            df_pseudo["label"] = np.where(p[keep] >= 0.5, 1, 2)
            total_added += len(df_pseudo)
            print(f"    Added {len(df_pseudo)} pseudo-labeled samples "
                  f"(total: {total_added})")

            current_train = pd.concat(
                [current_train[["summary_A", "summary_B", "label"]], df_pseudo],
                ignore_index=True,
            )
            self._fit_phases2_and_3(current_train, df_unlabeled)

        print(f"\n  Total pseudo-labeled: {total_added}")
        return total_added

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved: {path}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
