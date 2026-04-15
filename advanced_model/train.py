"""
Training script for the Advanced Hallucination Detector v2.

Usage:
    python -m advanced_model.train
    # or
    python advanced_model/train.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.data_loader import load_data
from advanced_model.detector import AdvancedHallucinationDetector


def main():
    print("=" * 70)
    print("  CATCH THE LIAR — Advanced Hallucination Detection v2")
    print("  Anti-overfitting rewrite: ~33 features, 3 models, 3 seeds")
    print("  References:")
    print("    [1] Kaczmarek et al., 'Fake or Real', arXiv:2507.13508")
    print("    [2] Sriramanan et al., 'LLM-Check', NeurIPS 2024")
    print("    [3] Yehuda et al., 'InterrogateLLM', arXiv:2403.02889")
    print("=" * 70)

    start_time = time.time()

    # Load data
    print("\nLoading training data...")
    df_train = load_data(base_path="data/train", labels_path="data/train.csv")
    print(f"  Training samples: {len(df_train)}")

    print("Loading test data...")
    df_test = load_data(base_path="data/test", labels_path=None)
    print(f"  Test samples: {len(df_test)}")

    # Initialize detector with anti-overfitting config
    detector = AdvancedHallucinationDetector(
        seeds=[42, 7, 99],             # 3 seeds (was 7)
        use_swap_consistency=True,      # keep — genuinely helpful
        use_language_model=True,        # keep — perplexity is strong signal
        bpe_vocab_size=3000,
        n_folds=5,
        n_repeats=3,                    # RepeatedStratifiedKFold
    )

    # Initial training
    detector.fit(df_train, df_test)
    print(f"\nInitial ensemble CV accuracy: {detector.cv_accuracy_:.4f}")

    # Conservative pseudo-labeling (1 round, 98% threshold)
    total_pseudo = detector.pseudo_label_refit(
        df_train=df_train,
        df_unlabeled=df_test,
        n_rounds=1,
        confidence_threshold=0.98,
        max_per_round=200,
    )

    print(f"\nFinal ensemble CV accuracy: {detector.cv_accuracy_:.4f}")
    print(f"Total pseudo-labeled samples used: {total_pseudo}")

    # Save model
    model_path = Path(__file__).resolve().parent / "model.pkl"
    detector.save(model_path)

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("Training complete!")


if __name__ == "__main__":
    main()
