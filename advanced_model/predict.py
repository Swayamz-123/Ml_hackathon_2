"""
Prediction script — generates submission.csv for Kaggle upload.

Usage:
    python -m advanced_model.predict
    # or
    python advanced_model/predict.py
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.data_loader import load_data
from advanced_model.detector import AdvancedHallucinationDetector


def main():
    print("=" * 70)
    print("  CATCH THE LIAR — Generating Predictions (v2)")
    print("=" * 70)

    model_path = Path(__file__).resolve().parent / "model.pkl"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run train.py first.")
        return

    print(f"\nLoading model from: {model_path}")
    detector = AdvancedHallucinationDetector.load(model_path)

    print("Loading test data...")
    df_test = load_data(base_path="data/test", labels_path=None)
    print(f"  Test samples: {len(df_test)}")

    print("\nGenerating predictions...")
    labels = detector.predict_labels(df_test)

    # Build submission
    submission = pd.DataFrame({
        "id": df_test["id"],
        "label": labels,
    })

    submission_path = Path(__file__).resolve().parent / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to: {submission_path}")
    print(f"  Total rows: {len(submission)}")
    print(f"  Label distribution: 1={int((labels==1).sum())}, 2={int((labels==2).sum())}")
    print("\nPrediction complete!")


if __name__ == "__main__":
    main()
