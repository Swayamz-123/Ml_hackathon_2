"""
Prediction script -- generates submission files.

Usage:
    python -m advanced_model.predict
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.data_loader import load_data
from advanced_model.detector import AdvancedHallucinationDetector


def main():
    print("=" * 70)
    print("  CATCH THE LIAR -- Generating Predictions (v4)")
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
    # 'labels' is an array where 1 means Text A is fake, 2 means Text B is fake
    labels = detector.predict_labels(df_test)

    # 1. Internal format (shows exactly which text was hallucinated)
    internal_submission = pd.DataFrame({
        "id": [f"{int(i):04d}" for i in df_test["id"]],
        "hallucinated_label": labels,
    })

    # 2. Kaggle format (real_text_id: 1=A is real, 2=B is real)
    # If A is fake (label 1) -> B is real -> real_text_id = 2
    # If B is fake (label 2) -> A is real -> real_text_id = 1
    real_text_ids = []
    for lbl in labels:
        if lbl == 1:
            real_text_ids.append(2)  
        else:
            real_text_ids.append(1)  

    kaggle_submission = pd.DataFrame({
        "id": [f"{int(i):04d}" for i in df_test["id"]],
        "real_text_id": real_text_ids,
    })

    internal_path = PROJECT_ROOT / "submission_internal_format.csv"
    kaggle_path = PROJECT_ROOT / "submission_kaggle_upload.csv"

    internal_submission.to_csv(internal_path, index=False)
    kaggle_submission.to_csv(kaggle_path, index=False)

    print(f"\nSaved TWO files to your main folder (c:\\ml_hackathon_scse):")
    print(f"  1. KAGGLE UPLOAD: {kaggle_path.name}")
    print(f"     (Contains 'real_text_id' column required by competition)")
    print(f"  2. INTERNAL USE:  {internal_path.name}")
    print(f"     (Contains 'hallucinated_label' -> 1=A is fake, 2=B is fake)")
    
    print(f"\n  Total rows evaluated: {len(kaggle_submission)}")
    print(f"  Internal label distribution: 1={int((labels==1).sum())}, 2={int((labels==2).sum())}")
    print(f"  Kaggle 'real_text_id' distribution: 1={sum(1 for x in real_text_ids if x==1)}, 2={sum(1 for x in real_text_ids if x==2)}")
    print("\nPrediction complete! Upload 'submission_kaggle_upload.csv' to Kaggle.")


if __name__ == "__main__":
    main()
