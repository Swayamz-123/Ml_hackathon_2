# 🔍 Hallucination Detector

> Detecting AI-generated hallucinations in text pairs using a from-scratch NLP pipeline — Kneser-Ney language models, BPE tokenization, and a LightGBM ensemble.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Output](#output)
- [References](#references)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Overview

**🏆 Private Leaderboard Score:** 0.88225

This project tackles the problem of identifying which of two texts in an article pair contains AI-generated hallucinations. Given two versions of a document (`file_1.txt` and `file_2.txt`), the system predicts which one is the "real" (human-written) text and which has been altered or fabricated by a language model.

The pipeline is built entirely from scratch — no pretrained transformers — using classical NLP techniques: a Kneser-Ney trigram language model, Byte Pair Encoding (BPE) tokenization, 33 hand-crafted features, and a stacked ensemble of LightGBM, SVM, and Logistic Regression.

The core intuition is that hallucinated text tends to have subtly different statistical properties from genuine text — higher perplexity under a well-fitted language model, different lexical diversity, and inconsistent factual density. The feature set is designed to capture these signals, and the ensemble combines their evidence for a robust final prediction.

---

## Project Structure

```
.
├── data/                        # Training & test data (not included in repo)
│   ├── train/                   # 95 article folders with file_1.txt / file_2.txt
│   ├── test/                    # 1068 article folders
│   └── train.csv                # Labels (real_text_id)
│
├── advanced_model/              # Core pipeline
│   ├── detector.py              # Main orchestrator (AdvancedHallucinationDetector)
│   ├── features.py              # Feature extraction (33 features)
│   ├── language_model.py        # Kneser-Ney trigram LM + perplexity scoring
│   ├── models.py                # Ensemble trainer (LightGBM, SVM, LR)
│   ├── tokenizer.py             # BPE tokenizer (built from scratch)
│   ├── train.py                 # Training script
│   └── predict.py               # Submission generation script
│
├── data_loader/
│   └── data_loader.py           # Reads article folders and labels
│
├── _analyze.py                  # Compare two submission files (debugging)
├── requirements.txt             # Python dependencies
├── approach.md                  # Full technical write-up
└── README.md                    # This file
```

---

## How It Works

The detector runs in five phases:

1. **Tokenization** — A Byte Pair Encoding (BPE) tokenizer is trained from scratch directly on the corpus. This avoids the vocabulary mismatch that comes with off-the-shelf tokenizers and keeps subword representation faithful to the domain.

2. **Language Modelling** — A Kneser-Ney smoothed trigram model is fitted on the training corpus. At inference time, the model scores each text by perplexity — hallucinated text tends to use unusual token sequences that the model assigns lower probability, resulting in higher perplexity.

3. **Feature Extraction** — 33 features are computed per text pair, covering a range of signals: type-token ratio, sentence length variance, named entity density, punctuation patterns, perplexity delta between the two files, and more. See `features.py` for the full list.

4. **Swap Consistency** — Inspired by the InterrogateLLM paper, each pair is evaluated in both orders (`file_1` vs `file_2` and `file_2` vs `file_1`). Predictions that are inconsistent across swaps are flagged and resolved by confidence weighting, reducing order-dependent bias.

5. **Ensemble Classification** — LightGBM, SVM, and Logistic Regression are each trained on the feature vectors and combined via stacked generalisation with 5-fold cross-validation. The meta-learner learns which base model to trust in different regions of feature space.

For full technical details — feature definitions, mathematical formulas, hyperparameters, and design decisions — see [`approach.md`](approach.md).

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Swayamz-123/Ml_hackathon_2.git

```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the data

Place the competition data in the following structure:

```
data/
├── train/
│   ├── article_0/
│   │   ├── file_1.txt
│   │   └── file_2.txt
│   └── ...
├── test/
│   ├── article_0/
│   │   ├── file_1.txt
│   │   └── file_2.txt
│   └── ...
└── train.csv          # columns: id, real_text_id
```

---

## Usage

### Train the model

```bash
python -m advanced_model.train
```

Runs all five phases end-to-end. During training you'll see per-fold validation scores logged to stdout. The final trained ensemble is saved to `advanced_model/model.pkl`. Training on the full dataset typically takes a few minutes on a standard laptop.

### Generate a submission

```bash
python -m advanced_model.predict
```

Loads `advanced_model/model.pkl`, runs inference on the test set, and writes both output files (see [Output](#output) below).

---

## Output

| File | Description |
|------|-------------|
| `submission_kaggle_upload.csv` | Ready to upload to Kaggle — columns: `id`, `real_text_id` |
| `submission_internal_format.csv` | Debug format — `1` = file A is fake, `2` = file B is fake |

---

## References

1. Kaczmarek et al., *"Fake or Real: The Impostor Hunt in Texts for Space Operations"*, arXiv:2507.13508 (2025) — competition background.
2. Sriramanan et al., *"LLM-Check: Investigating Detection of Hallucinations in LLMs"*, NeurIPS 2024 — perplexity as a hallucination signal.
3. Yehuda et al., *"InterrogateLLM: Zero-Resource Hallucination Detection"*, arXiv:2403.02889 (2024) — swap consistency concept.

All implementations are original.

---

## Team

| Name | Contributions |
|------|--------------|
| **Harshit Kumar** | Architecture design, feature engineering, language model implementation |
| **Swayam Agarwal** | Ensemble training, cross-validation, swap consistency |
| **Rishav Kashyap** | Data loading, pseudo-labelling, submission generation |

---

## Acknowledgements

Thanks to the competition organisers for the challenging problem, and to the open-source community behind `scikit-learn`, `LightGBM`, `pandas`, and `numpy`.
