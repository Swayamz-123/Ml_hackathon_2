# Approach: Hallucination Detection in Text Pairs

## Problem Statement

Given two summaries of the same scientific article — Text A and Text B — the task is to identify which one has been hallucinated by an AI model. One text is authentic (human-written or faithfully generated), and the other contains fabricated or distorted content introduced by a manipulated language model.

**Output convention:**
- `1` → Text A is hallucinated
- `2` → Text B is hallucinated

### Constraints
- No pretrained models permitted (BERT, GPT, Word2Vec, etc.)
- All representations must be built from scratch
- Only 95 labeled training samples available

---

## Core Idea

With only 95 labeled examples, any deep learning approach would immediately overfit. We therefore reframe the problem: rather than classifying a single text, we treat it as a **structured pairwise comparison** — asking "which of these two texts is more consistent with natural, human-written language?"

This reframing is powerful because it sidesteps the need for large labeled datasets. Many of our signals are unsupervised (derived from the texts themselves), and the labeled data is only needed to learn how to *weight* those signals.

> **Core hypothesis:** Hallucinated text exhibits systematic statistical, linguistic, and structural differences from authentic text — higher predictability under a language model, weaker factual grounding, and looser semantic coherence.

---

## Pipeline Overview

The system runs in five sequential phases:

```
Text A + Text B
      │
      ▼
Phase 1 — Unsupervised Language Learning
      │   (BPE tokenizer, Kneser-Ney LM, TF-IDF)
      ▼
Phase 2 — Feature Engineering
      │   (33 features per pair)
      ▼
Phase 3 — Ensemble Classification
      │   (LightGBM + SVM + Logistic Regression)
      ▼
Phase 4 — Swap Consistency Check
      │   (bias elimination)
      ▼
Phase 5 — Pseudo-Labelling
      │   (dataset expansion)
      ▼
Final Prediction
```

---

## Phase 1 — Unsupervised Language Learning

Before any classification, we build a statistical model of what "normal" text looks like. This phase is entirely unsupervised — no labels are used.

### 1.1 Byte Pair Encoding (BPE) Tokenizer

We train a BPE tokenizer from scratch on the full corpus. BPE works by iteratively merging the most frequent pairs of characters or subwords until a target vocabulary size is reached.

**Why BPE?**
Standard word-level tokenization struggles with rare words and domain-specific terminology. BPE handles unknown and compound words naturally by breaking them into familiar subword units — without relying on any external vocabulary.

### 1.2 Kneser-Ney Trigram Language Model

We fit a trigram language model using Kneser-Ney smoothing on the training corpus. Given the previous two tokens, the model estimates the probability of the next token:

$$P(w_n \mid w_{n-2},\, w_{n-1})$$

Kneser-Ney smoothing handles the zero-probability problem (unseen trigrams) by redistributing probability mass to lower-order n-grams in a principled way — it is widely regarded as the best-performing classical smoothing technique.

**Trained only on authentic text**, this model effectively learns the statistical fingerprint of genuine writing. At inference, text that deviates from this fingerprint receives a high perplexity score.

### 1.3 TF-IDF Vectorization

Each text is also represented as a TF-IDF vector, which weights terms by how informative they are relative to the corpus. This representation is used for computing semantic similarity between the two texts in a pair.

---

## Phase 2 — Feature Engineering

Each text pair is converted into a vector of **33 numerical features**. These features are grouped into four categories.

### 2.1 Linguistic Features

These are computed independently for Text A and Text B, then combined as both a **delta** (A − B) and a **mean** ((A + B) / 2):

$$\text{delta} = f(A) - f(B), \qquad \text{mean} = \frac{f(A) + f(B)}{2}$$

Using deltas rather than absolute values makes the features **domain-independent** — the model learns which text is the odd one out, rather than memorising absolute thresholds that may shift across topics.

Features in this group include:

| Feature | What it captures |
|---|---|
| Word count | Length differences between summaries |
| Mean sentence length | Structural complexity |
| Type-Token Ratio (TTR) | Vocabulary richness and diversity |
| Flesch Reading Ease | Overall readability |
| Hedging word frequency | Uncertainty markers (*possibly*, *likely*, *may*) |
| Vague quantifier frequency | Imprecision markers (*many*, *several*, *some*) |
| Specificity score | Density of numbers, proper nouns, and acronyms |

**Key insight:** Hallucinated text tends to be vaguer — fewer specific numbers and named entities, more hedging language — because the model is fabricating details it doesn't have.

### 2.2 Cross-Text Comparison Features

These features measure how similar the two summaries are to each other:

| Feature | Description |
|---|---|
| Unigram Jaccard Similarity | Word-level overlap between A and B |
| Bigram Jaccard Similarity | Phrase-level overlap |
| Named Entity Overlap | Shared people, places, and organisations |
| Numeric Value Overlap | Shared numbers and quantities |

**Key insight:** Numeric overlap is a particularly strong signal. Hallucinated summaries frequently alter or invent specific numbers (dates, measurements, statistics) while keeping the surrounding prose plausible-sounding. A mismatch in numbers between A and B is a reliable indicator that one has been tampered with.

### 2.3 Perplexity Features

Perplexity measures how "surprised" the language model is by a piece of text. Formally, for a sequence of N tokens:

$$\text{Perplexity} = e^{-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i)}$$

Lower perplexity means the text is more predictable given the model. Features derived from perplexity:

- Perplexity of Text A
- Perplexity of Text B
- Absolute difference (A − B)
- Ratio (A / B)

**Key insight:** AI-generated text, particularly hallucinated content, tends to be smoother and more predictable than natural human writing. It avoids unusual constructions and stays close to high-probability continuations. This makes perplexity an effective discriminator — hallucinated text typically scores *lower* perplexity under our language model.

### 2.4 Semantic Similarity

The TF-IDF cosine similarity between Text A and Text B:

$$\text{similarity}(A, B) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \cdot \|\vec{B}\|}$$

**Key insight:** While the two summaries describe the same article, a hallucinated summary often drifts semantically — introducing unrelated concepts or losing key themes. Lower cosine similarity between the pair is a soft signal of hallucination.

---

## Phase 3 — Ensemble Classification

We train three classifiers on the 33-dimensional feature vectors and combine their predictions.

### Models

**LightGBM (Gradient Boosted Trees)**
Captures non-linear interactions between features. For example, the combination of high perplexity delta *and* low numeric overlap may be a stronger signal than either feature alone. Regularised with `min_child_samples` and `max_depth` to prevent overfitting on the small dataset.

**Support Vector Machine (RBF kernel)**
Finds the maximum-margin decision boundary in feature space. SVMs are effective in high-dimensional settings with few samples, as they are naturally resistant to overfitting when the margin is maximised correctly.

**Logistic Regression (L2 regularisation)**
A stable linear baseline. Its simplicity acts as a regulariser on the overall ensemble — it prevents the other models from over-relying on complex feature interactions that may not generalise.

### Training Strategy

All three models are trained using **Repeated Stratified 5-Fold Cross-Validation**. Final predictions are the average of all three model outputs:

$$P_{\text{final}} = \frac{1}{3}\left(P_{\text{LGB}} + P_{\text{SVM}} + P_{\text{LR}}\right)$$

Each model contributes a complementary perspective — trees capture interactions, SVMs capture geometry, and linear models provide stability.

---

## Phase 4 — Swap Consistency

### Problem
Even with good features, a classifier trained on small data can develop **positional bias** — systematically favouring whichever text appears first (as Text A) simply due to presentation order in the training data.

### Solution
Every pair is evaluated twice:
- Forward: predict using (A, B)
- Swapped: predict using (B, A)

The final prediction symmetrises these two scores:

$$P_{\text{final}} = \frac{1}{2}\left(P(A, B) + \left(1 - P(B, A)\right)\right)$$

If the model is consistent, both evaluations agree and reinforce each other. If the model is biased by position, the swap exposes and cancels it. This makes the predictions **order-invariant** by construction.

---

## Phase 5 — Pseudo-Labelling

### Motivation
95 training samples is a severe constraint. Pseudo-labelling is a semi-supervised technique for safely expanding the training set using the test data itself.

### Procedure
1. Train the ensemble on the 95 labelled samples.
2. Run inference on all 1068 test pairs.
3. Select test samples where the model is highly confident:
   $$P < 0.02 \quad \text{or} \quad P > 0.98$$
4. Treat these as labelled samples and add them to the training set.
5. Retrain the full ensemble on the expanded dataset.

The strict confidence threshold (0.02 / 0.98) ensures only near-certain predictions are included, minimising the risk of injecting noisy labels. In practice this typically adds several hundred reliable samples, meaningfully improving generalisation on the remaining test set.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Pairwise comparison framing | Eliminates the need for absolute thresholds; model learns relative differences |
| Delta + mean feature encoding | Makes features domain-independent and topic-agnostic |
| Kneser-Ney over simple n-gram | Handles unseen sequences without collapsing to zero probability |
| BPE over word tokenisation | Handles rare, compound, and domain-specific terms robustly |
| Swap consistency | Eliminates order bias, enforces symmetric predictions |
| Conservative pseudo-labelling | Expands data with minimal label noise |

---

## Conclusion

This system demonstrates that hallucination detection can be solved effectively without deep learning, even under strict data constraints. By combining a hand-crafted language model, a carefully designed feature space, and a robust ensemble strategy, the pipeline achieves strong generalisation from only 95 labelled examples.

The approach is lightweight, fully interpretable, and auditable — every prediction can be traced back to specific features and model scores. This makes it well-suited not just for this competition, but as a template for low-resource NLP classification tasks in general.

For implementation details, refer to the source files in `advanced_model/`.
For implementation details, refer to the source files in `advanced_model/`.
