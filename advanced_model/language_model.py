"""
N-gram Language Model with Modified Kneser-Ney Smoothing — v2.

Extracts only 4 perplexity-based features (down from 28).
Surprisal distribution stats were dropped — they need far more training
data to be stable, and added mostly noise on 95 clean texts.

Key insight from LLM-Check [2]:
  Hallucinated text has HIGHER perplexity when scored by a model trained
  on clean data only.  The perplexity ratio (A/B) is the most
  discriminative single feature from this tower.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?|[^\s\w]")

BOS = "<BOS>"
EOS = "<EOS>"


def _tokenize(text: str) -> list[str]:
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


class KneserNeyLM:
    """
    N-gram language model with modified Kneser-Ney smoothing.
    Built entirely from scratch — no NLTK, no pretrained components.
    """

    def __init__(self, n: int = 3, discount: float = 0.75):
        self.n = n
        self.discount = discount
        self._ngram_counts: dict[int, Counter] = {}
        self._continuation_counts: dict[int, Counter] = {}
        self._context_counts: dict[int, Counter] = {}
        self._vocab: set[str] = set()
        self._fitted = False

    def fit(self, texts: list[str]) -> "KneserNeyLM":
        """Train the language model on a list of texts."""
        self._ngram_counts = {i: Counter() for i in range(1, self.n + 1)}
        self._continuation_counts = {i: defaultdict(set) for i in range(2, self.n + 1)}
        self._context_counts = {i: Counter() for i in range(2, self.n + 1)}

        for text in texts:
            tokens = [BOS] * (self.n - 1) + _tokenize(text) + [EOS]
            self._vocab.update(tokens)

            for order in range(1, self.n + 1):
                for i in range(len(tokens) - order + 1):
                    ngram = tuple(tokens[i:i + order])
                    self._ngram_counts[order][ngram] += 1

                    if order >= 2:
                        context = ngram[:-1]
                        word = ngram[-1]
                        self._context_counts[order][context] += 1
                        self._continuation_counts[order][word].add(context)

        self._vocab_size = max(len(self._vocab), 1)

        # Precompute unique_following for O(1) lookup
        self._unique_following: dict[int, dict] = {}
        for order in range(2, self.n + 1):
            uf: dict = defaultdict(int)
            for ngram in self._ngram_counts[order]:
                if self._ngram_counts[order][ngram] > 0:
                    uf[ngram[:-1]] += 1
            self._unique_following[order] = dict(uf)

        self._fitted = True
        return self

    def _kn_prob(self, word: str, context: tuple[str, ...], order: int) -> float:
        if order == 1:
            if order < self.n and 2 in self._continuation_counts:
                cont = len(self._continuation_counts[2].get(word, set()))
                total_cont = sum(len(s) for s in self._continuation_counts[2].values())
                return max(cont, 1) / max(total_cont, 1)
            else:
                total = sum(self._ngram_counts[1].values())
                return max(self._ngram_counts[1].get((word,), 0), 1) / max(total, 1)

        ngram = context + (word,)
        ngram_count = self._ngram_counts[order].get(ngram, 0)
        context_count = self._context_counts[order].get(context, 0)

        if context_count == 0:
            return self._kn_prob(word, context[1:], order - 1)

        first_term = max(ngram_count - self.discount, 0) / context_count
        unique_following = self._unique_following.get(order, {}).get(context, 0)
        lambda_weight = (self.discount * unique_following) / context_count
        lower_prob = self._kn_prob(word, context[1:], order - 1)

        return first_term + lambda_weight * lower_prob

    def log_prob(self, word: str, context: tuple[str, ...]) -> float:
        ctx = context[-(self.n - 1):] if len(context) >= self.n - 1 else context
        prob = self._kn_prob(word, ctx, min(len(ctx) + 1, self.n))
        return math.log(prob + 1e-15)

    def text_log_probs(self, text: str) -> list[float]:
        tokens = [BOS] * (self.n - 1) + _tokenize(text) + [EOS]
        log_probs = []
        for i in range(self.n - 1, len(tokens)):
            ctx = tuple(tokens[max(0, i - self.n + 1):i])
            lp = self.log_prob(tokens[i], ctx)
            log_probs.append(lp)
        return log_probs

    def perplexity(self, text: str) -> float:
        log_probs = self.text_log_probs(text)
        if not log_probs:
            return float("inf")
        avg_log_prob = sum(log_probs) / len(log_probs)
        return math.exp(-avg_log_prob)


class LanguageModelFeatureExtractor:
    """
    Extracts 4 perplexity-based features from text pairs.

    Trains a 3-gram KN-LM on clean texts only, then computes:
      - perplexity_a, perplexity_b  (raw)
      - perplexity_diff  (A - B)
      - perplexity_ratio  (log(A/B))
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.lm: KneserNeyLM | None = None
        self._fitted = False

    def fit(self, clean_texts: list[str]) -> "LanguageModelFeatureExtractor":
        self.lm = KneserNeyLM(n=self.n, discount=0.75)
        self.lm.fit(clean_texts)
        self._fitted = True
        return self

    def extract_features(self, text_a: str, text_b: str) -> np.ndarray:
        """Extract 4 perplexity features for a text pair."""
        if not self._fitted:
            raise RuntimeError("LM feature extractor not fitted.")

        ppl_a = self.lm.perplexity(text_a)
        ppl_b = self.lm.perplexity(text_b)
        ppl_diff = ppl_a - ppl_b
        ppl_ratio = _safe_log_ratio(ppl_a, ppl_b)

        return np.array([ppl_a, ppl_b, ppl_diff, ppl_ratio], dtype=np.float64)

    def get_feature_names(self) -> list[str]:
        return ["lm_ppl_a", "lm_ppl_b", "lm_ppl_diff", "lm_ppl_ratio"]


def _safe_log_ratio(a: float, b: float) -> float:
    if b <= 0 or a <= 0:
        return 0.0
    return math.log(a / b + 1e-10)
