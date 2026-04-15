"""
Custom Skip-gram Word Embeddings -- trained from scratch.

Implements Skip-gram with Negative Sampling (SGNS) in pure NumPy.
No Word2Vec, GloVe, FastText, or any pretrained embeddings are used.

Inspired by InterrogateLLM [3]:
  - Embeddings enable intra-text consistency scoring
  - Sentence-level embeddings via TF-IDF weighted averaging
  - Cosine similarity features for semantic drift analysis

References:
  Mikolov et al., "Distributed Representations of Words and Phrases
  and their Compositionality", NeurIPS 2013.
"""

import re
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?")


def _tokenize_simple(text: str) -> list[str]:
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


class SkipGramEmbeddings:
    """
    Skip-gram with Negative Sampling, implemented from scratch in NumPy.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    window : int
        Context window size (one side).
    min_count : int
        Minimum word frequency to include in vocabulary.
    neg_samples : int
        Number of negative samples per positive pair.
    epochs : int
        Training epochs over the corpus.
    lr : float
        Learning rate.
    subsample_t : float
        Subsampling threshold for frequent words.
    """

    def __init__(
        self,
        dim: int = 100,
        window: int = 5,
        min_count: int = 2,
        neg_samples: int = 5,
        epochs: int = 10,
        lr: float = 0.025,
        subsample_t: float = 1e-4,
    ):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.neg_samples = neg_samples
        self.epochs = epochs
        self.lr = lr
        self.subsample_t = subsample_t

        self.word2id: dict[str, int] = {}
        self.id2word: dict[int, str] = {}
        self.W: np.ndarray | None = None  # target embeddings
        self.C: np.ndarray | None = None  # context embeddings
        self._neg_table: np.ndarray | None = None
        self._fitted = False

    def _build_vocab(self, tokenized_corpus: list[list[str]]):
        counter: Counter = Counter()
        for tokens in tokenized_corpus:
            counter.update(tokens)

        vocab = [(w, c) for w, c in counter.items() if c >= self.min_count]
        vocab.sort(key=lambda x: -x[1])

        self.word2id = {w: i for i, (w, _) in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self._freqs = np.array([c for _, c in vocab], dtype=np.float64)
        self._total = self._freqs.sum()

        # Build negative sampling table (unigram^0.75 distribution)
        powered = self._freqs ** 0.75
        powered /= powered.sum()
        table_size = min(int(1e6), max(int(1e5), len(vocab) * 100))
        self._neg_table = np.zeros(table_size, dtype=np.int32)
        idx, cumulative = 0, powered[0]
        for i in range(table_size):
            self._neg_table[i] = idx
            if i / table_size > cumulative and idx < len(vocab) - 1:
                idx += 1
                cumulative += powered[idx]

    def _subsample_prob(self, word_id: int) -> float:
        """Probability of *keeping* a frequent word."""
        freq = self._freqs[word_id] / self._total
        if freq == 0:
            return 1.0
        return min(1.0, (np.sqrt(freq / self.subsample_t) + 1) * (self.subsample_t / freq))

    def fit(self, texts: list[str]) -> "SkipGramEmbeddings":
        """Train skip-gram embeddings from raw texts."""
        tokenized = [_tokenize_simple(t) for t in texts]
        self._build_vocab(tokenized)

        V = len(self.word2id)
        if V == 0:
            self._fitted = True
            self.W = np.zeros((1, self.dim))
            self.C = np.zeros((1, self.dim))
            return self

        rng = np.random.RandomState(42)
        self.W = (rng.randn(V, self.dim) * 0.01).astype(np.float32)
        self.C = (rng.randn(V, self.dim) * 0.01).astype(np.float32)

        # Convert corpus to id sequences with subsampling
        id_seqs: list[list[int]] = []
        for tokens in tokenized:
            seq = []
            for t in tokens:
                wid = self.word2id.get(t)
                if wid is not None:
                    if rng.rand() < self._subsample_prob(wid):
                        seq.append(wid)
            if seq:
                id_seqs.append(seq)

        # Training loop
        lr = self.lr
        table_len = len(self._neg_table)

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_pairs = 0

            for seq in id_seqs:
                for i, target in enumerate(seq):
                    # Dynamic window
                    win = rng.randint(1, self.window + 1)
                    start = max(0, i - win)
                    end = min(len(seq), i + win + 1)

                    for j in range(start, end):
                        if j == i:
                            continue
                        context = seq[j]

                        # Positive sample
                        score = np.dot(self.W[target], self.C[context])
                        sig = 1.0 / (1.0 + np.exp(-np.clip(score, -6, 6)))
                        grad = (1.0 - sig) * lr
                        total_loss += -np.log(sig + 1e-10)

                        g_w = grad * self.C[context]
                        g_c = grad * self.W[target]
                        self.W[target] += g_w
                        self.C[context] += g_c

                        # Negative samples
                        neg_ids = self._neg_table[rng.randint(0, table_len, size=self.neg_samples)]
                        for neg in neg_ids:
                            if neg == context:
                                continue
                            score_neg = np.dot(self.W[target], self.C[neg])
                            sig_neg = 1.0 / (1.0 + np.exp(-np.clip(score_neg, -6, 6)))
                            grad_neg = -sig_neg * lr
                            total_loss += -np.log(1 - sig_neg + 1e-10)

                            self.W[target] += grad_neg * self.C[neg]
                            self.C[neg] += grad_neg * self.W[target]

                        n_pairs += 1

            # Decay learning rate
            lr = self.lr * (1.0 - (epoch + 1) / self.epochs)
            lr = max(lr, self.lr * 0.01)

        self._fitted = True
        return self

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get embedding for a single word."""
        wid = self.word2id.get(word.lower())
        if wid is not None:
            return self.W[wid].copy()
        return np.zeros(self.dim, dtype=np.float32)

    def get_sentence_embedding(
        self,
        text: str,
        tfidf_weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Compute sentence embedding via TF-IDF weighted averaging.

        Inspired by InterrogateLLM's use of embeddings for
        consistency-based similarity measurements.
        """
        tokens = _tokenize_simple(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        vecs = []
        weights = []
        for t in tokens:
            wid = self.word2id.get(t)
            if wid is not None:
                vecs.append(self.W[wid])
                w = tfidf_weights.get(t, 1.0) if tfidf_weights else 1.0
                weights.append(w)

        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)

        vecs = np.array(vecs, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum() + 1e-10
        return (vecs * weights[:, None]).sum(axis=0)

    def get_segmented_embeddings(
        self,
        text: str,
        n_segments: int = 4,
        tfidf_weights: dict[str, float] | None = None,
    ) -> list[np.ndarray]:
        """
        Split text into n_segments and embed each.

        Used for InterrogateLLM-inspired intra-text consistency analysis.
        """
        tokens = _tokenize_simple(text)
        if len(tokens) < n_segments:
            seg_texts = [" ".join(tokens)] if tokens else [""]
        else:
            chunk = len(tokens) // n_segments
            seg_texts = []
            for i in range(n_segments):
                start = i * chunk
                end = start + chunk if i < n_segments - 1 else len(tokens)
                seg_texts.append(" ".join(tokens[start:end]))

        return [self.get_sentence_embedding(s, tfidf_weights) for s in seg_texts]

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def intra_text_consistency(
        self,
        text: str,
        n_segments: int = 4,
        tfidf_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        InterrogateLLM-inspired consistency scoring.

        Measures how consistent different parts of the text are with each other.
        Hallucinated text tends to have lower internal consistency.

        Returns dict with mean, std, min of pairwise similarities.
        """
        embeddings = self.get_segmented_embeddings(text, n_segments, tfidf_weights)
        if len(embeddings) < 2:
            return {"consistency_mean": 0.0, "consistency_std": 0.0, "consistency_min": 0.0}

        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(self.cosine_similarity(embeddings[i], embeddings[j]))

        sims = np.array(sims)
        return {
            "consistency_mean": float(sims.mean()),
            "consistency_std": float(sims.std()),
            "consistency_min": float(sims.min()),
        }

