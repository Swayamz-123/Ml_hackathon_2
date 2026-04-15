"""
Feature Engineering for Hallucination Detection — v2 (Anti-Overfitting).

Design for 95 training samples: every feature is chosen for a clear
semantic reason AND demonstrated discriminative power.  The total count
is kept to ~30-35 features (≤0.4 features per sample) to prevent the
classifiers from memorizing the training set.

Feature groups (per text → delta only, not raw A/B separately):
  1. Structural      (6 per text → 6 deltas + 6 means)
  2. Readability     (1 per text → 1 delta + 1 mean)
  3. Hallucination   (4 per text → 4 deltas + 4 means)
  4. Cross-pair      (5 features: Jaccard 1/2-gram, TF-IDF cosine,
                       entity overlap, numeric overlap)
  5. Citation-aware  (1 per text → 1 delta)

Total: ~28 features

Why this works:
  - Tree models (LightGBM) use delta sign as a split, so raw A/B
    values are redundant with deltas.
  - Mean = (A+B)/2 captures absolute magnitude without doubling dim.
  - Cross-pair features are symmetric and naturally 1-D.
"""

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?")
WORD_RE  = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
NUM_RE   = re.compile(r"[0-9]+(?:\.[0-9]+)?")
SENT_RE  = re.compile(r"(?<=[.!?])\s+")
SYLLABLE_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
CITATION_RE = re.compile(r"\[\d+\]")
DOI_RE      = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# ---------------------------------------------------------------------------
# Word lists (compact, targeted)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","if","in","into",
    "is","it","its","no","not","of","on","or","such","that","the","their",
    "then","there","these","they","this","to","was","will","with","we","our",
    "from","than","has","have","had","been","being","were","which","who",
    "whom","what","when","where","how","all","each","do","does","did","can",
    "could","would","should","may","might","shall","must","also","very",
    "just","about","above","after","again","any","because","before",
    "between","both","during","few","further","here","more","most","other",
    "over","own","same","so","some","through","under","until","up","while",
    "you","your",
}

HEDGE_WORDS = {
    "approximately","roughly","about","estimated","around","nearly",
    "possibly","perhaps","probably","likely","seemingly","apparently",
    "supposedly","reportedly","allegedly","might","may","could",
    "suggest","suggests","suggested","indicate","indicates","indicated",
    "appear","appears","appeared","seem","seems","seemed",
}

VAGUE_WORDS = {
    "some","many","several","various","numerous","few","diverse",
    "multiple","countless","certain","particular","significant",
    "substantial","considerable","extensive","vast","enormous",
    "tremendous","huge","massive","incredible","amazing",
}

INFORMAL_WORDS = {
    "wow","amazing","incredible","awesome","cool","great","fantastic",
    "wonderful","brilliant","super","nice","hope","hopefully","heres",
    "lets","dont","wont","cant","isnt","arent","wasnt","werent",
    "im","ive","id","ill","youre","theyre","weve","theyve",
    "think","thinking","basically","literally","actually","honestly",
    "totally","absolutely","definitely","obviously","clearly",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def _words(text):
    return [m.group().lower() for m in WORD_RE.finditer(text)]


def _tokens(text):
    return [m.group().lower() for m in TOKEN_RE.finditer(text)]


def _sentences(text):
    parts = SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _syllables(word):
    c = len(SYLLABLE_RE.findall(word.lower()))
    if word.endswith("e"):
        c = max(1, c - 1)
    return max(1, c)


# ===========================================================================
# Per-text feature extraction (compact: 12 features per text)
# ===========================================================================
def _per_text_features(text):
    """
    Extract a focused set of per-text features.

    Returns dict with 12 features per text.
    """
    sents = _sentences(text)
    ws = _words(text)
    N = len(ws)
    nc = len(text)

    # --- Structural (6 features) ---
    n_words = float(N)
    n_chars = float(nc)
    n_sents = float(len(sents))

    wl = [len(w) for w in ws]
    avg_wl = float(np.mean(wl)) if wl else 0.0

    sl = [len(_words(s)) for s in sents] if sents else [0]
    longest_sent = float(max(sl))

    # Max repeat: how many times the most frequent word appears
    if ws:
        freq = Counter(ws)
        max_repeat = float(max(freq.values()))
    else:
        max_repeat = 0.0

    # --- Readability (1 feature) ---
    if N > 0 and n_sents > 0:
        sylls = sum(_syllables(w) for w in ws)
        aws = N / max(n_sents, 1)
        asw = sylls / N
        flesch = 206.835 - 1.015 * aws - 84.6 * asw
    else:
        flesch = 0.0

    # --- Hallucination signals (4 features) ---
    hedge_ratio = _safe_div(sum(1 for w in ws if w in HEDGE_WORDS), N)
    vague_ratio = _safe_div(sum(1 for w in ws if w in VAGUE_WORDS), N)
    informal_ratio = _safe_div(sum(1 for w in ws if w in INFORMAL_WORDS), N)

    # Specificity: proper nouns + acronyms + precise numbers
    orig_toks = TOKEN_RE.findall(text)
    proper   = sum(1 for t in orig_toks if t[0].isupper() and len(t) > 1 and t.lower() not in STOPWORDS)
    acronyms = sum(1 for t in orig_toks if t.isupper() and len(t) >= 2 and t.isalpha())
    prec_nums = sum(1 for t in orig_toks if NUM_RE.fullmatch(t) and "." in t)
    specificity = _safe_div(proper + acronyms + prec_nums, max(len(orig_toks), 1))

    # --- Citation count (1 feature) ---
    n_citations = float(len(CITATION_RE.findall(text)) + len(DOI_RE.findall(text)))

    return {
        "n_words":       n_words,
        "n_chars":       n_chars,
        "n_sents":       n_sents,
        "avg_word_len":  avg_wl,
        "longest_sent":  longest_sent,
        "max_repeat":    max_repeat,
        "flesch":        flesch,
        "hedge_ratio":   hedge_ratio,
        "vague_ratio":   vague_ratio,
        "informal_ratio": informal_ratio,
        "specificity":   specificity,
        "n_citations":   n_citations,
    }


# ===========================================================================
# Cross-pair features (5 features)
# ===========================================================================
def _cross_pair(text_a, text_b):
    """Compute pairwise comparison features (symmetric, 1-D each)."""
    ta = _tokens(text_a)
    tb = _tokens(text_b)

    sa, sb = set(ta), set(tb)

    def jaccard(s1, s2):
        u = len(s1 | s2)
        return len(s1 & s2) / u if u else 0.0

    def ngrams(seq, n):
        return set(zip(*[seq[i:] for i in range(n)])) if len(seq) >= n else set()

    j1 = jaccard(sa, sb)
    j2 = jaccard(ngrams(ta, 2), ngrams(tb, 2))

    # Named entity overlap (capitalized non-stopword tokens)
    def entities(text):
        return set(t for t in TOKEN_RE.findall(text)
                   if t[0].isupper() and len(t) > 1 and t.lower() not in STOPWORDS)
    ea, eb = entities(text_a), entities(text_b)
    ent_overlap = jaccard(ea, eb)

    # Numeric overlap
    na = set(NUM_RE.findall(text_a))
    nb = set(NUM_RE.findall(text_b))
    num_overlap = jaccard(na, nb)

    return {
        "jaccard_1gram":  j1,
        "jaccard_2gram":  j2,
        "entity_overlap": ent_overlap,
        "num_overlap":    num_overlap,
    }


# ===========================================================================
# TF-IDF cosine similarity (1 scalar feature per pair)
# ===========================================================================
class TfidfCosineSimilarity:
    """
    Fit a single TF-IDF vectorizer on all texts (train+test), then compute
    cosine similarity between each (A, B) pair as a single scalar feature.

    Unlike TF-IDF SVD (which produces 50-200 dim features that overfit),
    this produces exactly 1 feature per pair — the overall textual similarity.
    """

    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=max_features, min_df=2, sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, all_texts):
        self.vectorizer.fit(all_texts)
        self._fitted = True
        return self

    def similarity(self, text_a_list, text_b_list):
        """Return array of cosine similarities, shape (n_pairs,)."""
        va = self.vectorizer.transform(text_a_list)
        vb = self.vectorizer.transform(text_b_list)
        # Row-wise cosine similarity
        sims = np.array([
            float(sklearn_cosine(va[i], vb[i])[0, 0])
            for i in range(va.shape[0])
        ])
        return sims


# ===========================================================================
# Master builder
# ===========================================================================
def extract_pair_features(text_a, text_b):
    """
    Full feature vector for a single pair.

    Returns 1-D numpy array with:
      - delta  (A - B) for each per-text feature  [12 features]
      - mean   (A+B)/2 for each per-text feature   [12 features]
      - cross-pair features                         [4 features]
    Total: 28 features (+ 1 TF-IDF cosine added externally)
    """
    fa = _per_text_features(text_a)
    fb = _per_text_features(text_b)
    keys = sorted(fa.keys())

    va = np.array([fa[k] for k in keys], dtype=np.float64)
    vb = np.array([fb[k] for k in keys], dtype=np.float64)

    cp = _cross_pair(text_a, text_b)
    cp_keys = sorted(cp.keys())
    vc = np.array([cp[k] for k in cp_keys], dtype=np.float64)

    delta = va - vb
    mean = (va + vb) / 2.0

    return np.concatenate([delta, mean, vc])


def build_pair_feature_matrix(text_a_list, text_b_list):
    """Build feature matrix for all pairs. Returns (n_pairs, n_features) array."""
    rows = [extract_pair_features(a, b) for a, b in zip(text_a_list, text_b_list)]
    return np.vstack(rows)


def get_feature_names():
    """Get ordered feature names matching the output of extract_pair_features."""
    fa = _per_text_features("sample text here for testing purposes")
    keys = sorted(fa.keys())
    cp = _cross_pair("sample text", "other text")
    cp_keys = sorted(cp.keys())
    names = (
        [f"delta_{k}" for k in keys] +
        [f"mean_{k}" for k in keys] +
        list(cp_keys)
    )
    return names
