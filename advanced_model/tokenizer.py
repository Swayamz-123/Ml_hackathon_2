"""
Custom BPE (Byte-Pair Encoding) Tokenizer -- trained from scratch.

No pretrained tokenizers are used. The entire vocabulary is learned
from the competition corpus (train + test texts).

This module provides subword tokenization for:
  - The n-gram language model (Component 4)
  - The skip-gram embeddings trainer (Component 2)

Reference:
  Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", ACL 2016.
"""

import re
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?|[^\s\w]")


def _pre_tokenize(text: str) -> list[str]:
    """Split raw text into coarse word-level tokens (lowercase)."""
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------
class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained entirely from scratch on the corpus.

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size (including special tokens).
    min_freq : int
        Minimum pair frequency to consider a merge.
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, vocab_size: int = 8000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.merges: list[tuple[str, str]] = []
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._fitted = False

    # ---- training --------------------------------------------------------

    @staticmethod
    def _word_to_symbols(word: str) -> tuple[str, ...]:
        """Convert a word into a tuple of characters + end-of-word marker."""
        return tuple(list(word) + ["</w>"])

    @staticmethod
    def _get_pair_counts(
        word_freqs: dict[tuple[str, ...], int],
    ) -> Counter:
        pairs: Counter = Counter()
        for symbols, freq in word_freqs.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_pair(
        pair: tuple[str, str],
        word_freqs: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, ...], int]:
        new_word_freqs: dict[tuple[str, ...], int] = {}
        bigram = pair
        for symbols, freq in word_freqs.items():
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == bigram[0]
                    and symbols[i + 1] == bigram[1]
                ):
                    new_symbols.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word_freqs[tuple(new_symbols)] = freq
        return new_word_freqs

    def fit(self, texts: list[str]) -> "BPETokenizer":
        """Learn BPE merges from a list of raw texts."""
        # 1. Build word frequency table
        word_counter: Counter = Counter()
        for text in texts:
            for word in _pre_tokenize(text):
                word_counter[word] += 1

        # 2. Convert words to character tuples
        word_freqs: dict[tuple[str, ...], int] = {
            self._word_to_symbols(w): c for w, c in word_counter.items()
        }

        # 3. Collect base vocabulary (all single characters + </w>)
        vocab: set[str] = set()
        for symbols in word_freqs:
            for s in symbols:
                vocab.add(s)

        target = self.vocab_size - len(self.SPECIAL_TOKENS)
        self.merges = []

        while len(vocab) < target:
            pair_counts = self._get_pair_counts(word_freqs)
            if not pair_counts:
                break
            best_pair = pair_counts.most_common(1)[0]
            if best_pair[1] < self.min_freq:
                break
            pair = best_pair[0]
            word_freqs = self._merge_pair(pair, word_freqs)
            merged_token = pair[0] + pair[1]
            vocab.add(merged_token)
            self.merges.append(pair)

        # 4. Build token ↔ id mappings
        self.token2id = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}
        for i, tok in enumerate(sorted(vocab), start=len(self.SPECIAL_TOKENS)):
            self.token2id[tok] = i
        self.id2token = {v: k for k, v in self.token2id.items()}
        self._fitted = True
        return self

    # ---- encoding --------------------------------------------------------

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        for left, right in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == left and symbols[i + 1] == right:
                    symbols = symbols[:i] + [left + right] + symbols[i + 2 :]
                else:
                    i += 1
        return symbols

    def encode_word(self, word: str) -> list[str]:
        """Encode a single word into subword tokens."""
        symbols = list(word) + ["</w>"]
        return self._apply_merges(symbols)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """
        Encode raw text -> list of token ids.

        Parameters
        ----------
        text : str
            Raw input text.
        add_special : bool
            If True, prepend <BOS> and append <EOS>.
        """
        if not self._fitted:
            raise RuntimeError("Tokenizer has not been fitted. Call fit() first.")

        unk_id = self.token2id["<UNK>"]
        ids: list[int] = []
        if add_special:
            ids.append(self.token2id["<BOS>"])

        for word in _pre_tokenize(text):
            for tok in self.encode_word(word):
                ids.append(self.token2id.get(tok, unk_id))

        if add_special:
            ids.append(self.token2id["<EOS>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode list of token ids back to text (approximate)."""
        tokens = [self.id2token.get(i, "<UNK>") for i in ids]
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    @property
    def vocab_len(self) -> int:
        return len(self.token2id)

