"""Ghost Lexicon — detects loss of low-frequency, high-precision terminology.

After context compression, agents often lose specialized vocabulary while
retaining common terms. This instrument tracks which precise terms disappear.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from drift_monitor.instruments.base import Instrument, InstrumentReading, Severity
from drift_monitor.window import DualWindow

# Match word tokens, including hyphenated compounds
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*(?:'[a-zA-Z]+)?")

# Common English words to exclude from specialized vocabulary detection
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "not",
        "no",
        "nor",
        "if",
        "then",
        "else",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "up",
        "out",
        "all",
        "also",
        "as",
        "into",
        "only",
        "other",
        "new",
        "some",
        "more",
        "any",
        "each",
        "here",
        "there",
    }
)


def tokenize(text: str) -> list[str]:
    """Extract word tokens from text, lowercased."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def extract_specialized_vocab(
    texts: list[str],
    min_freq: int = 2,
    min_length: int = 4,
) -> set[str]:
    """Identify specialized vocabulary: non-stop-word terms appearing
    min_freq+ times and at least min_length characters long.

    These are the precise, domain-specific terms most vulnerable to compression.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # Remove stop words
    for sw in _STOP_WORDS:
        counter.pop(sw, None)

    if not counter:
        return set()

    # Specialized = appears consistently and is long enough to be domain-specific
    return {
        word
        for word, count in counter.items()
        if count >= min_freq and len(word) >= min_length
    }


class GhostLexicon(Instrument):
    """Measures vocabulary decay across compression boundaries.

    Compares specialized terms present in anchor window against those
    in the recent window. A high score means many precise terms were lost.
    """

    name = "ghost_lexicon"
    high_threshold = 0.3
    moderate_threshold = 0.1

    def __init__(
        self,
        window_size: int = 50,
        min_freq: int = 2,
        min_length: int = 4,
    ) -> None:
        self.windows = DualWindow(window_size=window_size)
        self.min_freq = min_freq
        self.min_length = min_length

    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.windows.add(text, metadata)

    def mark_boundary(self) -> None:
        self.windows.mark_boundary()

    def score(self) -> float:
        """Compute decay score: fraction of anchor specialized vocab missing from recent.

        Returns 0.0 if no boundary marked or insufficient data.
        """
        if not self.windows.boundary_marked or not self.windows.has_enough_data(1):
            return 0.0

        anchor_vocab = extract_specialized_vocab(
            self.windows.anchor.texts,
            min_freq=self.min_freq,
            min_length=self.min_length,
        )

        if not anchor_vocab:
            return 0.0

        recent_vocab = extract_specialized_vocab(
            self.windows.recent.texts,
            min_freq=1,  # Lower threshold for recent — we're checking presence
            min_length=self.min_length,
        )

        # Also check raw token presence in recent texts for terms that
        # may appear once (below specialized threshold)
        recent_all_tokens = set()
        for text in self.windows.recent.texts:
            recent_all_tokens.update(tokenize(text))

        lost = anchor_vocab - recent_all_tokens
        decay = len(lost) / len(anchor_vocab)
        return min(1.0, decay)

    def read(self) -> InstrumentReading:
        s = self.score()
        anchor_vocab = extract_specialized_vocab(
            self.windows.anchor.texts,
            min_freq=self.min_freq,
            min_length=self.min_length,
        )
        recent_tokens = set()
        for text in self.windows.recent.texts:
            recent_tokens.update(tokenize(text))

        lost = anchor_vocab - recent_tokens if anchor_vocab else set()

        return InstrumentReading(
            instrument=self.name,
            score=s,
            severity=self._classify(s),
            details={
                "anchor_specialized_count": len(anchor_vocab),
                "lost_terms": sorted(lost)[:20],  # Cap for readability
                "lost_count": len(lost),
            },
        )

    def reset(self) -> None:
        self.windows.reset()
