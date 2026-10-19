"""Ghost Lexicon — detects loss of low-frequency, high-precision terminology.

After context compression, agents often lose specialized vocabulary while
retaining common terms. This instrument tracks which precise terms disappear.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from drift_monitor.instruments.base import Instrument, InstrumentReading
from drift_monitor.window import DualWindow

# Match word tokens, including hyphenated compounds. Unicode-aware: any
# letter (not digits/underscore) starts a token, so non-English text
# tokenizes correctly instead of silently yielding nothing. For pure-ASCII
# input the token stream is identical to the historical [a-zA-Z] pattern.
_TOKEN_RE = re.compile(r"[^\W\d_][\w-]*(?:'[\w]+)?", re.UNICODE)

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


def within_window_idf(observations: list[str]) -> dict[str, float]:
    """Smooth, within-window inverse document frequency per token.

    Treats each observation as its own document and measures ``idf`` as the term's
    *concentration* across the window's observations — rarity relative to the
    window, not a reference corpus (keeps the library zero-data, per the design
    decision). Uses the ``smooth_idf`` form ``log((N + 1) / (df + 1)) + 1`` so the
    weight is nonzero even for a term present in every observation, and grows as a
    term becomes rarer (present in fewer observations).

    Deterministic and self-contained; shared by the ghost lexicon (rarity-weighted
    decay) and semantic keyword overlap (tf-idf).
    """
    doc_freq: dict[str, int] = {}
    n = 0
    for observation in observations:
        n += 1
        for token in set(tokenize(observation)):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    if n == 0:
        return {}
    return {
        token: math.log((n + 1) / (df + 1)) + 1.0
        for token, df in doc_freq.items()
    }


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
        idf_weighting: bool = False,
    ) -> None:
        self.windows = DualWindow(window_size=window_size)
        self.min_freq = min_freq
        self.min_length = min_length
        self.idf_weighting = idf_weighting

    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.windows.add(text, metadata)

    def mark_boundary(self) -> None:
        self.windows.mark_boundary()

    def score(self) -> float:
        """Compute decay score: fraction of anchor specialized vocab missing from recent.

        Returns 0.0 if no boundary marked or insufficient data.
        """
        # Fast path: no measurement is possible, skip the (now shared)
        # full computation. read() still reports details via _measure().
        if not self.windows.boundary_marked or not self.windows.has_enough_data(1):
            return 0.0
        return self._measure()[0]

    def _measure(self) -> tuple[float, dict[str, Any]]:
        """Single-pass score + details computation.

        score() and read() both need the vocab extraction and tokenisation,
        so they share this method instead of each redoing the work.

        The no-measurement guard runs first, so read() reports empty details
        (not a spurious ``lost_count``) when no boundary has been crossed —
        matching BehaviouralFootprint._measure.
        """
        if not self.windows.boundary_marked or not self.windows.has_enough_data(1):
            return 0.0, {}

        anchor_vocab = extract_specialized_vocab(
            self.windows.anchor.texts,
            min_freq=self.min_freq,
            min_length=self.min_length,
        )
        recent_tokens = set()
        for text in self.windows.recent.texts:
            recent_tokens.update(tokenize(text))

        lost = anchor_vocab - recent_tokens if anchor_vocab else set()

        if not anchor_vocab:
            return 0.0, {
                "anchor_specialized_count": 0,
                "lost_terms": [],
                "lost_count": 0,
            }

        # Decay is measured against RAW token presence in the recent window (a
        # term counts as retained if it appears at all, even below the
        # specialized-vocab threshold), so no specialized recent-vocab set is needed.
        #
        # By default every lost term counts equally. With idf_weighting, each term
        # is weighted by its within-window rarity, so losing a rare (concentrated)
        # jargon term moves the score more than losing a term used in every
        # observation.
        if self.idf_weighting:
            idf = within_window_idf(self.windows.anchor.texts)
            weighted_lost = sum(idf.get(t, 1.0) for t in lost)
            weighted_total = sum(idf.get(t, 1.0) for t in anchor_vocab)
            decay = weighted_lost / weighted_total if weighted_total else 0.0
        else:
            decay = len(lost) / len(anchor_vocab)
        return min(1.0, decay), {
            "anchor_specialized_count": len(anchor_vocab),
            "lost_terms": sorted(lost)[:20],  # Cap for readability
            "lost_count": len(lost),
        }

    def read(self) -> InstrumentReading:
        s, details = self._measure()
        return InstrumentReading(
            instrument=self.name,
            score=s,
            severity=self._classify(s),
            details=details,
        )

    def reset(self) -> None:
        self.windows.reset()

    def _config_to_state(self) -> dict[str, Any]:
        return {
            "min_freq": self.min_freq,
            "min_length": self.min_length,
            "idf_weighting": self.idf_weighting,
        }

    @classmethod
    def _construct_from_state(cls, state: dict[str, Any]) -> Instrument:
        cfg = state["config"]
        return cls(
            window_size=state["windows"]["window_size"],
            min_freq=cfg["min_freq"],
            min_length=cfg["min_length"],
            # default False keeps state written before idf_weighting existed loadable
            idf_weighting=cfg.get("idf_weighting", False),
        )
