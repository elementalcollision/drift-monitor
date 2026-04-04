"""Semantic Drift — measures conceptual center-of-gravity movement.

Detects when the agent's conceptual focus shifts after compression,
even if surface vocabulary is preserved.

Uses sentence-transformers embeddings when available, with a keyword
overlap fallback for zero-dependency operation.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from drift_monitor.instruments.base import Instrument, InstrumentReading, Severity
from drift_monitor.instruments.ghost_lexicon import tokenize, _STOP_WORDS
from drift_monitor.window import DualWindow

# Optional embedding support
try:
    from sentence_transformers import SentenceTransformer

    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False


def _keyword_overlap(texts_a: list[str], texts_b: list[str]) -> float:
    """Compute semantic drift via keyword distribution overlap.

    Returns 0.0 (identical) to 1.0 (completely different).
    This is the zero-dependency fallback.
    """
    if not texts_a or not texts_b:
        return 0.0

    def _build_distribution(texts: list[str]) -> dict[str, float]:
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = [t for t in tokenize(text) if t not in _STOP_WORDS and len(t) > 2]
            counter.update(tokens)
        total = sum(counter.values()) or 1
        return {w: c / total for w, c in counter.items()}

    dist_a = _build_distribution(texts_a)
    dist_b = _build_distribution(texts_b)

    all_keys = set(dist_a) | set(dist_b)
    if not all_keys:
        return 0.0

    # Cosine similarity on keyword distributions
    dot = sum(dist_a.get(k, 0.0) * dist_b.get(k, 0.0) for k in all_keys)
    mag_a = math.sqrt(sum(v**2 for v in dist_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in dist_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 1.0

    similarity = dot / (mag_a * mag_b)
    return min(1.0, max(0.0, 1.0 - similarity))


def _embedding_drift(texts_a: list[str], texts_b: list[str], model_name: str) -> float:
    """Compute drift via embedding centroid distance.

    Requires sentence-transformers. Returns 0.0-1.0.
    """
    model = SentenceTransformer(model_name)

    emb_a = model.encode(texts_a)
    emb_b = model.encode(texts_b)

    # Compute centroids
    centroid_a = [sum(col) / len(col) for col in zip(*emb_a)]
    centroid_b = [sum(col) / len(col) for col in zip(*emb_b)]

    # Cosine similarity between centroids
    dot = sum(a * b for a, b in zip(centroid_a, centroid_b))
    mag_a = math.sqrt(sum(a**2 for a in centroid_a))
    mag_b = math.sqrt(sum(b**2 for b in centroid_b))

    if mag_a == 0 or mag_b == 0:
        return 1.0

    similarity = dot / (mag_a * mag_b)
    return min(1.0, max(0.0, 1.0 - similarity))


class SemanticDrift(Instrument):
    """Measures conceptual drift across compression boundaries.

    Uses sentence-transformers embeddings if available, otherwise falls back
    to keyword distribution overlap.
    """

    name = "semantic_drift"
    high_threshold = 0.15
    moderate_threshold = 0.05

    def __init__(
        self,
        window_size: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
        use_embeddings: bool | None = None,
    ) -> None:
        self.windows = DualWindow(window_size=window_size)
        self.model_name = model_name
        # Auto-detect embedding support if not explicitly set
        self._use_embeddings = (
            use_embeddings if use_embeddings is not None else _HAS_EMBEDDINGS
        )

    @property
    def method(self) -> str:
        return "embedding" if self._use_embeddings else "keyword"

    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.windows.add(text, metadata)

    def mark_boundary(self) -> None:
        self.windows.mark_boundary()

    def score(self) -> float:
        if not self.windows.boundary_marked or not self.windows.has_enough_data(1):
            return 0.0

        anchor_texts = self.windows.anchor.texts
        recent_texts = self.windows.recent.texts

        if self._use_embeddings and _HAS_EMBEDDINGS:
            return _embedding_drift(anchor_texts, recent_texts, self.model_name)

        return _keyword_overlap(anchor_texts, recent_texts)

    def read(self) -> InstrumentReading:
        s = self.score()
        return InstrumentReading(
            instrument=self.name,
            score=s,
            severity=self._classify(s),
            details={"method": self.method},
        )

    def reset(self) -> None:
        self.windows.reset()
