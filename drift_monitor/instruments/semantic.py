"""Semantic Drift — measures conceptual center-of-gravity movement.

Detects when the agent's conceptual focus shifts after compression,
even if surface vocabulary is preserved.

Uses sentence-transformers embeddings when available, with a keyword
overlap fallback for zero-dependency operation.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from functools import lru_cache
from typing import Any

from drift_monitor.instruments.base import Instrument, InstrumentReading
from drift_monitor.instruments.ghost_lexicon import tokenize, _STOP_WORDS, within_window_idf
from drift_monitor.window import DualWindow

# Optional embedding support
try:
    from sentence_transformers import SentenceTransformer

    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False

logger = logging.getLogger(__name__)

# Model cache: loading a SentenceTransformer dominates runtime, so load each
# model once per process instead of once per score() call.
_MODEL_CACHE: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    """Return a cached SentenceTransformer instance for the given model."""
    model = _MODEL_CACHE.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
    return model


@lru_cache(maxsize=256)
def _encode_centroid(texts: tuple[str, ...], model_name: str) -> tuple[float, ...]:
    """Embed a window and return its centroid (cached per window content)."""
    model = _get_model(model_name)
    emb = model.encode(list(texts))
    return tuple(sum(col) / len(col) for col in zip(*emb))


def _keyword_overlap(
    texts_a: list[str],
    texts_b: list[str],
    idf_weighting: bool = False,
) -> float:
    """Compute semantic drift via keyword distribution overlap.

    Returns 0.0 (identical) to 1.0 (completely different).
    This is the zero-dependency fallback.

    With ``idf_weighting``, the distribution becomes tf-idf: each term's
    frequency is multiplied by its within-window inverse document frequency, so
    rare (high-information) terms dominate the similarity instead of common
    non-stop words.
    """
    if not texts_a or not texts_b:
        return 0.0

    idf = (
        within_window_idf(list(texts_a) + list(texts_b))
        if idf_weighting
        else None
    )

    def _build_distribution(texts: list[str]) -> dict[str, float]:
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = [t for t in tokenize(text) if t not in _STOP_WORDS and len(t) > 2]
            counter.update(tokens)
        total = sum(counter.values()) or 1
        if idf is None:
            return {w: c / total for w, c in counter.items()}
        return {w: (c / total) * idf.get(w, 1.0) for w, c in counter.items()}

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
    centroid_a = _encode_centroid(tuple(texts_a), model_name)
    centroid_b = _encode_centroid(tuple(texts_b), model_name)

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
        idf_weighting: bool = False,
    ) -> None:
        self.windows = DualWindow(window_size=window_size)
        self.model_name = model_name
        self.idf_weighting = idf_weighting
        # Auto-detect embedding support if not explicitly set
        self._use_embeddings = (
            use_embeddings if use_embeddings is not None else _HAS_EMBEDDINGS
        )
        if self._use_embeddings and not _HAS_EMBEDDINGS:
            logger.warning(
                "SemanticDrift: use_embeddings=True but sentence-transformers "
                "is not installed; falling back to keyword overlap. Install "
                'with: pip install "drift-monitor[embeddings]"'
            )
            self._use_embeddings = False

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

        return _keyword_overlap(
            anchor_texts, recent_texts, idf_weighting=self.idf_weighting
        )

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

    def _config_to_state(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "use_embeddings": self._use_embeddings,
            "idf_weighting": self.idf_weighting,
        }

    @classmethod
    def _construct_from_state(cls, state: dict[str, Any]) -> Instrument:
        cfg = state["config"]
        return cls(
            window_size=state["windows"]["window_size"],
            model_name=cfg["model_name"],
            use_embeddings=cfg["use_embeddings"],
            # default False keeps pre-idf state loadable
            idf_weighting=cfg.get("idf_weighting", False),
        )
