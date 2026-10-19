"""Hedging Language — detects erosion of assertive/confident language.

After compression (or under pressure), agents often drift toward hedged,
over-cautious phrasing: "must" becomes "could potentially", "requires"
becomes "might benefit from". The surface trio (ghost lexicon, behavioral,
semantic) does not flag this: vocabulary is preserved and the topic is
unchanged. This instrument closes that gap.

Scope note: this is a lexical, surface-level detector for hedging *style*
drift. It does not detect deep framing drift that leaves no lexical trace —
DriftMode.FRAMING's guarantee of invisibility applies to the three surface
instruments only; when its reframing uses hedging qualifiers (as the
synthetic generator's does), this instrument will fire.
"""

from __future__ import annotations

import re
from typing import Any

from drift_monitor.instruments.base import Instrument, InstrumentReading
from drift_monitor.window import DualWindow

# Longest-first so multi-word hedges count once, not as their parts.
#
# Deliberately excludes the bare modals "may"/"might"/"could": they are
# ubiquitous in ordinary technical prose (and "May" collides with the month
# under IGNORECASE), so counting them as hedges over-fires on real text. The
# multi-word qualifiers ("could potentially", "might benefit") still catch the
# confidence-erosion substitutions the framing generator makes.
_HEDGE_RE = re.compile(
    r"\b(?:"
    r"could potentially|might benefit|may benefit|it seems that|it appears that|"
    r"appears to|seems to|tends to|to some extent|worth considering|"
    r"arguably|possibly|perhaps|maybe|potentially|presumably|somewhat|"
    r"generally"
    r")\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[^\W\d_]+(?:[\w'-]*[^\W\d_])?", re.UNICODE)

# A hedge-density increase of this many hedges per word saturates the score.
# 0.05 = "5 more hedges per 100 words" reads as full confidence erosion.
_SATURATION_DENSITY = 0.05


def _hedge_density(texts: list[str]) -> tuple[float, dict[str, int]]:
    """Return (hedges per word, per-phrase counts) over a window's texts."""
    hedge_count = 0
    word_count = 0
    counts: dict[str, int] = {}
    for text in texts:
        words = _WORD_RE.findall(text)
        word_count += len(words)
        for m in _HEDGE_RE.finditer(text):
            hedge_count += 1
            key = m.group(0).lower()
            counts[key] = counts.get(key, 0) + 1
    density = hedge_count / word_count if word_count else 0.0
    return density, counts


class HedgingLanguage(Instrument):
    """Measures increase in hedging language across compression boundaries.

    Score is one-sided: only an *increase* in hedge density from the anchor
    to the recent window counts as drift (growing confidence is not a
    compression symptom). A score of 1.0 means hedge density rose by
    ``_SATURATION_DENSITY`` (5 hedges per 100 words) or more.
    """

    name = "hedging_language"
    high_threshold = 0.5
    moderate_threshold = 0.2

    def __init__(self, window_size: int = 50) -> None:
        self.windows = DualWindow(window_size=window_size)

    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.windows.add(text, metadata)

    def mark_boundary(self) -> None:
        self.windows.mark_boundary()

    def score(self) -> float:
        if not self.windows.boundary_marked or not self.windows.has_enough_data(1):
            return 0.0
        return self._measure()[0]

    def _measure(self) -> tuple[float, dict[str, Any]]:
        if not self.windows.boundary_marked or not self.windows.has_enough_data(1):
            return 0.0, {}

        anchor_density, _ = _hedge_density(self.windows.anchor.texts)
        recent_density, recent_counts = _hedge_density(self.windows.recent.texts)

        increase = max(0.0, recent_density - anchor_density)
        score = min(1.0, increase / _SATURATION_DENSITY)

        top_hedges = sorted(recent_counts.items(), key=lambda kv: -kv[1])[:10]
        details = {
            "anchor_hedge_density": round(anchor_density, 4),
            "recent_hedge_density": round(recent_density, 4),
            "top_hedge_phrases": top_hedges,
        }
        return score, details

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
