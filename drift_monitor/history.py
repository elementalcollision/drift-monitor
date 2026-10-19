"""Multi-segment drift history.

``DualWindow`` compares exactly two windows (anchor vs recent) around a *single*
compression boundary. ``SegmentHistory`` layers on top of it to track an
arbitrary number of boundaries: each ``mark_boundary()`` closes the current
segment and opens a new one, and per-boundary drift is computed by comparing
consecutive *closed* segments with the standard instrument trio — exposing the
drift *trend* over a session rather than a single before/after snapshot.

This is deliberately additive: it does not touch ``DualWindow``, the instruments,
or their existing tests. It reuses the same instrument trio + keyword-mode
semantics as :mod:`drift_monitor.significance` (deterministic, zero-dependency),
so a trend is comparable against a single-boundary reading.

Semantics note (the one-boundary lag): the drift at boundary *k* compares closed
segment *k-1* (anchor) against closed segment *k* (recent). Because the segment
*after* a boundary needs observations before it can be compared, boundary *k* is
only reportable once segment *k+1* has been closed — i.e. one boundary later.
``boundary_reports()`` returns ``max(0, len(closed) - 1)`` reports accordingly.
"""

from __future__ import annotations

from typing import Any

from drift_monitor.instruments.base import Instrument
from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.persistence import STATE_SCHEMA_VERSION, migrate_state
from drift_monitor.scoring import DriftReport, DriftScorer
from drift_monitor.window import Observation


class SegmentHistory:
    """Track observations across multiple compression boundaries and their drift trend."""

    def __init__(
        self,
        window_size: int = 50,
        weights: dict[str, float] | None = None,
    ) -> None:
        # weights=None delegates to DriftScorer's defaults (the standard trio weights)
        self.window_size = window_size
        self.weights = dict(weights) if weights is not None else None
        self._closed: list[list[Observation]] = []
        self._current: list[Observation] = []

    # ------------------------------------------------------------------
    # Feeding
    # ------------------------------------------------------------------

    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Record an observation in the current (open) segment."""
        self._current.append(Observation(text=text, metadata=metadata or {}))

    def mark_boundary(self) -> None:
        """Close the current segment (a compression boundary) and open a new one."""
        self._closed.append(self._current)
        self._current = []

    def reset(self) -> None:
        """Drop all segments and start over."""
        self._closed = []
        self._current = []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def segment_count(self) -> int:
        """Number of segments: closed segments plus the current open one."""
        return len(self._closed) + 1

    @property
    def boundary_count(self) -> int:
        """Number of compression boundaries crossed (``mark_boundary`` calls)."""
        return len(self._closed)

    # ------------------------------------------------------------------
    # Drift trend
    # ------------------------------------------------------------------

    def _instruments(self) -> list[Instrument]:
        return [
            GhostLexicon(window_size=self.window_size),
            BehavioralFootprint(window_size=self.window_size),
            SemanticDrift(window_size=self.window_size, use_embeddings=False),
        ]

    def _score_pair(
        self, anchor: list[Observation], recent: list[Observation]
    ) -> DriftReport:
        instruments = self._instruments()
        for instr in instruments:
            for obs in anchor:
                instr.observe(obs.text, obs.metadata)
            instr.mark_boundary()
            for obs in recent:
                instr.observe(obs.text, obs.metadata)
        return DriftScorer(weights=self.weights).score(
            [i.read() for i in instruments]
        )

    def boundary_reports(self) -> list[DriftReport]:
        """One :class:`DriftReport` per completed boundary, in chronological order.

        Boundary ``i`` compares closed segment ``i-1`` (anchor) against segment
        ``i`` (recent). Segments are compared over their last ``window_size``
        observations (the comparison horizon); the full segments remain in
        history for audit.
        """
        return [
            self._score_pair(self._closed[i - 1], self._closed[i])
            for i in range(1, len(self._closed))
        ]

    def trend(self) -> list[float]:
        """Composite drift score per boundary, in chronological order.

        This is the headline number for "is drift accumulating or recovering
        across this session's compression boundaries?".
        """
        return [r.composite_score for r in self.boundary_reports()]

    # ------------------------------------------------------------------
    # Persistence (see drift_monitor.persistence for the versioning contract)
    # ------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """Serialize segments + config; reports are recomputed on load (deterministic)."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "window_size": self.window_size,
            "weights": self.weights,
            "closed": [[o.to_state() for o in seg] for seg in self._closed],
            "current": [o.to_state() for o in self._current],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SegmentHistory":
        """Reconstruct from a ``to_state()`` dict (validates/migrates the schema)."""
        state = migrate_state(state)
        sh = cls(
            window_size=state["window_size"],
            weights=state.get("weights"),
        )
        sh._closed = [
            [Observation(text=o["text"], metadata=o.get("metadata", {})) for o in seg]
            for seg in state["closed"]
        ]
        sh._current = [
            Observation(text=o["text"], metadata=o.get("metadata", {}))
            for o in state["current"]
        ]
        return sh