"""High-level convenience facade over the three instruments + the scorer."""

from __future__ import annotations

from typing import Any

from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.scoring import DriftReport, DriftScorer


class DriftMonitor:
    """One-object drift monitoring across all three instruments.

    Bundles :class:`GhostLexicon`, :class:`BehavioralFootprint` and
    :class:`SemanticDrift` with a :class:`DriftScorer` so callers can
    ``observe`` -> ``mark_boundary`` -> ``observe`` -> ``score`` against a single
    object instead of wiring each instrument by hand. For finer control
    (individual instruments, per-instrument options, custom instrument sets) use
    those classes together with :class:`DriftScorer` directly.

    Example::

        monitor = DriftMonitor()
        for text in pre_compression_outputs:
            monitor.observe(text, {"tools": ["search", "code"]})
        monitor.mark_boundary()
        for text in post_compression_outputs:
            monitor.observe(text, {"tools": ["search"]})
        report = monitor.score()
        print(report.composite_score, report.compression_type.name)
    """

    def __init__(
        self,
        *,
        window_size: int = 50,
        weights: dict[str, float] | None = None,
        use_embeddings: bool = False,
    ) -> None:
        self.instruments = [
            GhostLexicon(window_size=window_size),
            BehavioralFootprint(window_size=window_size),
            SemanticDrift(window_size=window_size, use_embeddings=use_embeddings),
        ]
        self.scorer = DriftScorer(weights=weights)

    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Feed one observation to every instrument.

        Tool names go under ``metadata["tools"]`` (used by the behavioral
        instrument; ignored by the others).
        """
        for instrument in self.instruments:
            instrument.observe(text, metadata)

    def mark_boundary(self) -> None:
        """Mark the compression boundary on every instrument."""
        for instrument in self.instruments:
            instrument.mark_boundary()

    def score(self) -> DriftReport:
        """Read all instruments and combine them into a composite report."""
        return self.scorer.score([i.read() for i in self.instruments])

    def reset(self) -> None:
        """Clear all observations from every instrument."""
        for instrument in self.instruments:
            instrument.reset()
