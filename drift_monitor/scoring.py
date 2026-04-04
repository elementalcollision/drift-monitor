"""Composite scoring and compression type classification.

Combines readings from multiple instruments and uses the temporal ordering
of which instruments fire to classify the type of compression event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from drift_monitor.instruments.base import InstrumentReading, Severity


class CompressionType(Enum):
    """Classification of compression events based on instrument firing patterns."""

    NONE = "none"
    VOCABULARY_ONLY = "vocabulary_only"  # Ghost lexicon alone
    OPERATIONAL = "operational"  # Ghost + behavioral, semantic stable
    FULL_BOUNDARY = "full_boundary"  # All three fire
    INFRASTRUCTURE = "infrastructure"  # Behavioral leads, may be model swap
    SEMANTIC_ONLY = "semantic_only"  # Only semantic fires (topic shift)


@dataclass
class DriftReport:
    """Complete drift assessment from all instruments."""

    composite_score: float
    compression_type: CompressionType
    readings: list[InstrumentReading]
    severity: Severity
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 4),
            "compression_type": self.compression_type.value,
            "severity": self.severity.value,
            "readings": [r.to_dict() for r in self.readings],
            "details": self.details,
        }

    @property
    def fired(self) -> list[str]:
        """Names of instruments that exceeded their moderate threshold."""
        return [r.instrument for r in self.readings if r.severity != Severity.LOW]


# Default instrument weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "ghost_lexicon": 0.35,
    "behavioral_footprint": 0.35,
    "semantic_drift": 0.30,
}


class DriftScorer:
    """Combines instrument readings into a composite drift assessment."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def score(self, readings: list[InstrumentReading]) -> DriftReport:
        """Produce a composite drift report from instrument readings."""
        if not readings:
            return DriftReport(
                composite_score=0.0,
                compression_type=CompressionType.NONE,
                readings=[],
                severity=Severity.LOW,
            )

        # Weighted composite
        total_weight = 0.0
        weighted_sum = 0.0
        for reading in readings:
            w = self.weights.get(reading.instrument, 1.0 / len(readings))
            weighted_sum += reading.score * w
            total_weight += w

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        composite = min(1.0, composite)

        # Classify compression type based on firing pattern
        compression_type = self._classify_compression(readings)

        # Overall severity
        if composite > 0.3:
            severity = Severity.HIGH
        elif composite > 0.1:
            severity = Severity.MODERATE
        else:
            severity = Severity.LOW

        return DriftReport(
            composite_score=composite,
            compression_type=compression_type,
            readings=readings,
            severity=severity,
            details={
                "weights": self.weights,
                "fired_instruments": [
                    r.instrument
                    for r in readings
                    if r.severity != Severity.LOW
                ],
            },
        )

    def _classify_compression(
        self,
        readings: list[InstrumentReading],
    ) -> CompressionType:
        """Classify compression type based on which instruments fired."""
        fired = {
            r.instrument
            for r in readings
            if r.severity != Severity.LOW
        }

        if not fired:
            return CompressionType.NONE

        has_ghost = "ghost_lexicon" in fired
        has_behavioral = "behavioral_footprint" in fired
        has_semantic = "semantic_drift" in fired

        if has_ghost and has_behavioral and has_semantic:
            return CompressionType.FULL_BOUNDARY

        if has_behavioral and not has_ghost:
            return CompressionType.INFRASTRUCTURE

        if has_ghost and has_behavioral:
            return CompressionType.OPERATIONAL

        if has_ghost and not has_behavioral and not has_semantic:
            return CompressionType.VOCABULARY_ONLY

        if has_semantic and not has_ghost and not has_behavioral:
            return CompressionType.SEMANTIC_ONLY

        # Fallback for unusual combinations
        return CompressionType.FULL_BOUNDARY
