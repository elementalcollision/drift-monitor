"""Base class for drift detection instruments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class InstrumentReading:
    """Result from a single instrument measurement."""

    instrument: str
    score: float  # 0.0 (no drift) to 1.0 (complete drift)
    severity: Severity
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "score": self.score,
            "severity": self.severity.value,
            "details": self.details,
        }


class Instrument(ABC):
    """Abstract base class for drift detection instruments.

    Each instrument compares observations from two windows (anchor and recent)
    to produce a drift score between 0.0 and 1.0.
    """

    # Subclasses set these
    name: str = ""
    high_threshold: float = 0.3
    moderate_threshold: float = 0.1

    @abstractmethod
    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Record an observation in the current window."""

    @abstractmethod
    def score(self) -> float:
        """Compute drift score from current observations. Returns 0.0-1.0."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all observations."""

    def read(self) -> InstrumentReading:
        """Take a full reading with severity classification."""
        s = self.score()
        return InstrumentReading(
            instrument=self.name,
            score=s,
            severity=self._classify(s),
        )

    def _classify(self, score: float) -> Severity:
        if score > self.high_threshold:
            return Severity.HIGH
        if score > self.moderate_threshold:
            return Severity.MODERATE
        return Severity.LOW
