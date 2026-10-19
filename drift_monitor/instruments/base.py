"""Base class for drift detection instruments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from drift_monitor.window import DualWindow


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

    # ------------------------------------------------------------------
    # State serialization (see drift_monitor.persistence for the contract)
    # ------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """Serialize this instrument's full state to a plain dict.

        Captures the instrument name, its constructor config, and both
        observation windows (including the boundary flag) so ``from_state``
        can reconstruct an equivalent instance. See
        :mod:`drift_monitor.persistence` for the schema and versioning rules.
        """
        return {
            "name": self.name,
            "config": self._config_to_state(),
            "windows": self.windows.to_state(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Instrument:
        """Reconstruct an instrument instance from a ``to_state()`` dict.

        Construction is a two-step: ``_construct_from_state`` rebuilds the
        instrument with its original config, then the observation windows are
        replaced by ``DualWindow.from_state`` so the boundary flag and window
        contents are restored exactly.
        """
        instr = cls._construct_from_state(state)
        instr.windows = DualWindow.from_state(state["windows"])
        return instr

    def _config_to_state(self) -> dict[str, Any]:
        """Instrument-specific constructor config, as a plain dict.

        Overridden by instruments that carry config beyond ``window_size``.
        """
        return {}

    @classmethod
    def _construct_from_state(cls, state: dict[str, Any]) -> Instrument:
        """Rebuild an empty instrument of this class from ``state``.

        Default: the only constructor argument is ``window_size`` (read from the
        serialized windows). Instruments with additional config override this to
        pass their own fields through.
        """
        return cls(window_size=state["windows"]["window_size"])

    def _classify(self, score: float) -> Severity:
        if score > self.high_threshold:
            return Severity.HIGH
        if score > self.moderate_threshold:
            return Severity.MODERATE
        return Severity.LOW
