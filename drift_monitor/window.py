"""Sliding observation window management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """A single observation with text and optional metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> dict[str, Any]:
        """Serialize this observation to a plain dict (used by persistence)."""
        return {"text": self.text, "metadata": self.metadata}


class ObservationWindow:
    """Fixed-size sliding window of observations.

    Used to maintain anchor (pre-compression) and recent (post-compression)
    windows for comparison by instruments.
    """

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self._observations: deque[Observation] = deque(maxlen=max_size)
        self._texts_cache: list[str] | None = None
        self._observations_cache: list[Observation] | None = None

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._observations.append(Observation(text=text, metadata=metadata or {}))
        self._texts_cache = None
        self._observations_cache = None

    @property
    def texts(self) -> list[str]:
        """Window texts, in order. Read-only — do not mutate the result."""
        if self._texts_cache is None:
            self._texts_cache = [o.text for o in self._observations]
        return self._texts_cache

    @property
    def observations(self) -> list[Observation]:
        """Window observations, in order. Read-only — do not mutate the result."""
        if self._observations_cache is None:
            self._observations_cache = list(self._observations)
        return self._observations_cache

    def __len__(self) -> int:
        return len(self._observations)

    def clear(self) -> None:
        self._observations.clear()
        self._texts_cache = None
        self._observations_cache = None


class DualWindow:
    """Manages anchor and recent windows for before/after comparison.

    Starts by filling the anchor window. Call `mark_boundary()` when a
    compression event is detected to begin filling the recent window.
    """

    def __init__(self, window_size: int = 50) -> None:
        self.anchor = ObservationWindow(max_size=window_size)
        self.recent = ObservationWindow(max_size=window_size)
        self._post_boundary = False

    @property
    def boundary_marked(self) -> bool:
        return self._post_boundary

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        if self._post_boundary:
            self.recent.add(text, metadata)
        else:
            self.anchor.add(text, metadata)

    def mark_boundary(self) -> None:
        """Signal that a compression event has occurred."""
        self._post_boundary = True

    def reset(self) -> None:
        """Clear both windows and reset boundary state."""
        self.anchor.clear()
        self.recent.clear()
        self._post_boundary = False

    def has_enough_data(self, min_observations: int = 3) -> bool:
        """Check if both windows have enough data for meaningful comparison."""
        return (
            len(self.anchor) >= min_observations
            and len(self.recent) >= min_observations
        )

    # ------------------------------------------------------------------
    # State serialization (see drift_monitor.persistence)
    # ------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """Serialize both windows and the boundary flag to a plain dict."""
        return {
            "window_size": self.anchor.max_size,
            "anchor": [o.to_state() for o in self.anchor.observations],
            "recent": [o.to_state() for o in self.recent.observations],
            "boundary_marked": self.boundary_marked,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "DualWindow":
        """Reconstruct a ``DualWindow`` from a ``to_state()`` dict.

        Anchor observations are added first, then recent; the boundary flag is
        applied last so the reconstructed window routes future ``add`` calls
        exactly as the original did.
        """
        dw = cls(window_size=state["window_size"])
        for obs in state.get("anchor", []):
            dw.anchor.add(obs["text"], obs.get("metadata", {}))
        for obs in state.get("recent", []):
            dw.recent.add(obs["text"], obs.get("metadata", {}))
        if state.get("boundary_marked"):
            dw.mark_boundary()
        return dw
