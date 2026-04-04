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


class ObservationWindow:
    """Fixed-size sliding window of observations.

    Used to maintain anchor (pre-compression) and recent (post-compression)
    windows for comparison by instruments.
    """

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self._observations: deque[Observation] = deque(maxlen=max_size)

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._observations.append(Observation(text=text, metadata=metadata or {}))

    @property
    def texts(self) -> list[str]:
        return [o.text for o in self._observations]

    @property
    def observations(self) -> list[Observation]:
        return list(self._observations)

    def __len__(self) -> int:
        return len(self._observations)

    def clear(self) -> None:
        self._observations.clear()


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
