"""Behavioral Footprint — tracks shifts in tool-call patterns and response shape.

Detects when an agent changes its operational behavior after compression:
different tool preferences, altered response lengths, shifted latency patterns.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from drift_monitor.instruments.base import Instrument, InstrumentReading, Severity
from drift_monitor.window import DualWindow


@dataclass
class BehaviorFingerprint:
    """A statistical summary of agent behavior over a window."""

    tool_distribution: dict[str, float] = field(default_factory=dict)
    avg_response_length: float = 0.0
    response_length_std: float = 0.0
    total_observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_distribution": self.tool_distribution,
            "avg_response_length": round(self.avg_response_length, 1),
            "response_length_std": round(self.response_length_std, 1),
            "total_observations": self.total_observations,
        }


def _compute_fingerprint(
    texts: list[str],
    metadata_list: list[dict[str, Any]],
) -> BehaviorFingerprint:
    """Build a fingerprint from a window of observations."""
    if not texts:
        return BehaviorFingerprint()

    # Response lengths
    lengths = [len(t) for t in texts]
    avg_len = sum(lengths) / len(lengths)
    variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
    std_len = math.sqrt(variance)

    # Tool distribution from metadata
    tool_counts: Counter[str] = Counter()
    for meta in metadata_list:
        tools = meta.get("tools", [])
        if isinstance(tools, list):
            tool_counts.update(tools)
        elif isinstance(tools, str):
            tool_counts[tools] += 1

    total_tools = sum(tool_counts.values()) or 1
    tool_dist = {tool: count / total_tools for tool, count in tool_counts.items()}

    return BehaviorFingerprint(
        tool_distribution=tool_dist,
        avg_response_length=avg_len,
        response_length_std=std_len,
        total_observations=len(texts),
    )


def _distribution_distance(d1: dict[str, float], d2: dict[str, float]) -> float:
    """Compute Jensen-Shannon-like distance between two distributions.

    Returns 0.0-1.0. Uses a simplified symmetric KL approach without log
    to avoid zero-division, normalized to [0,1].
    """
    all_keys = set(d1) | set(d2)
    if not all_keys:
        return 0.0

    # Total variation distance (simpler, bounded [0,1])
    total_diff = sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in all_keys)
    return min(1.0, total_diff / 2.0)


def _length_shift(fp1: BehaviorFingerprint, fp2: BehaviorFingerprint) -> float:
    """Measure how much response length distribution shifted. Returns 0.0-1.0."""
    if fp1.avg_response_length == 0 and fp2.avg_response_length == 0:
        return 0.0

    # Normalized difference in means
    denominator = max(fp1.avg_response_length, fp2.avg_response_length, 1.0)
    mean_shift = abs(fp1.avg_response_length - fp2.avg_response_length) / denominator

    return min(1.0, mean_shift)


class BehavioralFootprint(Instrument):
    """Measures behavioral drift via tool-call patterns and response shape.

    Pass metadata with a "tools" key (list of tool names used) for tool tracking.
    """

    name = "behavioral_footprint"
    high_threshold = 0.3
    moderate_threshold = 0.1

    def __init__(
        self,
        window_size: int = 50,
        tool_weight: float = 0.6,
        length_weight: float = 0.4,
    ) -> None:
        self.windows = DualWindow(window_size=window_size)
        self.tool_weight = tool_weight
        self.length_weight = length_weight

    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.windows.add(text, metadata)

    def mark_boundary(self) -> None:
        self.windows.mark_boundary()

    def score(self) -> float:
        if not self.windows.boundary_marked or not self.windows.has_enough_data(1):
            return 0.0

        anchor_fp = _compute_fingerprint(
            self.windows.anchor.texts,
            [o.metadata for o in self.windows.anchor.observations],
        )
        recent_fp = _compute_fingerprint(
            self.windows.recent.texts,
            [o.metadata for o in self.windows.recent.observations],
        )

        tool_dist = _distribution_distance(
            anchor_fp.tool_distribution,
            recent_fp.tool_distribution,
        )
        length_dist = _length_shift(anchor_fp, recent_fp)

        # If no tools were observed, weight entirely on length
        if not anchor_fp.tool_distribution and not recent_fp.tool_distribution:
            return min(1.0, length_dist)

        return min(
            1.0,
            self.tool_weight * tool_dist + self.length_weight * length_dist,
        )

    def read(self) -> InstrumentReading:
        s = self.score()

        anchor_fp = _compute_fingerprint(
            self.windows.anchor.texts,
            [o.metadata for o in self.windows.anchor.observations],
        )
        recent_fp = _compute_fingerprint(
            self.windows.recent.texts,
            [o.metadata for o in self.windows.recent.observations],
        )

        return InstrumentReading(
            instrument=self.name,
            score=s,
            severity=self._classify(s),
            details={
                "anchor_fingerprint": anchor_fp.to_dict(),
                "recent_fingerprint": recent_fp.to_dict(),
                "tool_distance": round(
                    _distribution_distance(
                        anchor_fp.tool_distribution,
                        recent_fp.tool_distribution,
                    ),
                    4,
                ),
                "length_shift": round(_length_shift(anchor_fp, recent_fp), 4),
            },
        )

    def reset(self) -> None:
        self.windows.reset()
