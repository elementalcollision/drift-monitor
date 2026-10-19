"""Statistical significance for composite drift scores.

Drift composites are weighted averages over small sliding windows; a 0.12
composite on a handful of observations can be noise. This module provides a
permutation test: pool the anchor and recent observations, repeatedly
re-split them at the same sizes across a sham boundary, recompute the
composite, and ask how often the sham composites reach the observed one.

The resulting p-value is the probability of seeing drift this strong (or
stronger) if the boundary were arbitrary — i.e. if nothing happened at all.

Cost note: each permutation rebuilds and re-scores the instrument set, so
this is an offline/analysis tool, not a hot-path call. Use keyword-mode
semantics (the default here) to keep permutations cheap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.scoring import DriftScorer
from drift_monitor.window import Observation


@dataclass
class SignificanceResult:
    """Outcome of a permutation test on the composite drift score."""

    observed_composite: float
    p_value: float
    n_permutations: int
    null_composites: list[float] = field(default_factory=list)

    @property
    def significant_at_5pct(self) -> bool:
        return self.p_value < 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_composite": round(self.observed_composite, 4),
            "p_value": round(self.p_value, 4),
            "n_permutations": self.n_permutations,
            "null_mean": (
                round(sum(self.null_composites) / len(self.null_composites), 4)
                if self.null_composites
                else None
            ),
            "significant_at_5pct": self.significant_at_5pct,
        }


def _composite_for_split(
    anchor: list[Observation],
    recent: list[Observation],
    weights: dict[str, float] | None,
) -> float:
    """Run the standard instrument trio on an explicit anchor/recent split."""
    window_size = max(len(anchor), len(recent), 1)
    instruments = [
        GhostLexicon(window_size=window_size),
        BehavioralFootprint(window_size=window_size),
        SemanticDrift(window_size=window_size, use_embeddings=False),
    ]
    for instr in instruments:
        for obs in anchor:
            instr.observe(obs.text, obs.metadata)
        instr.mark_boundary()
        for obs in recent:
            instr.observe(obs.text, obs.metadata)
    scorer = DriftScorer(weights=weights)
    return scorer.score([i.read() for i in instruments]).composite_score


def permutation_test(
    anchor: list[Observation],
    recent: list[Observation],
    weights: dict[str, float] | None = None,
    n_permutations: int = 200,
    seed: int = 0,
) -> SignificanceResult:
    """Permutation test for composite drift across a boundary.

    Pools anchor + recent observations, shuffles, re-splits at the original
    sizes, and recomputes the composite ``n_permutations`` times. The p-value
    is the fraction of null composites >= observed, with add-one smoothing
    so p > 0 even when no permutation matches.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")

    observed = _composite_for_split(anchor, recent, weights)

    pool = list(anchor) + list(recent)
    n_anchor = len(anchor)
    rng = random.Random(seed)

    null_composites = []
    for _ in range(n_permutations):
        rng.shuffle(pool)
        null = _composite_for_split(pool[:n_anchor], pool[n_anchor:], weights)
        null_composites.append(null)

    extreme = sum(1 for c in null_composites if c >= observed)
    p_value = (1 + extreme) / (n_permutations + 1)

    return SignificanceResult(
        observed_composite=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        null_composites=[round(c, 6) for c in null_composites],
    )


def monitor_significance(
    monitor: Any,
    n_permutations: int = 200,
    seed: int = 0,
) -> SignificanceResult:
    """Convenience wrapper: run the permutation test on a DriftMonitor's windows.

    Uses the monitor's scorer weights. Embeddings are not used in the null
    recomputations (keyword mode) — see the module docstring.

    The null recomputation rebuilds the standard trio in keyword mode, so this
    wrapper only accepts a monitor whose composite that recomputation can
    faithfully reproduce: the default instrument set (GhostLexicon +
    BehaviouralFootprint + SemanticDrift) with keyword-mode semantics. A custom
    instrument set or embedding-mode semantics would make the reported
    ``observed_composite`` diverge silently from ``monitor.score()``, so those
    raise :class:`ValueError` — use :func:`permutation_test` directly for such
    monitors.
    """
    # All instruments in a DriftMonitor share the same observation stream;
    # read the windows from the first one.
    windows = monitor.instruments[0].windows
    if not windows.boundary_marked:
        raise ValueError("monitor has no marked boundary to test")

    names = {i.name for i in monitor.instruments}
    if names != {"ghost_lexicon", "behavioral_footprint", "semantic_drift"}:
        raise ValueError(
            "monitor_significance reproduces the standard trio composite, but "
            f"the monitor's instruments are {sorted(names)}; the permutation "
            "null would not match the monitor's own composite. Call "
            "permutation_test(anchor, recent, weights=...) directly for custom "
            "instrument sets."
        )
    if any(getattr(i, "_use_embeddings", False) for i in monitor.instruments):
        raise ValueError(
            "the monitor uses embedding-mode SemanticDrift, but the "
            "permutation null recomputes in keyword mode; the p-value would "
            "not reflect the monitor's embedding-based composite. Run on "
            "keyword-mode readings or call permutation_test() directly."
        )

    return permutation_test(
        list(windows.anchor.observations),
        list(windows.recent.observations),
        weights=monitor.scorer.weights,
        n_permutations=n_permutations,
        seed=seed,
    )
