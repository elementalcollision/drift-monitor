"""Autoresearch harness — live drift monitoring for experiment loops.

Wraps the autoresearch-unified OrchestratorCallbacks to intercept experiment
events and feed them into drift instruments in real time. Writes drift readings
to a JSONL file alongside results.tsv.

Design:
- Zero modifications to autoresearch-unified source code required
- Attaches by wrapping existing callbacks at construction time
- Produces actionable nudge messages when drift thresholds are exceeded
- Operates on the experiment's description + notes (the LLM-generated text)
  and the strategy category (which hyperparameter lever was pulled)

Usage:
    from drift_monitor.harness import DriftHarness

    harness = DriftHarness(results_dir=".")
    wrapped_callbacks = harness.wrap(original_callbacks)
    # Pass wrapped_callbacks to ExperimentOrchestrator instead
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.alerts import Alert, AlertHook
from drift_monitor.persistence import (
    STATE_SCHEMA_VERSION,
    EventLog,
    instrument_from_state,
    migrate_state,
)
from drift_monitor.scoring import DriftScorer, DriftReport
from drift_monitor.storage import append_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol for the ExperimentResult we receive (avoid importing autoresearch)
# ---------------------------------------------------------------------------

class ExperimentResultLike(Protocol):
    """Minimal interface matching autoresearch ExperimentResult."""
    exp: str
    description: str
    status: str
    notes: str


# ---------------------------------------------------------------------------
# Strategy classification (mirrors autoresearch's classify_experiment)
# ---------------------------------------------------------------------------

_STRATEGY_PATTERNS: list[tuple[str, list[str]]] = [
    ("batch_size", ["batch_size", "batch size", "total_batch"]),
    ("architecture", ["depth", "head_dim", "window_pattern", "window pattern",
                       "mlp_ratio", "aspect_ratio", "num_heads"]),
    ("schedule", ["warmup", "warmdown", "final_lr_frac", "schedule", "cooldown"]),
    ("regularization", ["weight_decay", "weight decay", "adam_beta", "regularization"]),
    ("learning_rate", ["_lr", "learning rate", "learning_rate", "matrix_lr",
                        "scalar_lr", "embedding_lr", "unembedding_lr"]),
    ("infrastructure", ["activation_checkpointing", "compile_mode", "compile mode"]),
]


def classify_strategy(description: str) -> str:
    """Classify an experiment description into a strategy category."""
    desc = description.lower()
    for category, patterns in _STRATEGY_PATTERNS:
        if any(p in desc for p in patterns):
            return category
    return "other"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DriftConfig:
    """Configuration for the drift harness."""

    # Window sizes
    anchor_window: int = 25       # Number of experiments in the anchor window
    assessment_interval: int = 10  # Run drift assessment every N experiments

    # Thresholds for nudge generation
    ghost_nudge_threshold: float = 0.35
    behavioral_nudge_threshold: float = 0.30
    composite_nudge_threshold: float = 0.30

    # Output
    output_filename: str = "drift_readings.jsonl"
    status_filename: str = ".drift_status.json"

    # Event tail (append-only audit + crash recovery). ``None`` disables it.
    # When set, every observation and boundary is appended so state can be
    # reconstructed past the last snapshot (see DriftHarness.restore).
    event_log_path: str | None = None

    # Scoring weights
    weights: dict[str, float] = field(default_factory=lambda: {
        "ghost_lexicon": 0.35,
        "behavioral_footprint": 0.35,
        "semantic_drift": 0.30,
    })


# ---------------------------------------------------------------------------
# Nudge templates
# ---------------------------------------------------------------------------

_NUDGE_VOCABULARY = (
    "\n\nDRIFT ALERT (vocabulary narrowing): Your recent experiments have dropped "
    "terminology related to {lost_categories}. Consider revisiting: {suggestions}. "
    "Vocabulary diversity correlates with finding novel improvements."
)

_NUDGE_BEHAVIORAL = (
    "\n\nDRIFT ALERT (strategy collapse): {dominant_pct:.0%} of your recent experiments "
    "targeted {dominant_category}. The exploration/exploitation balance has shifted "
    "too far toward exploitation. Try a fundamentally different lever: {alternatives}."
)

_NUDGE_COMPOSITE = (
    "\n\nDRIFT ALERT (full behavioral drift detected, composite={score:.2f}): "
    "Your approach has narrowed significantly since the start of this run. "
    "Lost vocabulary: {lost_terms}. "
    "Dominant strategy: {dominant_category} ({dominant_pct:.0%}). "
    "Recommended: try {suggestion} to break out of the current optimization basin."
)

# Map categories to concrete suggestions
_CATEGORY_SUGGESTIONS: dict[str, str] = {
    "learning_rate": "architectural changes (DEPTH, WINDOW_PATTERN, MLP_RATIO, ASPECT_RATIO)",
    "architecture": "schedule shape changes (WARMUP_RATIO>0, FINAL_LR_FRAC, WARMDOWN_RATIO)",
    "schedule": "regularization tuning (WEIGHT_DECAY, ADAM_BETAS)",
    "regularization": "learning rate exploration across parameter groups",
    "batch_size": "architectural or schedule experiments",
    "other": "systematic learning rate or architecture exploration",
}


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------

class DriftHarness:
    """Live drift monitoring harness for autoresearch experiment loops.

    Intercepts experiment completion events, feeds them to drift instruments,
    and generates actionable nudge messages when drift thresholds are exceeded.
    """

    def __init__(
        self,
        results_dir: str = ".",
        config: DriftConfig | None = None,
    ) -> None:
        self.config = config or DriftConfig()
        self.results_dir = Path(results_dir)

        # Instruments — start in anchor-filling mode
        self._ghost = GhostLexicon(
            window_size=self.config.anchor_window * 2,
            min_freq=2,
            min_length=4,
        )
        self._behavioral = BehavioralFootprint(
            window_size=self.config.anchor_window * 2,
        )
        self._semantic = SemanticDrift(
            window_size=self.config.anchor_window * 2,
            use_embeddings=False,
        )
        self._scorer = DriftScorer(weights=self.config.weights)

        # State
        self._experiment_count = 0
        self._boundary_set = False
        self._last_report: DriftReport | None = None
        self._strategy_history: list[str] = []  # category per experiment

        # Output paths
        self._readings_path = self.results_dir / self.config.output_filename
        self._status_path = self.results_dir / self.config.status_filename

        # Optional append-only event tail (audit + crash recovery)
        self._event_log: EventLog | None = None
        if self.config.event_log_path:
            ep = Path(self.config.event_log_path)
            if not ep.is_absolute():
                ep = self.results_dir / ep
            self._event_log = EventLog(ep)

        # Alert hooks (opt-in, out-of-band notifications)
        self._alert_hooks: list[AlertHook] = []
        self._alert_firing: dict[str, bool] = {}  # per alert type, edge-trigger state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_experiment(self, result: ExperimentResultLike) -> None:
        """Feed a completed experiment to the drift instruments.

        Call this after each experiment completes. The harness handles
        anchor/recent window management automatically.
        """
        if result.status == "baseline":
            return

        self._experiment_count += 1

        # Auto-set boundary after anchor window fills
        if (
            not self._boundary_set
            and self._experiment_count >= self.config.anchor_window
        ):
            self.mark_boundary()
            self._boundary_set = True

        # Build observation text and metadata
        text = f"{result.description}. {result.notes}" if result.notes else result.description
        strategy = classify_strategy(result.description)
        self._strategy_history.append(strategy)

        metadata = {
            "tools": [strategy, f"outcome:{result.status}"],
            "exp": result.exp,
            "status": result.status,
        }

        self.observe(text, metadata)

        # Run assessment at intervals (after boundary is set)
        if (
            self._boundary_set
            and self._experiment_count % self.config.assessment_interval == 0
        ):
            self._run_assessment()

    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Feed one observation to all instruments (primitive API).

        The event-tail primitive; also used internally by ``observe_experiment``.
        When an event log is configured, the observation is appended to it.
        """
        self._ghost.observe(text, metadata)
        self._behavioral.observe(text, metadata)
        self._semantic.observe(text, metadata)
        if self._event_log is not None:
            self._event_log.observe(text, metadata)

    def mark_boundary(self) -> None:
        """Mark the compression boundary on all instruments (primitive API)."""
        self._ghost.mark_boundary()
        self._behavioral.mark_boundary()
        self._semantic.mark_boundary()
        if self._event_log is not None:
            self._event_log.mark_boundary()

    def reset(self) -> None:
        """Reset all instruments and bookkeeping to a clean state."""
        self._ghost.reset()
        self._behavioral.reset()
        self._semantic.reset()
        self._experiment_count = 0
        self._boundary_set = False
        self._strategy_history = []
        self._last_report = None
        self._alert_firing = {}
        if self._event_log is not None:
            self._event_log.reset()

    def add_alert_hook(self, hook: AlertHook) -> None:
        """Register a callback to receive drift :class:`Alert` objects.

        ``hook`` is any ``(Alert) -> Any`` callable (a plain function or a
        :class:`WebhookHook`). Alerts are edge-triggered: a hook fires once when
        a score crosses above its threshold, and only fires again after the
        score drops back below. Hooks run synchronously during assessment; a
        raising hook is isolated so it does not stop the experiment loop.
        """
        self._alert_hooks.append(hook)

    def remove_alert_hook(self, hook: AlertHook) -> None:
        """Remove a previously registered alert hook (no-op if not present)."""
        if hook in self._alert_hooks:
            self._alert_hooks.remove(hook)

    def clear_alert_hooks(self) -> None:
        """Remove every registered alert hook."""
        self._alert_hooks.clear()

    def get_drift_nudge(self) -> str | None:
        """Get a nudge message if drift thresholds are exceeded.

        Returns None if drift is within acceptable bounds.
        Call this from the orchestrator's prompt-building step.
        """
        if self._last_report is None:
            return None

        report = self._last_report
        readings_by_name = {r.instrument: r for r in report.readings}

        ghost_reading = readings_by_name.get("ghost_lexicon")
        behavioral_reading = readings_by_name.get("behavioral_footprint")

        # Composite nudge (highest priority)
        if report.composite_score >= self.config.composite_nudge_threshold:
            dominant = self._dominant_strategy()
            lost = ghost_reading.details.get("lost_terms", [])[:5] if ghost_reading else []
            suggestion = _CATEGORY_SUGGESTIONS.get(
                dominant["category"],
                "a fundamentally different approach",
            )
            return _NUDGE_COMPOSITE.format(
                score=report.composite_score,
                lost_terms=", ".join(lost) if lost else "several specialized terms",
                dominant_category=dominant["category"],
                dominant_pct=dominant["pct"],
                suggestion=suggestion,
            )

        # Vocabulary nudge
        if (
            ghost_reading
            and ghost_reading.score >= self.config.ghost_nudge_threshold
        ):
            lost = ghost_reading.details.get("lost_terms", [])[:5]
            lost_cats = self._infer_lost_categories(lost)
            suggestions = ", ".join(
                _CATEGORY_SUGGESTIONS.get(c, c) for c in lost_cats[:2]
            ) or "different strategy categories"
            return _NUDGE_VOCABULARY.format(
                lost_categories=", ".join(lost_cats) or "several areas",
                suggestions=suggestions,
            )

        # Behavioral nudge
        if (
            behavioral_reading
            and behavioral_reading.score >= self.config.behavioral_nudge_threshold
        ):
            dominant = self._dominant_strategy()
            alternatives = [
                _CATEGORY_SUGGESTIONS[c]
                for c in _CATEGORY_SUGGESTIONS
                if c != dominant["category"]
            ][:2]
            return _NUDGE_BEHAVIORAL.format(
                dominant_pct=dominant["pct"],
                dominant_category=dominant["category"],
                alternatives="; or ".join(alternatives),
            )

        return None

    @property
    def last_report(self) -> DriftReport | None:
        """Most recent drift assessment report."""
        return self._last_report

    @property
    def experiment_count(self) -> int:
        return self._experiment_count

    @property
    def boundary_set(self) -> bool:
        return self._boundary_set

    def wrap_callbacks(self, callbacks: Any) -> Any:
        """Wrap OrchestratorCallbacks to intercept experiment completion.

        Returns a new callbacks object that feeds experiments to the harness
        before delegating to the original callbacks.

        Works with any object that has an `on_experiment_complete` attribute.
        """
        original_on_complete = callbacks.on_experiment_complete

        def intercepted_on_complete(result: Any) -> None:
            # Feed to drift harness first
            self.observe_experiment(result)
            # Then delegate to original
            original_on_complete(result)

        # Create a shallow copy-like wrapper
        # Works with both dataclass instances and plain objects
        import copy
        wrapped = copy.copy(callbacks)
        wrapped.on_experiment_complete = intercepted_on_complete
        return wrapped

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_assessment(self) -> None:
        """Run drift assessment and write results."""
        readings = [
            self._ghost.read(),
            self._behavioral.read(),
            self._semantic.read(),
        ]
        self._last_report = self._scorer.score(readings)

        # Dispatch any newly-crossed drift thresholds to alert hooks.
        self._dispatch_alerts()

        # Write to JSONL
        record = {
            "experiment_count": self._experiment_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **self._last_report.to_dict(),
            "strategy_distribution": self._strategy_distribution(),
        }
        append_jsonl(self._readings_path, record)

        # Write status file (overwrite)
        status = {
            "experiment_count": self._experiment_count,
            "composite_score": round(self._last_report.composite_score, 4),
            "compression_type": self._last_report.compression_type.value,
            "severity": self._last_report.severity.value,
            "nudge_active": self._last_report.composite_score >= self.config.composite_nudge_threshold,
            "strategy_distribution": self._strategy_distribution(),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            from drift_monitor.storage import atomic_write
            atomic_write(
                self._status_path,
                json.dumps(status, indent=2) + "\n",
            )
        except Exception:
            # Non-critical for the experiment loop, but a stale status file
            # looks identical to a live one — surface it so consumers of the
            # status file can distinguish a dead monitor from a quiet one.
            logger.warning(
                "Failed to write drift status file %s; status may be stale",
                self._status_path,
                exc_info=True,
            )

    def _dispatch_alerts(self) -> None:
        """Dispatch edge-triggered alerts for thresholds newly crossed.

        For each alert type (composite / vocabulary / behavioral), fire when the
        score crosses *above* its threshold from below, and re-arm when it drops
        back below — one alert per drift episode. A raising hook is isolated so a
        bad webhook never breaks the assessment loop.
        """
        if not self._alert_hooks or self._last_report is None:
            return

        report = self._last_report
        readings = {r.instrument: r for r in report.readings}
        ghost = readings.get("ghost_lexicon")
        behavioral = readings.get("behavioral_footprint")

        candidates = [
            (
                "composite",
                report.composite_score >= self.config.composite_nudge_threshold,
                report.severity,
                f"Full drift detected (composite={report.composite_score:.2f})",
            ),
            (
                "vocabulary",
                ghost is not None and ghost.score >= self.config.ghost_nudge_threshold,
                ghost.severity if ghost else None,
                f"Vocabulary narrowing detected (ghost_lexicon={ghost.score:.2f})"
                if ghost
                else "Vocabulary narrowing detected",
            ),
            (
                "behavioral",
                behavioral is not None
                and behavioral.score >= self.config.behavioral_nudge_threshold,
                behavioral.severity if behavioral else None,
                f"Strategy collapse detected (behavioral_footprint={behavioral.score:.2f})"
                if behavioral
                else "Strategy collapse detected",
            ),
        ]

        for alert_type, active, severity, message in candidates:
            was_firing = self._alert_firing.get(alert_type, False)
            if active and not was_firing:
                self._alert_firing[alert_type] = True
                alert = Alert(
                    type=alert_type,
                    severity=severity,
                    composite_score=report.composite_score,
                    message=message,
                    experiment_count=self._experiment_count,
                    report=report.to_dict(),
                )
                for hook in list(self._alert_hooks):
                    try:
                        hook(alert)
                    except Exception:
                        logger.exception(
                            "alert hook %r failed handling %r", hook, alert_type
                        )
            elif not active and was_firing:
                self._alert_firing[alert_type] = False

    def _strategy_distribution(self) -> dict[str, float]:
        """Current strategy category distribution."""
        if not self._strategy_history:
            return {}
        # Use recent window only (post-boundary)
        recent = self._strategy_history[self.config.anchor_window:]
        if not recent:
            recent = self._strategy_history
        counts: dict[str, int] = {}
        for s in recent:
            counts[s] = counts.get(s, 0) + 1
        total = len(recent)
        return {k: round(v / total, 3) for k, v in sorted(counts.items())}

    def _dominant_strategy(self) -> dict[str, Any]:
        """Find the dominant strategy category in recent experiments."""
        dist = self._strategy_distribution()
        if not dist:
            return {"category": "unknown", "pct": 0.0}
        top = max(dist, key=lambda k: dist[k])
        return {"category": top, "pct": dist[top]}

    def _infer_lost_categories(self, lost_terms: list[str]) -> list[str]:
        """Infer which strategy categories the lost terms belong to."""
        categories = set()
        for term in lost_terms:
            t = term.lower()
            for category, patterns in _STRATEGY_PATTERNS:
                if any(p in t or t in p for p in patterns):
                    categories.add(category)
                    break
            else:
                # Check if the term relates to common strategy concepts
                if any(w in t for w in ["depth", "layer", "head", "window", "mlp"]):
                    categories.add("architecture")
                elif any(w in t for w in ["batch", "gradient"]):
                    categories.add("batch_size")
        return sorted(categories)

    # ------------------------------------------------------------------
    # State persistence (see drift_monitor.persistence for the contract)
    # ------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """Serialize the harness's full state to a plain, versioned dict.

        Captures the three instruments, the scorer weights, and the harness-level
        bookkeeping (experiment count, boundary flag, strategy history). When an
        event log is attached, ``last_event_seq`` records how much of it this
        snapshot already reflects.
        """
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "experiment_count": self._experiment_count,
            "boundary_set": self._boundary_set,
            "strategy_history": list(self._strategy_history),
            "instruments": [
                self._ghost.to_state(),
                self._behavioral.to_state(),
                self._semantic.to_state(),
            ],
            "weights": dict(self._scorer.weights),
        }
        if self._event_log is not None:
            state["last_event_seq"] = self._event_log.next_seq - 1
        return state

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        results_dir: str = ".",
        config: DriftConfig | None = None,
    ) -> "DriftHarness":
        """Reconstruct a harness from a ``to_state()`` dict.

        ``results_dir`` and ``config`` are deployment configuration (output
        paths, thresholds, window sizes) — not drift state — so they are supplied
        on restart rather than persisted. Instruments, weights, and bookkeeping
        counters are restored from ``state``.
        """
        state = migrate_state(state)
        harness = cls(results_dir=results_dir, config=config)
        by_name = {
            i.name: i for i in (instrument_from_state(s) for s in state["instruments"])
        }
        harness._ghost = by_name["ghost_lexicon"]
        harness._behavioral = by_name["behavioral_footprint"]
        harness._semantic = by_name["semantic_drift"]
        harness._experiment_count = int(state.get("experiment_count", 0))
        harness._boundary_set = bool(state.get("boundary_set", False))
        harness._strategy_history = list(state.get("strategy_history", []))
        weights = state.get("weights")
        if weights is not None:
            harness._scorer = DriftScorer(weights=weights)
        return harness

    def save_state(self, path: str | Path) -> None:
        """Write a versioned snapshot of this harness's state atomically."""
        from drift_monitor.persistence import save_state as _save_state

        _save_state(self.to_state(), path)

    @classmethod
    def restore(
        cls,
        snapshot_path: str | Path,
        *,
        results_dir: str = ".",
        config: DriftConfig | None = None,
        event_log_path: str | Path | None = None,
    ) -> "DriftHarness":
        """Reconstruct a harness from a snapshot, fast-forwarding any event tail.

        Loads the snapshot (full state), then replays events whose ``seq`` exceeds
        the snapshot's ``last_event_seq`` to recover observations logged after the
        snapshot. Both the instrument windows *and* the harness bookkeeping
        counters (``experiment_count``, ``strategy_history``) are recovered: the
        counters are recomputed from the replayed ``observe`` events, since every
        non-baseline experiment emits exactly one such event and its ``tools``
        metadata records the strategy category. The event log is detached during
        replay so recovered events are not re-appended to the very log being read.
        """
        from drift_monitor.persistence import load_state, read_events, replay_events

        state = load_state(snapshot_path)
        harness = cls.from_state(state, results_dir=results_dir, config=config)
        if event_log_path is not None:
            harness._event_log = None  # don't re-append while replaying
            events = read_events(event_log_path)
            last_seq = state.get("last_event_seq", -1)
            replay_events(harness, events, last_seq=last_seq)

            # Recover the bookkeeping counters from the replayed tail.
            tail_observes = [
                e
                for e in events
                if e.get("seq", -1) > last_seq and e.get("type") == "observe"
            ]
            harness._experiment_count += len(tail_observes)
            for event in tail_observes:
                tools = event.get("metadata", {}).get("tools", [])
                if tools:
                    harness._strategy_history.append(tools[0])

            ep = Path(event_log_path)
            if not ep.is_absolute():
                ep = Path(results_dir) / ep
            harness._event_log = EventLog(ep)
        return harness


# ---------------------------------------------------------------------------
# Standalone analysis (for existing results.tsv files)
# ---------------------------------------------------------------------------

def analyze_tsv(
    tsv_path: str | Path,
    config: DriftConfig | None = None,
) -> list[dict[str, Any]]:
    """Run drift analysis on an existing results.tsv file.

    Returns a list of assessment records (one per assessment interval).
    """
    import csv

    config = config or DriftConfig()
    harness = DriftHarness(
        results_dir=str(Path(tsv_path).parent),
        config=config,
    )

    # Minimal result-like object
    @dataclass
    class _FakeResult:
        exp: str
        description: str
        status: str
        notes: str

    records = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            result = _FakeResult(
                exp=row.get("exp", ""),
                description=row.get("description", ""),
                status=row.get("status", ""),
                notes=row.get("notes", ""),
            )
            harness.observe_experiment(result)

            if harness.last_report and harness.experiment_count % config.assessment_interval == 0:
                nudge = harness.get_drift_nudge()
                records.append({
                    "experiment_count": harness.experiment_count,
                    "report": harness.last_report.to_dict(),
                    "nudge": nudge,
                    "strategy_dist": harness._strategy_distribution(),
                })

    return records
