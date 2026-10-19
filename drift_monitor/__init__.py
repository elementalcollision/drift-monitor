"""Behavioral drift detection for AI agents across context compression boundaries.

Methodology inspired by morrow.run's compression-monitor (v0.2.1, MIT License).
This is an independent implementation — no code was copied from the original.
See: https://morrow.run for the original research toolkit.
"""

from drift_monitor.instruments.base import InstrumentReading, Severity
from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.instruments.hedging import HedgingLanguage
from drift_monitor.monitor import DriftMonitor
from drift_monitor.scoring import DriftScorer, DriftReport, CompressionType
from drift_monitor.significance import (
    SignificanceResult,
    monitor_significance,
    permutation_test,
)
from drift_monitor.persistence import (
    STATE_SCHEMA_VERSION,
    StateVersionError,
    EventLog,
    read_events,
    replay_events,
    register_migration,
)
from drift_monitor.alerts import Alert, WebhookHook
from drift_monitor.history import SegmentHistory

__version__ = "0.1.0"

__all__ = [
    "DriftMonitor",
    "GhostLexicon",
    "BehavioralFootprint",
    "SemanticDrift",
    "HedgingLanguage",
    "DriftScorer",
    "DriftReport",
    "CompressionType",
    "InstrumentReading",
    "Severity",
    "SignificanceResult",
    "permutation_test",
    "monitor_significance",
    "STATE_SCHEMA_VERSION",
    "StateVersionError",
    "EventLog",
    "read_events",
    "replay_events",
    "register_migration",
    "Alert",
    "WebhookHook",
    "SegmentHistory",
]
