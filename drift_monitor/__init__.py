"""Behavioral drift detection for AI agents across context compression boundaries.

Methodology inspired by morrow.run's compression-monitor (v0.2.1, MIT License).
This is an independent implementation — no code was copied from the original.
See: https://morrow.run for the original research toolkit.
"""

from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.scoring import DriftScorer, DriftReport, CompressionType

__version__ = "0.1.0"

__all__ = [
    "GhostLexicon",
    "BehavioralFootprint",
    "SemanticDrift",
    "DriftScorer",
    "DriftReport",
    "CompressionType",
]
