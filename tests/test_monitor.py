"""Tests for the DriftMonitor convenience facade."""

from drift_monitor import DriftMonitor, DriftReport
from drift_monitor.scoring import CompressionType


def _drive(monitor, pre, post, tools_pre=None, tools_post=None):
    for t in pre:
        monitor.observe(t, {"tools": tools_pre or []})
    monitor.mark_boundary()
    for t in post:
        monitor.observe(t, {"tools": tools_post or []})
    return monitor.score()


def test_facade_returns_drift_report():
    m = DriftMonitor()
    report = _drive(
        m,
        ["the idempotent handler uses memoization and backpressure"] * 3,
        ["the process handles the operation in a generic way"] * 3,
    )
    assert isinstance(report, DriftReport)
    assert 0.0 <= report.composite_score <= 1.0


def test_facade_detects_vocabulary_drift():
    m = DriftMonitor()
    report = _drive(
        m,
        ["idempotent memoization backpressure linearizable sharding quorum"] * 4,
        ["process handle manage configure operation function method"] * 4,
    )
    # Specialized vocabulary is lost across the boundary -> drift is detected.
    assert report.composite_score > 0.0


def test_facade_no_drift_when_stable():
    m = DriftMonitor()
    text = "idempotent memoization backpressure linearizable sharding"
    report = _drive(m, [text] * 4, [text] * 4)
    assert report.compression_type == CompressionType.NONE


def test_facade_reset_clears_state():
    m = DriftMonitor()
    m.observe("idempotent memoization backpressure", {"tools": ["a"]})
    m.mark_boundary()
    m.observe("generic process handle", {"tools": ["b"]})
    m.reset()
    # After reset there is no boundary -> a fresh, no-drift report.
    report = m.score()
    assert report.composite_score == 0.0
    assert report.compression_type == CompressionType.NONE


def test_facade_custom_weights_passed_through():
    m = DriftMonitor(weights={
        "ghost_lexicon": 0.8,
        "behavioral_footprint": 0.1,
        "semantic_drift": 0.1,
    })
    assert m.scorer.weights["ghost_lexicon"] == 0.8
