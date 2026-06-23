"""Tests for composite scoring and compression type classification."""

from drift_monitor.instruments.base import InstrumentReading, Severity
from drift_monitor.scoring import CompressionType, DriftScorer


def _reading(name: str, score: float) -> InstrumentReading:
    if score > 0.3:
        sev = Severity.HIGH
    elif score > 0.1:
        sev = Severity.MODERATE
    else:
        sev = Severity.LOW
    return InstrumentReading(instrument=name, score=score, severity=sev)


def test_no_drift():
    scorer = DriftScorer()
    report = scorer.score([
        _reading("ghost_lexicon", 0.02),
        _reading("behavioral_footprint", 0.01),
        _reading("semantic_drift", 0.03),
    ])
    assert report.compression_type == CompressionType.NONE
    assert report.composite_score < 0.1
    assert report.severity == Severity.LOW


def test_vocabulary_only():
    scorer = DriftScorer()
    report = scorer.score([
        _reading("ghost_lexicon", 0.5),
        _reading("behavioral_footprint", 0.05),
        _reading("semantic_drift", 0.03),
    ])
    assert report.compression_type == CompressionType.VOCABULARY_ONLY


def test_operational():
    scorer = DriftScorer()
    report = scorer.score([
        _reading("ghost_lexicon", 0.4),
        _reading("behavioral_footprint", 0.35),
        _reading("semantic_drift", 0.03),
    ])
    assert report.compression_type == CompressionType.OPERATIONAL


def test_full_boundary():
    scorer = DriftScorer()
    report = scorer.score([
        _reading("ghost_lexicon", 0.5),
        _reading("behavioral_footprint", 0.4),
        _reading("semantic_drift", 0.3),
    ])
    assert report.compression_type == CompressionType.FULL_BOUNDARY
    assert report.severity == Severity.HIGH


def test_infrastructure():
    scorer = DriftScorer()
    report = scorer.score([
        _reading("ghost_lexicon", 0.05),
        _reading("behavioral_footprint", 0.6),
        _reading("semantic_drift", 0.02),
    ])
    assert report.compression_type == CompressionType.INFRASTRUCTURE


def test_semantic_only():
    scorer = DriftScorer()
    report = scorer.score([
        _reading("ghost_lexicon", 0.02),
        _reading("behavioral_footprint", 0.05),
        _reading("semantic_drift", 0.4),
    ])
    assert report.compression_type == CompressionType.SEMANTIC_ONLY


def test_ghost_semantic():
    # Ghost + semantic fire but behavioral stays stable — must NOT be labelled
    # FULL_BOUNDARY (only two instruments fired).
    scorer = DriftScorer()
    report = scorer.score([
        _reading("ghost_lexicon", 0.5),
        _reading("behavioral_footprint", 0.02),
        _reading("semantic_drift", 0.4),
    ])
    assert report.compression_type == CompressionType.GHOST_SEMANTIC


def test_composite_clamped_to_unit_interval():
    # Out-of-contract negative reading scores must not yield a negative composite.
    scorer = DriftScorer()
    report = scorer.score([
        _reading("ghost_lexicon", -0.5),
        _reading("behavioral_footprint", -0.3),
        _reading("semantic_drift", -0.2),
    ])
    assert report.composite_score == 0.0


def test_empty_readings():
    scorer = DriftScorer()
    report = scorer.score([])
    assert report.compression_type == CompressionType.NONE
    assert report.composite_score == 0.0


def test_custom_weights():
    scorer = DriftScorer(weights={
        "ghost_lexicon": 0.8,
        "behavioral_footprint": 0.1,
        "semantic_drift": 0.1,
    })
    report = scorer.score([
        _reading("ghost_lexicon", 0.5),
        _reading("behavioral_footprint", 0.05),
        _reading("semantic_drift", 0.03),
    ])
    # Heavy ghost weight should push composite higher
    assert report.composite_score > 0.3


def test_fired_property():
    scorer = DriftScorer()
    report = scorer.score([
        _reading("ghost_lexicon", 0.5),
        _reading("behavioral_footprint", 0.05),
        _reading("semantic_drift", 0.03),
    ])
    assert "ghost_lexicon" in report.fired
    assert "behavioral_footprint" not in report.fired


def test_to_dict():
    scorer = DriftScorer()
    report = scorer.score([_reading("ghost_lexicon", 0.5)])
    d = report.to_dict()
    assert "composite_score" in d
    assert "compression_type" in d
    assert "readings" in d
