"""Tests for the behavioral footprint instrument."""

from drift_monitor.instruments.behavioral import (
    BehavioralFootprint,
    _compute_fingerprint,
    _distribution_distance,
)


def test_fingerprint_captures_tool_distribution():
    texts = ["response one", "response two"]
    metadata = [{"tools": ["read_file", "grep"]}, {"tools": ["read_file"]}]
    fp = _compute_fingerprint(texts, metadata)
    assert "read_file" in fp.tool_distribution
    assert fp.tool_distribution["read_file"] > fp.tool_distribution["grep"]
    assert fp.total_observations == 2


def test_distribution_distance_identical():
    d = {"a": 0.5, "b": 0.5}
    assert _distribution_distance(d, d) == 0.0


def test_distribution_distance_different():
    d1 = {"a": 1.0}
    d2 = {"b": 1.0}
    dist = _distribution_distance(d1, d2)
    assert dist == 1.0


def test_no_drift_same_behavior():
    bf = BehavioralFootprint()
    for _ in range(5):
        bf.observe("Short response here.", {"tools": ["read_file"]})
    bf.mark_boundary()
    for _ in range(5):
        bf.observe("Short response here.", {"tools": ["read_file"]})

    assert bf.score() < 0.05


def test_tool_shift_detected():
    bf = BehavioralFootprint()
    for _ in range(5):
        bf.observe("response", {"tools": ["read_file", "grep"]})
    bf.mark_boundary()
    for _ in range(5):
        bf.observe("response", {"tools": ["bash", "write_file"]})

    assert bf.score() > 0.2


def test_length_shift_detected():
    bf = BehavioralFootprint()
    for _ in range(5):
        bf.observe("x" * 100)
    bf.mark_boundary()
    for _ in range(5):
        bf.observe("x" * 1000)

    assert bf.score() > 0.3


def test_read_includes_details():
    bf = BehavioralFootprint()
    bf.observe("hello", {"tools": ["read"]})
    bf.mark_boundary()
    bf.observe("hello", {"tools": ["write"]})

    reading = bf.read()
    assert "anchor_fingerprint" in reading.details
    assert "recent_fingerprint" in reading.details
    assert "tool_distance" in reading.details


def test_score_zero_before_boundary():
    bf = BehavioralFootprint()
    bf.observe("text", {"tools": ["read"]})
    assert bf.score() == 0.0
