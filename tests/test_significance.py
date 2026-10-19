"""Tests for permutation-test significance of composite drift scores."""

import pytest

from drift_monitor import DriftMonitor
from drift_monitor.simulate import DriftMode, generate_drift_pair
from drift_monitor.significance import permutation_test, monitor_significance
from drift_monitor.window import Observation


def _observations(records):
    return [Observation(text=r["text"], metadata={"tools": r.get("tools", [])}) for r in records]


def test_real_drift_is_significant():
    pre, post = generate_drift_pair(DriftMode.COMBINED, seed=42)
    # Amplify sample size so the test has power
    pre = pre * 4
    post = post * 4
    result = permutation_test(_observations(pre), _observations(post), n_permutations=100, seed=0)
    assert result.observed_composite > 0.2
    assert result.p_value <= 0.05, (
        f"combined drift should be distinguishable from a random split "
        f"(observed={result.observed_composite:.3f}, p={result.p_value:.3f})"
    )
    assert result.significant_at_5pct


def test_identical_windows_not_significant():
    pre, _ = generate_drift_pair(DriftMode.TOPIC, seed=42)
    anchor = _observations(pre)
    recent = _observations(pre)  # same content
    result = permutation_test(anchor, recent, n_permutations=100, seed=0)
    # Same distribution on both sides: the observed composite is typical of
    # random splits, so p should be high.
    assert result.p_value > 0.05
    assert not result.significant_at_5pct


def test_deterministic_given_seed():
    pre, post = generate_drift_pair(DriftMode.COMBINED, seed=42)
    a = permutation_test(_observations(pre), _observations(post), n_permutations=50, seed=7)
    b = permutation_test(_observations(pre), _observations(post), n_permutations=50, seed=7)
    assert a.p_value == b.p_value
    assert a.null_composites == b.null_composites


def test_p_value_smoothing_never_zero():
    pre, post = generate_drift_pair(DriftMode.COMBINED, seed=42)
    result = permutation_test(_observations(pre), _observations(post), n_permutations=20, seed=0)
    assert result.p_value >= 1 / 21


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        permutation_test([], [], n_permutations=0)


def test_monitor_convenience_wrapper():
    monitor = DriftMonitor()
    pre, post = generate_drift_pair(DriftMode.COMBINED, seed=42)
    for r in pre * 4:
        monitor.observe(r["text"], {"tools": r.get("tools", [])})
    monitor.mark_boundary()
    for r in post * 4:
        monitor.observe(r["text"], {"tools": r.get("tools", [])})

    result = monitor_significance(monitor, n_permutations=50, seed=0)
    assert result.p_value <= 0.05


def test_monitor_wrapper_requires_boundary():
    monitor = DriftMonitor()
    monitor.observe("text")
    with pytest.raises(ValueError):
        monitor_significance(monitor)


def test_monitor_wrapper_rejects_custom_instruments():
    from drift_monitor.instruments.hedging import HedgingLanguage

    monitor = DriftMonitor()
    monitor.instruments.append(HedgingLanguage())  # diverges from the trio
    pre, post = generate_drift_pair(DriftMode.COMBINED, seed=42)
    for r in pre:
        monitor.observe(r["text"], {"tools": r.get("tools", [])})
    monitor.mark_boundary()
    for r in post:
        monitor.observe(r["text"], {"tools": r.get("tools", [])})
    with pytest.raises(ValueError, match="standard trio"):
        monitor_significance(monitor, n_permutations=10)


def test_monitor_wrapper_rejects_embedding_mode():
    monitor = DriftMonitor()
    # Simulate a monitor whose semantic instrument resolved to embeddings; the
    # permutation null is keyword-mode and would silently diverge.
    monitor.instruments[2]._use_embeddings = True
    pre, post = generate_drift_pair(DriftMode.COMBINED, seed=42)
    for r in pre:
        monitor.observe(r["text"], {"tools": r.get("tools", [])})
    monitor.mark_boundary()
    for r in post:
        monitor.observe(r["text"], {"tools": r.get("tools", [])})
    with pytest.raises(ValueError, match="embedding"):
        monitor_significance(monitor, n_permutations=10)
