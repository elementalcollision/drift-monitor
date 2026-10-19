"""Tests for the hedging language instrument."""

from drift_monitor.instruments.hedging import HedgingLanguage, _hedge_density
from drift_monitor.simulate import (
    SAMPLE_PRE_RESPONSES,
    apply_hedging_drift,
    generate_drift_pair,
    DriftMode,
)


def test_zero_before_boundary():
    hl = HedgingLanguage()
    hl.observe("perhaps this might work")
    assert hl.score() == 0.0


def test_silent_on_identical_text():
    hl = HedgingLanguage()
    texts = [
        "The idempotent retry handler caches previous results.",
        "We need linearizable reads because the sharding strategy distributes records.",
    ]
    for t in texts:
        hl.observe(t)
    hl.mark_boundary()
    for t in texts:
        hl.observe(t)
    assert hl.score() == 0.0


def test_detects_hedge_increase():
    hl = HedgingLanguage()
    for t in SAMPLE_PRE_RESPONSES:
        hl.observe(t)
    hl.mark_boundary()
    for t in apply_hedging_drift(SAMPLE_PRE_RESPONSES, seed=1):
        hl.observe(t)

    score = hl.score()
    assert score > 0.2, f"expected hedging drift to fire, got {score}"
    reading = hl.read()
    assert reading.details["recent_hedge_density"] > reading.details["anchor_hedge_density"]


def test_one_sided_confidence_gain_is_not_drift():
    hl = HedgingLanguage()
    for t in apply_hedging_drift(SAMPLE_PRE_RESPONSES, seed=1):
        hl.observe(t)
    hl.mark_boundary()
    for t in SAMPLE_PRE_RESPONSES:
        hl.observe(t)
    # Recent window is MORE assertive than anchor — not a compression symptom.
    assert hl.score() == 0.0


def test_vocabulary_drift_stays_silent():
    # Hedging must not fire on plain vocabulary decay (ghost lexicon's domain).
    pre, post = generate_drift_pair(DriftMode.VOCABULARY, seed=3)
    hl = HedgingLanguage()
    for r in pre:
        hl.observe(r["text"])
    hl.mark_boundary()
    for r in post:
        hl.observe(r["text"])
    assert hl.score() == 0.0


def test_catches_framing_mode_qualifiers():
    # FRAMING's substitutions are literally hedging qualifiers — the hedging
    # instrument is expected to narrow framing's invisibility to the trio.
    pre, post = generate_drift_pair(DriftMode.FRAMING, seed=4)
    hl = HedgingLanguage()
    for r in pre:
        hl.observe(r["text"])
    hl.mark_boundary()
    for r in post:
        hl.observe(r["text"])
    assert hl.score() > 0.1


def test_density_counts_multiword_once():
    density, counts = _hedge_density(["It could potentially fail, perhaps."])
    # "could potentially" is one multi-word hedge, "perhaps" is one more
    assert counts.get("could potentially") == 1
    assert counts.get("perhaps") == 1
    assert 0 < density < 0.5


def test_bare_modals_are_not_hedges():
    # "may"/"might"/"could" are ubiquitous in ordinary prose, and IGNORECASE
    # would make the month "May" a false positive. They must not count as hedges.
    density, counts = _hedge_density(
        ["The system may retry and could log a warning in May."]
    )
    assert density == 0.0
    assert counts == {}
