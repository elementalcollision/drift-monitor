"""Tests for the synthetic drift generator."""

from drift_monitor.simulate import (
    DriftMode,
    apply_vocabulary_drift,
    apply_topic_drift,
    apply_framing_drift,
    generate_drift_pair,
    validate_instruments,
    SAMPLE_PRE_RESPONSES,
    TECHNICAL_TERMS,
)


def test_vocabulary_drift_replaces_terms():
    texts = ["The idempotent handler uses memoization for caching."]
    drifted = apply_vocabulary_drift(texts, replacement_rate=1.0, seed=42)
    # At least one technical term should be replaced
    assert drifted[0] != texts[0]


def test_vocabulary_drift_deterministic():
    texts = SAMPLE_PRE_RESPONSES[:3]
    a = apply_vocabulary_drift(texts, seed=42)
    b = apply_vocabulary_drift(texts, seed=42)
    assert a == b


def test_topic_drift_replaces_all():
    texts = SAMPLE_PRE_RESPONSES[:3]
    drifted = apply_topic_drift(texts, seed=42)
    for orig, new in zip(texts, drifted):
        assert orig != new


def test_framing_drift_preserves_structure():
    texts = ["This must be done and requires careful handling."]
    drifted = apply_framing_drift(texts)
    # Should preserve most words but change modals
    assert "could potentially" in drifted[0] or "might benefit from" in drifted[0]


def test_generate_drift_pair_shapes():
    for mode in DriftMode:
        pre, post = generate_drift_pair(mode, n_samples=3, seed=42)
        assert len(pre) == 3
        assert len(post) == 3
        assert "text" in pre[0]
        assert "text" in post[0]


def test_generate_deterministic():
    pre1, post1 = generate_drift_pair(DriftMode.VOCABULARY, seed=42)
    pre2, post2 = generate_drift_pair(DriftMode.VOCABULARY, seed=42)
    assert pre1 == pre2
    assert post1 == post2


def test_validate_instruments_runs():
    results = validate_instruments(n_trials=3, seed=42)
    assert "vocabulary" in results
    assert "framing" in results
    assert "combined" in results

    # Framing drift should be mostly invisible
    for instr_name, stats in results["framing"].items():
        assert stats["mean_score"] < 0.3, (
            f"{instr_name} detected framing drift (score={stats['mean_score']:.2f})"
        )
