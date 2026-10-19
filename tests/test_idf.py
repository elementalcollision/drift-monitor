"""Tests for opt-in IDF weighting (ghost lexicon decay + semantic tf-idf)."""

from drift_monitor.instruments.ghost_lexicon import GhostLexicon, within_window_idf
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.simulate import DriftMode, generate_drift_pair


# ---------------------------------------------------------------------------
# within_window_idf
# ---------------------------------------------------------------------------


def test_idf_ranks_rarity():
    # "a" appears in 2 of 4 documents (rarer) than "b" (4/4)
    idf = within_window_idf(["a b", "a b", "b c", "b c"])
    assert idf["a"] > idf["b"]
    # smooth_idf is never zero, even for the most common term
    assert idf["b"] >= 1.0


def test_idf_layout_deterministic():
    docs = ["alpha beta", "alpha gamma", "beta gamma"]
    assert within_window_idf(docs) == within_window_idf(docs)


def test_idf_empty_input():
    assert within_window_idf([]) == {}


# ---------------------------------------------------------------------------
# Ghost lexicon: rarity-weighted decay
# ---------------------------------------------------------------------------


def _ghost_score(idf_weighting: bool) -> float:
    anchor = [
        "idempotent memoization handler",
        "idempotent memoization handler",
        "handler retry logic",
        "handler retry logic",
    ]
    g = GhostLexicon(min_freq=2, min_length=4, idf_weighting=idf_weighting)
    for text in anchor:
        g.observe(text)
    g.mark_boundary()
    for _ in range(3):
        g.observe("handler retry logic")  # loses idempotent + memoization
    return g.score()


def test_ghost_idf_weights_rare_terms_higher():
    # Losing the rare terms (idempotent/memoization, 2/4 docs) matters more than
    # the common retained term (handler, 4/4 docs), so the weighted score is
    # higher than the unweighted 2/5 = 0.4.
    assert _ghost_score(False) == 0.4
    assert _ghost_score(True) > _ghost_score(False)


def test_ghost_idf_off_is_default():
    assert GhostLexicon().idf_weighting is False
    assert GhostLexicon(idf_weighting=False).idf_weighting is False


# ---------------------------------------------------------------------------
# Semantic: tf-idf keyword overlap
# ---------------------------------------------------------------------------


def _semantic_score(idf_weighting: bool, mode: DriftMode) -> float:
    pre, post = generate_drift_pair(mode, seed=3)
    s = SemanticDrift(use_embeddings=False, idf_weighting=idf_weighting)
    for r in pre:
        s.observe(r["text"])
    s.mark_boundary()
    for r in post:
        s.observe(r["text"])
    return s.score()


def test_semantic_idf_off_is_default():
    assert SemanticDrift().idf_weighting is False


def test_semantic_idf_on_changes_topic_score():
    off = _semantic_score(False, DriftMode.TOPIC)
    on = _semantic_score(True, DriftMode.TOPIC)
    assert on != off


# ---------------------------------------------------------------------------
# Persistence: idf_weighting round-trips and back-compat
# ---------------------------------------------------------------------------


def test_idf_persistence_roundtrip():
    gl = GhostLexicon(idf_weighting=True)
    assert GhostLexicon.from_state(gl.to_state()).idf_weighting is True

    sd = SemanticDrift(use_embeddings=False, idf_weighting=True)
    assert SemanticDrift.from_state(sd.to_state()).idf_weighting is True


def test_pre_idf_state_loads_with_default_false():
    # A state written before idf_weighting existed (its config lacks the key)
    # must load with idf_weighting=False, not raise.
    state = GhostLexicon().to_state()
    del state["config"]["idf_weighting"]
    assert GhostLexicon.from_state(state).idf_weighting is False

    s_state = SemanticDrift(use_embeddings=False).to_state()
    del s_state["config"]["idf_weighting"]
    assert SemanticDrift.from_state(s_state).idf_weighting is False