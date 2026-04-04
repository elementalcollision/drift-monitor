"""Tests for the semantic drift instrument."""

from drift_monitor.instruments.semantic import SemanticDrift, _keyword_overlap


def test_keyword_overlap_identical():
    texts = ["The quick brown fox jumps over the lazy dog."]
    assert _keyword_overlap(texts, texts) < 0.01


def test_keyword_overlap_completely_different():
    a = ["quantum computing entanglement superposition qubits"]
    b = ["chocolate cake baking recipe oven temperature"]
    overlap = _keyword_overlap(a, b)
    assert overlap > 0.9


def test_keyword_overlap_partial():
    a = ["machine learning model training neural network optimization"]
    b = ["machine learning inference deployment production serving"]
    overlap = _keyword_overlap(a, b)
    assert 0.1 < overlap < 0.9


def test_no_drift_same_content():
    sd = SemanticDrift(use_embeddings=False)
    texts = [
        "The database uses B-tree indexes for fast lookups.",
        "Query optimization relies on index statistics.",
        "The planner chooses between sequential and index scans.",
    ]
    for t in texts:
        sd.observe(t)
    sd.mark_boundary()
    for t in texts:
        sd.observe(t)

    assert sd.score() < 0.05


def test_topic_change_detected():
    sd = SemanticDrift(use_embeddings=False)
    pre = [
        "The database uses B-tree indexes for fast lookups.",
        "Query optimization relies on index statistics.",
        "The planner chooses between sequential and index scans.",
    ]
    post = [
        "The recipe calls for two cups of flour.",
        "Bake the cake at three hundred fifty degrees.",
        "Frost with chocolate buttercream icing.",
    ]
    for t in pre:
        sd.observe(t)
    sd.mark_boundary()
    for t in post:
        sd.observe(t)

    assert sd.score() > 0.5


def test_method_reports_keyword():
    sd = SemanticDrift(use_embeddings=False)
    assert sd.method == "keyword"


def test_read_includes_method():
    sd = SemanticDrift(use_embeddings=False)
    sd.observe("text")
    sd.mark_boundary()
    sd.observe("text")
    reading = sd.read()
    assert reading.details["method"] == "keyword"


def test_score_zero_before_boundary():
    sd = SemanticDrift(use_embeddings=False)
    sd.observe("some text")
    assert sd.score() == 0.0
