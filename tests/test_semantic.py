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


# ---------------------------------------------------------------------------
# Embedding path (sentence-transformers monkeypatched — no network needed)
# ---------------------------------------------------------------------------


class _FakeTransformer:
    """Deterministic stand-in for SentenceTransformer: vectors from a hash."""

    instances = 0

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        type(self).instances += 1

    def encode(self, texts):
        out = []
        for text in texts:
            h = sum((i + 1) * ord(c) for i, c in enumerate(text)) % 97
            v = [((h >> k) & 1) * 1.0 + 0.1 for k in range(8)]
            out.append(v)
        return out


def test_model_loaded_once_per_process(monkeypatch):
    import drift_monitor.instruments.semantic as sem

    sem._MODEL_CACHE.clear()
    sem._encode_centroid.cache_clear()
    _FakeTransformer.instances = 0
    monkeypatch.setattr(sem, "SentenceTransformer", _FakeTransformer, raising=False)
    monkeypatch.setattr(sem, "_HAS_EMBEDDINGS", True)

    sd = sem.SemanticDrift(use_embeddings=True)
    for t in ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]:
        sd.observe(t)
    sd.mark_boundary()
    for t in ["kappa lambda mu", "nu xi omicron", "pi rho sigma"]:
        sd.observe(t)

    for _ in range(3):
        sd.score()

    # Two window encodes, one model instantiation — not one per score() call.
    assert _FakeTransformer.instances == 1
    assert sem._get_model("all-MiniLM-L6-v2").model_name == "all-MiniLM-L6-v2"


def test_missing_embeddings_warns_and_falls_back(monkeypatch, caplog):
    import logging

    import drift_monitor.instruments.semantic as sem

    monkeypatch.setattr(sem, "_HAS_EMBEDDINGS", False)
    with caplog.at_level(logging.WARNING):
        sd = sem.SemanticDrift(use_embeddings=True)
    assert not sd._use_embeddings
    assert sd.method == "keyword"
    assert any("falling back" in r.message for r in caplog.records)
