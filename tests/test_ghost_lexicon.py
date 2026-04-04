"""Tests for the ghost lexicon instrument."""

from drift_monitor.instruments.ghost_lexicon import (
    GhostLexicon,
    tokenize,
    extract_specialized_vocab,
)


def test_tokenize_basic():
    tokens = tokenize("Hello world, this is a test-case.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test-case" in tokens


def test_tokenize_preserves_underscores():
    tokens = tokenize("use_embeddings and model_name")
    assert "use_embeddings" in tokens
    assert "model_name" in tokens


def test_extract_specialized_vocab():
    texts = [
        "The idempotent handler uses memoization for caching.",
        "Apply idempotent logic with memoization to avoid recomputation.",
        "The system processes data efficiently.",
        "The system handles requests properly.",
    ]
    vocab = extract_specialized_vocab(texts, min_freq=2)
    assert "idempotent" in vocab
    assert "memoization" in vocab
    # Common words should be excluded
    assert "the" not in vocab


def test_no_decay_when_same_content():
    gl = GhostLexicon()
    texts = [
        "The idempotent retry handler uses memoization for caching results.",
        "Apply idempotent logic with memoization to the serialization layer.",
        "Check the idempotent guarantee before memoization lookup.",
    ]
    for t in texts:
        gl.observe(t)
    gl.mark_boundary()
    for t in texts:
        gl.observe(t)

    assert gl.score() < 0.05  # Near-zero decay


def test_high_decay_when_vocab_stripped():
    gl = GhostLexicon()
    pre = [
        "The idempotent retry handler uses memoization for caching results.",
        "Apply idempotent logic with memoization to the serialization layer.",
        "Check the idempotent guarantee before memoization lookup.",
    ]
    post = [
        "The process handler uses caching for results.",
        "Apply logic to the processing layer.",
        "Check the guarantee before lookup.",
    ]
    for t in pre:
        gl.observe(t)
    gl.mark_boundary()
    for t in post:
        gl.observe(t)

    score = gl.score()
    assert score > 0.3  # Significant decay


def test_read_returns_lost_terms():
    gl = GhostLexicon()
    pre = [
        "The semaphore guards the mutex acquisition path.",
        "Reset the semaphore after mutex release.",
        "The semaphore count tracks mutex contention.",
    ]
    post = [
        "The lock guards the acquisition path.",
        "Reset the lock after release.",
        "The counter tracks contention.",
    ]
    for t in pre:
        gl.observe(t)
    gl.mark_boundary()
    for t in post:
        gl.observe(t)

    reading = gl.read()
    assert reading.score > 0.1
    assert reading.details["lost_count"] > 0


def test_score_zero_before_boundary():
    gl = GhostLexicon()
    gl.observe("Some text with specialized vocabulary like memoization.")
    assert gl.score() == 0.0


def test_reset_clears_state():
    gl = GhostLexicon()
    gl.observe("text")
    gl.mark_boundary()
    gl.observe("text")
    gl.reset()
    assert gl.score() == 0.0
