"""Tests for the Unicode-aware tokeniser and the opt-in token length metric."""

import re

from drift_monitor.instruments.ghost_lexicon import tokenize
from drift_monitor.instruments.behavioral import BehavioralFootprint, _compute_fingerprint

_OLD_ASCII_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*(?:'[a-zA-Z]+)?")

_ASCII_SAMPLES = [
    "The idempotent retry handler uses memoization to cache previous results.",
    "We need linearizable reads because the sharding strategy distributes tombstone records.",
    "Apply debounce to the vectorization pipeline input; don't over-throttle it.",
    "exp0 baseline: WEIGHT_DECAY=0.05, MATRIX_LR=0.045 (auto-detected depth=8)",
]


def test_ascii_token_stream_identical_to_historical_pattern():
    for text in _ASCII_SAMPLES:
        old = [t.lower() for t in _OLD_ASCII_RE.findall(text)]
        new = tokenize(text)
        assert new == old, f"ASCII token stream changed for: {text!r}"


def test_non_english_text_tokenizes():
    # French/Spanish/German accented text previously tokenized to nothing.
    text = "Le mécanisme de géré automatiquement está configuré für Güterverkehr"
    tokens = tokenize(text)
    assert len(tokens) >= 6
    assert "mécanisme" in tokens
    assert "güterverkehr" in tokens


def test_cjk_does_not_crash_and_yields_tokens():
    tokens = tokenize("機械学習モデルの訓練と検証")
    assert isinstance(tokens, list)  # no segmentation claim — just no silent nothing
    assert len(tokens) >= 1


def test_behavioral_token_length_metric_opt_in():
    fp_chars = _compute_fingerprint(["a bb ccc"], [{}], length_metric="chars")
    fp_tokens = _compute_fingerprint(["a bb ccc"], [{}], length_metric="tokens")
    assert fp_chars.avg_response_length == 8.0  # "a bb ccc" has 8 chars
    assert fp_tokens.avg_response_length == 3.0  # 3 word tokens


def test_behavioral_default_is_chars_for_back_compat():
    bf_default = BehavioralFootprint()
    assert bf_default.length_metric == "chars"


def test_behavioral_rejects_unknown_metric():
    import pytest

    with pytest.raises(ValueError):
        _compute_fingerprint(["x"], [{}], length_metric="words")
