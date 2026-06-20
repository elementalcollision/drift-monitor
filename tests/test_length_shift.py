"""Tests for _length_shift in drift_monitor/instruments/behavioral.py."""

import pytest

from drift_monitor.instruments.behavioral import (
    BehaviorFingerprint,
    _length_shift,
)


def test_identical_lengths_return_zero():
    """Identical average response lengths should return 0.0."""
    fp1 = BehaviorFingerprint(avg_response_length=100.0)
    fp2 = BehaviorFingerprint(avg_response_length=100.0)
    assert _length_shift(fp1, fp2) == 0.0


def test_both_zero_return_zero():
    """Both zero should return 0.0."""
    fp1 = BehaviorFingerprint(avg_response_length=0.0)
    fp2 = BehaviorFingerprint(avg_response_length=0.0)
    assert _length_shift(fp1, fp2) == 0.0


def test_large_difference_capped_at_one():
    """A large length difference should be capped at 1.0.

    The cap is via min(1.0, |a-b|/max(a,b)). For non-zero pairs,
    |a-b|/max(a,b) < 1.0 always, so we verify via one-zero case.
    """
    fp1 = BehaviorFingerprint(avg_response_length=0.0)
    fp2 = BehaviorFingerprint(avg_response_length=1000.0)
    assert _length_shift(fp1, fp2) == 1.0
    # Also verify that a very large ratio but both nonzero stays below 1.0
    fp3 = BehaviorFingerprint(avg_response_length=1.0)
    fp4 = BehaviorFingerprint(avg_response_length=1_000_000.0)
    score = _length_shift(fp3, fp4)
    assert score <= 1.0
    assert score == pytest.approx(999_999 / 1_000_000, rel=1e-6)


def test_symmetric_in_argument_order():
    """The score should be symmetric in argument order."""
    fp1 = BehaviorFingerprint(avg_response_length=50.0)
    fp2 = BehaviorFingerprint(avg_response_length=100.0)
    assert _length_shift(fp1, fp2) == _length_shift(fp2, fp1)


def test_one_zero_one_nonzero():
    """One zero and one nonzero should return 1.0 (full shift)."""
    fp1 = BehaviorFingerprint(avg_response_length=0.0)
    fp2 = BehaviorFingerprint(avg_response_length=100.0)
    assert _length_shift(fp1, fp2) == 1.0


def test_moderate_difference():
    """A moderate difference should return a value between 0 and 1."""
    fp1 = BehaviorFingerprint(avg_response_length=100.0)
    fp2 = BehaviorFingerprint(avg_response_length=150.0)
    score = _length_shift(fp1, fp2)
    # denominator = max(100, 150, 1.0) = 150
    # mean_shift = abs(100 - 150) / 150 = 50 / 150 = 1/3
    assert score == pytest.approx(1 / 3, rel=1e-4)


def test_denominator_uses_max():
    """The denominator should be max of the two values (floored at 1.0)."""
    fp1 = BehaviorFingerprint(avg_response_length=10.0)
    fp2 = BehaviorFingerprint(avg_response_length=20.0)
    score = _length_shift(fp1, fp2)
    # denominator = max(10, 20, 1.0) = 20
    # mean_shift = abs(10 - 20) / 20 = 10 / 20 = 0.5
    assert score == pytest.approx(0.5, rel=1e-4)


def test_denominator_floor_at_one():
    """When both values are less than 1.0, denominator should be 1.0."""
    fp1 = BehaviorFingerprint(avg_response_length=0.1)
    fp2 = BehaviorFingerprint(avg_response_length=0.5)
    score = _length_shift(fp1, fp2)
    # denominator = max(0.1, 0.5, 1.0) = 1.0
    # mean_shift = abs(0.1 - 0.5) / 1.0 = 0.4
    assert score == pytest.approx(0.4, rel=1e-4)


def test_small_nonzero_values_below_one():
    """When both values are below 1.0 and nonzero, denominator floors to 1.0."""
    fp1 = BehaviorFingerprint(avg_response_length=0.3)
    fp2 = BehaviorFingerprint(avg_response_length=0.9)
    score = _length_shift(fp1, fp2)
    # denominator = max(0.3, 0.9, 1.0) = 1.0
    # mean_shift = abs(0.3 - 0.9) / 1.0 = 0.6
    assert score == pytest.approx(0.6, rel=1e-4)
