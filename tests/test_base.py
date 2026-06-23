"""Tests for the base instrument classes: InstrumentReading and Instrument."""

import pytest

from drift_monitor.instruments.base import Instrument, InstrumentReading, Severity


# ---------------------------------------------------------------------------
# Minimal concrete Instrument subclass for testing Instrument._classify / .read
# ---------------------------------------------------------------------------

class _ToyInstrument(Instrument):
    """Concrete instrument with a fixed score for testing the base class."""

    name = "toy"

    def __init__(self, score: float = 0.0) -> None:
        super().__init__()
        self._score = score

    def observe(self, text: str, metadata: dict | None = None) -> None:
        pass

    def score(self) -> float:
        return self._score

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# InstrumentReading.to_dict
# ---------------------------------------------------------------------------

def test_reading_to_dict_basic():
    reading = InstrumentReading(
        instrument="ghost_lexicon",
        score=0.42,
        severity=Severity.HIGH,
    )
    d = reading.to_dict()
    assert d["instrument"] == "ghost_lexicon"
    assert d["score"] == 0.42
    assert d["severity"] == "high"
    assert d["details"] == {}


def test_reading_to_dict_with_details():
    reading = InstrumentReading(
        instrument="behavioral_footprint",
        score=0.78,
        severity=Severity.HIGH,
        details={"tool_distance": 0.55, "length_shift": 0.23},
    )
    d = reading.to_dict()
    assert d["details"] == {"tool_distance": 0.55, "length_shift": 0.23}


def test_reading_to_dict_idempotent():
    reading = InstrumentReading(
        instrument="test_instr",
        score=0.0,
        severity=Severity.LOW,
    )
    d1 = reading.to_dict()
    d2 = reading.to_dict()
    assert d1 == d2


def test_reading_to_dict_roundtrip_severity_value():
    """Severity.value must round-trip through Severity(str)."""
    for sev in Severity:
        reading = InstrumentReading(instrument="x", score=0.0, severity=sev)
        reconstructed = Severity(reading.to_dict()["severity"])
        assert reconstructed is sev


# ---------------------------------------------------------------------------
# Instrument._classify -- severity thresholds (high_threshold=0.3, moderate=0.1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, Severity.LOW),
        (0.01, Severity.LOW),
        (0.05, Severity.LOW),
        (0.10, Severity.LOW),          # exactly at moderate_threshold -> LOW
        (0.1000001, Severity.MODERATE),  # just above moderate_threshold
        (0.11, Severity.MODERATE),
        (0.20, Severity.MODERATE),
        (0.30, Severity.MODERATE),      # exactly at high_threshold -> MODERATE
        (0.3000001, Severity.HIGH),     # just above high_threshold
        (0.31, Severity.HIGH),
        (0.50, Severity.HIGH),
        (0.99, Severity.HIGH),
        (1.00, Severity.HIGH),
    ],
)
def test_classify_default_thresholds(score, expected):
    instr = _ToyInstrument(score=score)
    assert instr._classify(score) is expected


def test_classify_custom_thresholds():
    class _CustomInstrument(Instrument):
        name = "custom"
        high_threshold = 0.6
        moderate_threshold = 0.4

        def observe(self, text, metadata=None):
            pass

        def score(self):
            return 0.0

        def reset(self):
            pass

    instr = _CustomInstrument()
    # 0.5 is below high (0.6) but above moderate (0.4) -> MODERATE
    assert instr._classify(0.5) is Severity.MODERATE
    # 0.65 is above high -> HIGH
    assert instr._classify(0.65) is Severity.HIGH
    # 0.3 is below moderate -> LOW
    assert instr._classify(0.3) is Severity.LOW


# ---------------------------------------------------------------------------
# Instrument.read
# ---------------------------------------------------------------------------

def test_read_returns_reading_with_correct_score():
    instr = _ToyInstrument(score=0.25)
    reading = instr.read()
    assert isinstance(reading, InstrumentReading)
    assert reading.instrument == "toy"
    assert reading.score == 0.25


def test_read_severity_low():
    instr = _ToyInstrument(score=0.05)
    assert instr.read().severity is Severity.LOW


def test_read_severity_moderate():
    instr = _ToyInstrument(score=0.15)
    assert instr.read().severity is Severity.MODERATE


def test_read_severity_high():
    instr = _ToyInstrument(score=0.35)
    assert instr.read().severity is Severity.HIGH


def test_read_details_present():
    instr = _ToyInstrument(score=0.0)
    reading = instr.read()
    assert isinstance(reading.details, dict)


def test_read_score_clamped():
    """read() delegates to score() verbatim -- no clamping inside read()."""
    instr = _ToyInstrument(score=0.777)
    assert instr.read().score == 0.777


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------

def test_severity_values():
    assert Severity.LOW.value == "low"
    assert Severity.MODERATE.value == "moderate"
    assert Severity.HIGH.value == "high"


def test_severity_from_string():
    assert Severity("low") is Severity.LOW
    assert Severity("moderate") is Severity.MODERATE
    assert Severity("high") is Severity.HIGH


def test_core_types_exported_from_package():
    # Severity and InstrumentReading are part of the public surface (a
    # DriftReport carries both) — they should import from the top-level package
    # and be the same objects as in instruments.base.
    import drift_monitor
    from drift_monitor import InstrumentReading as PkgReading
    from drift_monitor import Severity as PkgSeverity

    assert PkgSeverity is Severity
    assert PkgReading is InstrumentReading
    assert "Severity" in drift_monitor.__all__
    assert "InstrumentReading" in drift_monitor.__all__
