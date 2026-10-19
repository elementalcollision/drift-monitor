"""Tests for out-of-band drift alerts (edge-triggered + callback/webhook delivery)."""

import json
from dataclasses import dataclass

from drift_monitor import Alert, WebhookHook
from drift_monitor.harness import DriftConfig, DriftHarness
from drift_monitor.instruments.base import InstrumentReading, Severity
from drift_monitor.scoring import CompressionType, DriftReport


def _reading(name: str, score: float) -> InstrumentReading:
    sev = Severity.HIGH if score > 0.3 else (Severity.MODERATE if score > 0.1 else Severity.LOW)
    return InstrumentReading(instrument=name, score=score, severity=sev, details={})


def _report(
    composite: float,
    *,
    ghost: float = 0.0,
    behavioral: float = 0.0,
    semantic: float = 0.0,
) -> DriftReport:
    sev = Severity.HIGH if composite > 0.3 else (Severity.MODERATE if composite > 0.1 else Severity.LOW)
    return DriftReport(
        composite_score=composite,
        compression_type=CompressionType.NONE,
        readings=[
            _reading("ghost_lexicon", ghost),
            _reading("behavioral_footprint", behavioral),
            _reading("semantic_drift", semantic),
        ],
        severity=sev,
    )


def _harness(tmp_path) -> DriftHarness:
    return DriftHarness(results_dir=str(tmp_path), config=DriftConfig())


# ---------------------------------------------------------------------------
# Edge-triggered + re-arm behaviour
# ---------------------------------------------------------------------------


def test_composite_fires_once_then_rearms(tmp_path):
    h = _harness(tmp_path)
    fired = []
    h.add_alert_hook(lambda a: fired.append(a.type))

    h._last_report = _report(composite=0.5)  # crosses above 0.30
    h._dispatch_alerts()
    assert fired == ["composite"]

    h._dispatch_alerts()  # still above -> no new alert
    assert fired == ["composite"]

    h._last_report = _report(composite=0.1)  # drops below -> re-arm
    h._dispatch_alerts()
    assert fired == ["composite"]

    h._last_report = _report(composite=0.5)  # crosses again -> fire again
    h._dispatch_alerts()
    assert fired == ["composite", "composite"]


def test_vocabulary_and_behavioral_fire_independently(tmp_path):
    h = _harness(tmp_path)
    fired = []
    h.add_alert_hook(lambda a: fired.append(a.type))

    # composite below threshold, but both instruments above their own
    h._last_report = _report(composite=0.1, ghost=0.6, behavioral=0.6)
    h._dispatch_alerts()
    assert sorted(fired) == ["behavioral", "vocabulary"]


def test_below_threshold_fires_nothing(tmp_path):
    h = _harness(tmp_path)
    fired = []
    h.add_alert_hook(lambda a: fired.append(a.type))
    h._last_report = _report(composite=0.05, ghost=0.05, behavioral=0.05)
    h._dispatch_alerts()
    assert fired == []


def test_no_hooks_is_a_noop(tmp_path):
    h = _harness(tmp_path)
    h._last_report = _report(composite=0.9)
    h._dispatch_alerts()  # must not raise, must not fire anything


def test_alert_carries_full_payload(tmp_path):
    h = _harness(tmp_path)
    captured = []
    h.add_alert_hook(lambda a: captured.append(a))
    h._last_report = _report(composite=0.5)
    h._dispatch_alerts()

    alert = captured[0]
    assert alert.type == "composite"
    assert alert.severity is Severity.HIGH
    assert alert.composite_score == 0.5
    assert alert.experiment_count == 0
    assert isinstance(alert.message, str) and alert.message
    assert "composite_score" in alert.report  # serialized report embedded
    assert alert.timestamp  # non-empty UTC stamp


def test_hook_failure_is_isolated(tmp_path):
    h = _harness(tmp_path)

    def bad(alert):
        raise RuntimeError("boom")

    good = []
    h.add_alert_hook(bad)
    h.add_alert_hook(lambda a: good.append(a.type))
    h._last_report = _report(composite=0.5)
    h._dispatch_alerts()  # must not raise despite the failing hook

    assert good == ["composite"]


def test_remove_and_clear_hooks(tmp_path):
    h = _harness(tmp_path)
    fired = []

    def hook(a):
        fired.append(a.type)

    h.add_alert_hook(hook)
    h.remove_alert_hook(hook)
    h._last_report = _report(composite=0.5)
    h._dispatch_alerts()
    assert fired == []

    h.add_alert_hook(hook)
    h.clear_alert_hooks()
    h._dispatch_alerts()
    assert fired == []


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------


def test_webhook_hook_posts_json(monkeypatch):
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b""

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["data"] = request.data
        captured["content_type"] = request.headers.get("Content-type")
        return _Resp()

    monkeypatch.setattr("drift_monitor.alerts.urllib.request.urlopen", fake_urlopen)

    hook = WebhookHook("https://hooks.example/alerts")
    alert = Alert(
        type="composite",
        severity=Severity.HIGH,
        composite_score=0.5,
        message="m",
        experiment_count=3,
        report={},
    )
    hook(alert)

    assert captured["url"] == "https://hooks.example/alerts"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    body = json.loads(captured["data"].decode("utf-8"))
    assert body["type"] == "composite"
    assert body["severity"] == "high"
    assert body["composite_score"] == 0.5


# ---------------------------------------------------------------------------
# Integration: a real assessment dispatch
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    exp: str
    description: str
    status: str
    notes: str


def test_alerts_dispatch_from_run_assessment(tmp_path):
    cfg = DriftConfig(anchor_window=3, assessment_interval=1)
    h = DriftHarness(results_dir=str(tmp_path), config=cfg)
    fired = []
    h.add_alert_hook(lambda a: fired.append(a.type))

    rich = "idempotent handler uses memoization to cache sharding strategy results"
    poor = "handler caches results"

    for i in range(3):
        h.observe_experiment(_Result(f"e{i}", rich, "complete", ""))
    for i in range(3, 9):
        h.observe_experiment(_Result(f"e{i}", poor, "complete", ""))

    assert fired, "expected at least one alert on vocabulary decay"
    assert set(fired) <= {"composite", "vocabulary", "behavioral"}