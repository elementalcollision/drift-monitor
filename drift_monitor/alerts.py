"""Out-of-band drift alerts.

Alerts fire when a drift score crosses a threshold (edge-triggered: one alert
per *episode*, re-armed only after the score drops back below the threshold),
complementing the in-band ``DriftHarness.get_drift_nudge()`` prompts.

The delivery surface is a callback core — ``AlertHook``, any ``(Alert) -> Any``
callable — with an optional HTTP webhook adapter (``WebhookHook``) layered on
top. Alerts are therefore opt-in: an instrument/harness with no hooks registered
behaves exactly as before.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from drift_monitor.instruments.base import Severity


def _utc_timestamp() -> str:
    """Current time as an ISO-8601 UTC string (matches the storage layer)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Alert:
    """A single drift alert, emitted when a score crosses its threshold.

    ``type`` is one of ``"composite"``, ``"vocabulary"`` (ghost lexicon), or
    ``"behavioral"``. ``report`` is a full serialized :class:`DriftReport`
    (``to_dict()``), so a consumer can inspect every instrument reading without a
    second call.
    """

    type: str
    severity: Severity
    composite_score: float
    message: str
    experiment_count: int
    report: dict[str, Any]
    timestamp: str = field(default_factory=_utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict (note: ``default=str`` is applied by the webhook)."""
        return {
            "type": self.type,
            "severity": self.severity.value,
            "composite_score": round(self.composite_score, 4),
            "message": self.message,
            "experiment_count": self.experiment_count,
            "report": self.report,
            "timestamp": self.timestamp,
        }


# A hook is any callable that accepts an Alert: a plain function, a lambda, or
# a WebhookHook (which defines __call__). Callable expresses this precisely —
# a Protocol adds nothing here since there are no members beyond __call__.
AlertHook = Callable[[Alert], Any]


class WebhookHook:
    """An :class:`AlertHook` that POSTs the alert as JSON to a URL.

    Uses only the standard library (``urllib.request``), consistent with the
    library's zero-required-dependencies stance. Raises on HTTP error / timeout;
    :class:`DriftHarness` isolates hook failures so one bad webhook never breaks
    the experiment loop.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 5.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def __call__(self, alert: Alert) -> None:
        payload = json.dumps(alert.to_dict(), default=str).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers=self.headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response.read()