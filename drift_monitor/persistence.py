"""State persistence for drift monitoring (snapshot + append-only event tail).

Implements decision A3: a versioned, self-contained **snapshot** (fast load,
full state) plus an append-only **event tail** (audit trail + crash recovery).
The snapshot is authoritative for state; the event tail records every
``observe`` / ``mark_boundary`` / ``reset`` so that observations after the last
snapshot's ``last_event_seq`` can be replayed to catch up exactly.

Serialization contract
----------------------
- ``to_state()`` / ``from_state()`` are class methods (decision C2) that move
  between living objects and plain dictionaries; this module adds the *file*
  layer (``save_state`` / ``load_state``) and the event tail on top.
- State is full (decision B2): windows, boundary flag, per-instrument config,
  and scorer weights are all captured, so ``from_state`` reconstructs an
  equivalent object with no external inputs.
- All JSON uses ``default=str`` (decision D, matching ``storage.write_jsonl``),
  so exotic metadata values degrade to strings rather than raising.

Versioning & drift prevention
-----------------------------
Every snapshot carries ``schema_version``. Loading enforces, loudly:

* ``schema_version > STATE_SCHEMA_VERSION`` — the state was written by a newer
  version; refuse (cannot safely downgrade).
* ``schema_version < STATE_SCHEMA_VERSION`` — walk the registered migration
  chain (``register_migration``), one version at a time; if a step is missing,
  refuse rather than silently mis-read the old shape.

This is the "conditioning/prevention" mechanism: the version field conditions
how state is deserialized and prevents silent drift between writer and reader.

The ``config_diff`` helper exists for *manual, out-of-band* inspection of how a
stored instrument's config diverges from that class's current library defaults
— it is not applied automatically (see the module for why that would be noisy).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from drift_monitor.instruments.base import Instrument
from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.hedging import HedgingLanguage
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.storage import append_jsonl, atomic_write, read_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

STATE_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1


class StateVersionError(ValueError):
    """Raised when a snapshot's ``schema_version`` cannot be safely loaded."""


_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


def register_migration(from_version: int, fn: Callable[[dict], dict]) -> None:
    """Register a migration turning state at ``from_version`` into ``from_version + 1``.

    ``fn`` receives the full state dict and must return the same dict advanced
    one schema version (it is the responsibility of the migration to update the
    shape; ``migrate_state`` rewrites ``schema_version`` itself).
    """
    _MIGRATIONS[from_version] = fn


def migrate_state(state: dict[str, Any], *, current: int = STATE_SCHEMA_VERSION) -> dict[str, Any]:
    """Validate and upgrade ``state`` to ``current`` schema version in place.

    - ``> current`` → :class:`StateVersionError` (state from a newer build).
    - ``< current`` → apply the registered migration chain; missing step → error.
    """
    version = int(state.get("schema_version", 0))
    if version > current:
        raise StateVersionError(
            f"state schema_version {version} is newer than this library supports "
            f"({current}); load it with a matching drift-monitor version"
        )
    while version < current:
        upgrade = _MIGRATIONS.get(version)
        if upgrade is None:
            raise StateVersionError(
                f"state schema_version {version} predates the current version "
                f"({current}) and no migration is registered for {version} -> {version + 1}"
            )
        state = upgrade(state)
        version += 1
        state["schema_version"] = version
    return state


# ---------------------------------------------------------------------------
# Snapshot file I/O (thin wrappers over the class to_state/from_state methods)
# ---------------------------------------------------------------------------

def save_state(state: dict[str, Any], path: str | Path) -> None:
    """Write a state dict (from ``to_state()``) atomically as JSON.

    ``default=str`` keeps the write from failing on non-JSON-serializable
    metadata, matching :func:`drift_monitor.storage.write_jsonl`.
    """
    atomic_write(Path(path), json.dumps(state, indent=2, default=str) + "\n")


def load_state(path: str | Path) -> dict[str, Any]:
    """Read a state dict from disk and validate/migrate its schema version."""
    raw: Any = None
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise
    if not isinstance(raw, dict):
        raise StateVersionError(f"state file {path} does not contain a JSON object")
    return migrate_state(raw)


# ---------------------------------------------------------------------------
# Instrument reconstruction (registry dispatch)
# ---------------------------------------------------------------------------

INSTRUMENT_REGISTRY: dict[str, type[Instrument]] = {
    cls.name: cls
    for cls in (GhostLexicon, BehavioralFootprint, SemanticDrift, HedgingLanguage)
}


def instrument_from_state(state: dict[str, Any]) -> Instrument:
    """Reconstruct an instrument from its ``to_state()`` dict."""
    name = state.get("name")
    cls = INSTRUMENT_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(INSTRUMENT_REGISTRY))
        raise StateVersionError(
            f"unknown instrument {name!r} in state; known instruments: {known}"
        )
    return cls.from_state(state)


def config_diff(instruments: list[Instrument]) -> dict[str, list[str]]:
    """Report how each instrument's stored config differs from its class defaults.

    For manual, out-of-band drift inspection only. It is deliberately *not*
    wired into ``from_state``: a legitimate, explicit choice (e.g. the harness
    always forces keyword-mode semantic drift) would otherwise be misreported
    as drift on every load.
    """
    drifting: dict[str, list[str]] = {}
    for instr in instruments:
        default_cfg = instr.__class__()._config_to_state()
        stored_cfg = instr._config_to_state()
        diffs = [
            f"{k}: stored={stored_cfg.get(k)!r}, library_default={default_cfg.get(k)!r}"
            for k in sorted(set(default_cfg) | set(stored_cfg))
            if stored_cfg.get(k) != default_cfg.get(k)
        ]
        if diffs:
            drifting[instr.name] = diffs
    return drifting


# ---------------------------------------------------------------------------
# Event tail (append-only audit + crash recovery)
# ---------------------------------------------------------------------------

_EVENT_OBSERVE = "observe"
_EVENT_BOUNDARY = "mark_boundary"
_EVENT_RESET = "reset"


class EventLog:
    """Append-only JSONL of observation events.

    Each line is ``{"schema_version": 1, "seq": n, "type": ..., ...}``. ``seq``
    is a monotonically increasing per-log counter, so a snapshot can record its
    ``last_event_seq`` and a reader can replay only the events that came after.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        events = read_events(self.path)
        self._seq = (events[-1]["seq"] + 1) if events else 0

    @property
    def next_seq(self) -> int:
        """The sequence number the next appended event will carry."""
        return self._seq

    def observe(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._append({"type": _EVENT_OBSERVE, "text": text, "metadata": metadata or {}})

    def mark_boundary(self) -> None:
        self._append({"type": _EVENT_BOUNDARY})

    def reset(self) -> None:
        self._append({"type": _EVENT_RESET})

    def _append(self, payload: dict[str, Any]) -> None:
        event = dict(payload)
        event["schema_version"] = EVENT_SCHEMA_VERSION
        event["seq"] = self._seq
        append_jsonl(self.path, event)
        self._seq += 1


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read an event log; corrupt trailing lines are skipped (as in ``read_jsonl``)."""
    return read_jsonl(path)


def replay_events(
    target: Any,
    events: list[dict[str, Any]],
    *,
    last_seq: int = -1,
) -> None:
    """Apply ``observe`` / ``mark_boundary`` / ``reset`` events to ``target``.

    Only events with ``seq > last_seq`` are applied, so a caller can fast-forward
    from a snapshot's ``last_event_seq`` to the tail without double-counting.
    """
    for event in events:
        seq = event.get("seq", -1)
        if seq <= last_seq:
            continue
        kind = event.get("type")
        if kind == _EVENT_OBSERVE:
            target.observe(event["text"], event.get("metadata", {}))
        elif kind == _EVENT_BOUNDARY:
            target.mark_boundary()
        elif kind == _EVENT_RESET:
            target.reset()
        else:
            logger.warning("replay_events: skipping unknown event type %r (seq=%s)", kind, seq)