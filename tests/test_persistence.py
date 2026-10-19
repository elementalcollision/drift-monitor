"""Tests for state persistence (snapshot + event tail).

Covers the C2 API (``to_state`` / ``from_state``), file wrappers
(``save_state`` / ``load_state``), schema-version drift prevention, instrument
config restoration, and the A3 event tail (``EventLog`` + ``replay_events``).
"""

from dataclasses import dataclass

import pytest

from drift_monitor import (
    STATE_SCHEMA_VERSION,
    DriftMonitor,
    GhostLexicon,
    StateVersionError,
)
from drift_monitor.harness import DriftConfig, DriftHarness
from drift_monitor.persistence import (
    EventLog,
    load_state,
    migrate_state,
    read_events,
    replay_events,
    save_state,
)
from drift_monitor.simulate import DriftMode, generate_drift_pair


# ---------------------------------------------------------------------------
# Instrument-level round-trip
# ---------------------------------------------------------------------------


def test_instrument_roundtrip_preserves_config_and_windows():
    gl = GhostLexicon(window_size=8, min_freq=3, min_length=6)
    for _ in range(3):
        gl.observe("The idempotent retry handler uses memoization to cache results.")
    gl.mark_boundary()
    gl.observe("The idempotent handler caches results.")

    restored = GhostLexicon.from_state(gl.to_state())

    assert restored.min_freq == 3
    assert restored.min_length == 6
    assert restored.windows.anchor.max_size == 8
    assert restored.score() == gl.score()


# ---------------------------------------------------------------------------
# DriftMonitor-level round-trip
# ---------------------------------------------------------------------------


def _filled_monitor() -> DriftMonitor:
    pre, post = generate_drift_pair(DriftMode.COMBINED, seed=7)
    m = DriftMonitor()
    for r in pre:
        m.observe(r["text"], {"tools": r.get("tools", [])})
    m.mark_boundary()
    for r in post:
        m.observe(r["text"], {"tools": r.get("tools", [])})
    return m


def test_monitor_roundtrip_preserves_score():
    m = _filled_monitor()
    expected = m.score().composite_score

    restored = DriftMonitor.from_state(m.to_state())

    assert restored.score().composite_score == expected
    assert [i.score() for i in restored.instruments] == [
        i.score() for i in m.instruments
    ]


def test_monitor_save_load_file_roundtrip(tmp_path):
    m = _filled_monitor()
    path = tmp_path / "state.json"
    m.save_state(path)

    restored = DriftMonitor.load_state(path)

    assert restored.score().composite_score == m.score().composite_score


# ---------------------------------------------------------------------------
# Schema-version drift prevention
# ---------------------------------------------------------------------------


def test_rejects_future_schema_version():
    with pytest.raises(StateVersionError):
        migrate_state({"schema_version": STATE_SCHEMA_VERSION + 1})


def test_rejects_old_version_without_migration():
    # No migration is registered from version 0, so loading refuses rather than
    # silently mis-reading the old shape.
    with pytest.raises(StateVersionError):
        migrate_state({"schema_version": 0})


def test_registered_migration_applies():
    from drift_monitor import persistence

    persistence._MIGRATIONS[0] = lambda s: {**s, "instruments": []}
    try:
        out = migrate_state({"schema_version": 0})
        assert out["schema_version"] == 1
        assert "instruments" in out
    finally:
        persistence._MIGRATIONS.pop(0, None)


def test_malformed_state_not_a_dict(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(StateVersionError):
        load_state(bad)


# ---------------------------------------------------------------------------
# Event tail (A3)
# ---------------------------------------------------------------------------


def test_event_log_seq_and_replay_reconstructs_monitor(tmp_path):
    pre, post = generate_drift_pair(DriftMode.COMBINED, seed=7)

    # Build the "source" monitor and record its events
    src = DriftMonitor()
    log = EventLog(tmp_path / "events.jsonl")
    for r in pre:
        src.observe(r["text"], {"tools": r.get("tools", [])})
        log.observe(r["text"], {"tools": r.get("tools", [])})
    src.mark_boundary()
    log.mark_boundary()
    for r in post:
        src.observe(r["text"], {"tools": r.get("tools", [])})
        log.observe(r["text"], {"tools": r.get("tools", [])})

    # Rebuild a fresh monitor purely from the event log
    rebuilt = DriftMonitor()
    replay_events(rebuilt, read_events(log.path))

    assert rebuilt.score().composite_score == src.score().composite_score


def test_replay_honours_last_seq(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.observe("first", {})
    log.observe("second", {})

    target = DriftMonitor()
    replay_events(target, read_events(log.path), last_seq=0)

    # Only the event with seq 1 ("second") should have been applied.
    assert len(target.instruments[0].windows.anchor) == 1
    assert target.instruments[0].windows.anchor.texts == ["second"]


# ---------------------------------------------------------------------------
# Harness persistence
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    exp: str
    description: str
    status: str
    notes: str


def _feed(harness: DriftHarness, n: int) -> None:
    for i in range(n):
        harness.observe_experiment(
            _Result(
                exp=f"e{i}",
                description=f"experiment {i} tunes batch_size and depth",
                status="complete",
                notes="increased capacity",
            )
        )


def test_harness_roundtrip_preserves_state(tmp_path):
    cfg = DriftConfig(anchor_window=3, assessment_interval=10)
    h = DriftHarness(results_dir=str(tmp_path), config=cfg)
    _feed(h, 7)

    h2 = DriftHarness.from_state(h.to_state(), results_dir=str(tmp_path), config=cfg)

    assert h2.experiment_count == h.experiment_count
    assert h2.boundary_set is h.boundary_set
    assert h2._strategy_history == h._strategy_history
    assert h2._ghost.score() == h._ghost.score()


def test_harness_snapshot_then_restore_recovers_tail(tmp_path):
    cfg = DriftConfig(anchor_window=25, event_log_path="events.jsonl")
    h = DriftHarness(results_dir=str(tmp_path), config=cfg)
    _feed(h, 30)  # boundary auto-set at experiment 25
    h.save_state(tmp_path / "state.json")

    # More experiments after the snapshot — these exist only in the event tail.
    _feed(h, 5)

    h2 = DriftHarness.restore(
        tmp_path / "state.json",
        results_dir=str(tmp_path),
        config=cfg,
        event_log_path=tmp_path / "events.jsonl",
    )

    assert h2.experiment_count == h.experiment_count
    assert h2._ghost.score() == h._ghost.score()
    assert h2._strategy_history == h._strategy_history

    # The restored harness's event log must resume at the correct seq (no
    # double-counted re-append from the replay).
    assert h2._event_log.next_seq == h._event_log.next_seq


def test_save_state_roundtrips_via_module_function(tmp_path):
    m = _filled_monitor()
    path = tmp_path / "s.json"
    save_state(m.to_state(), path)
    assert load_state(path)["schema_version"] == STATE_SCHEMA_VERSION