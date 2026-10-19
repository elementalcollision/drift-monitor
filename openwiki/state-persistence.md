# State Persistence

**Status:** Accepted (Phase 2)
**Scope:** snapshot + append-only event tail for `DriftMonitor` and `DriftHarness`.

## Decisions

| # | Decision | Choice |
|---|---|---|
| A | Storage format | **A3** — a versioned atomic **snapshot** *and* an append-only **event tail** |
| B | Scope | **B2** — full state (windows, boundary flag, per-instrument config, scorer weights) |
| C | API shape | **C2** — `to_state()`/`from_state()` return/consume plain dicts; thin `save_state()`/`load_state()` wrappers do the file I/O |
| D | Serialization | `json.dumps(..., default=str)`, matching `storage.write_jsonl` |

Everything is versioned behind a `schema_version` field, which both *conditions* how
state is deserialized and *prevents* silent drift between writer and reader (see
[Versioning](#versioning)).

## Two artefacts

1. **Snapshot** (`*.json`) — a self-contained, atomically-written dict (decision B2's
   "full state") carrying `schema_version` and, when an event log is attached,
   `last_event_seq`. This is the fast-load source of truth.

2. **Event tail** (`*.jsonl`) — append-only; every `observe` / `mark_boundary` /
   `reset` is one line, each with a monotonically increasing `seq`. It is the audit
   trail and the crash-recovery source for observations logged *after* the last
   snapshot.

### Why both (A3)

The snapshot alone gives fast, atomic recovery but only up to the moment it was
written; observations after that are lost on crash. The event tail alone requires
replaying the entire history (slow) and offers no versioned config checkpoint.
Together: the snapshot is the checkpoint, the event tail is the durable gap up to
the present. `load`/`restore` fast-forward the tail by re-applying events whose
`seq` exceeds the snapshot's `last_event_seq`.

## Snapshot schema (`schema_version = 1`)

```json
{
  "schema_version": 1,
  "instruments": [
    {
      "name": "ghost_lexicon",
      "config": {"min_freq": 2, "min_length": 4},
      "windows": {
        "window_size": 50,
        "anchor": [{"text": "...", "metadata": {"tools": ["..."]}}],
        "recent": [],
        "boundary_marked": false
      }
    }
  ],
  "weights": {"ghost_lexicon": 0.35, "behavioral_footprint": 0.35, "semantic_drift": 0.30}
}
```

`DriftHarness.to_state()` adds harness-only keys — `experiment_count`,
`boundary_set`, `strategy_history` — and, when an event log is attached,
`last_event_seq`. Deployment configuration (output paths, thresholds, window
sizes) is **not** persisted: it belongs to the restart, not the drift state.

## Event schema (`schema_version = 1`)

```jsonl
{"schema_version": 1, "seq": 0, "type": "observe", "text": "...", "metadata": {"tools": ["architecture"]}}
{"schema_version": 1, "seq": 1, "type": "mark_boundary"}
{"schema_version": 2, "seq": 2, "type": "observe", "text": "...", "metadata": {}}
```

## API reference

### Instruments and `DualWindow`

- `DualWindow.to_state() -> dict` / `DualWindow.from_state(dict) -> DualWindow`
- `Observation.to_state() -> dict`
- `Instrument.to_state() -> dict` — name, config (`_config_to_state()`), windows.
- `Instrument.from_state(dict) -> Instrument` — rebuilds via `_construct_from_state()`
  then restores windows. Subclasses override `_config_to_state`/`_construct_from_state`
  to carry their extra constructor args (`min_freq`/`min_length`, `tool_weight`/
  `length_weight`/`length_metric`, `model_name`/`use_embeddings`).

### `DriftMonitor`

- `to_state() -> dict` — versioned dict of instruments (in order) + weights.
- `from_state(dict) -> DriftMonitor` (classmethod) — validates/migrates the version,
  rebuilds each instrument via the registry, restores weights.
- `save_state(path)` / `load_state(path)` (classmethod) — thin atomic file wrappers.

### `DriftHarness`

- `observe(text, metadata)` / `mark_boundary()` / `reset()` — primitive API over the
  three instruments; the event tail is appended through these.
- `to_state() -> dict` / `from_state(dict, *, results_dir, config)` — full state plus
  harness counters; deployment config passed on restart.
- `save_state(path)` / `restore(snapshot_path, *, results_dir, config, event_log_path)`
  — `restore` loads the snapshot then fast-forwards the event tail, recovering both
  instrument windows *and* the bookkeeping counters from the replayed `observe`
  events (each non-baseline experiment emits exactly one such event, and its
  `tools` metadata records the strategy category).

### `drift_monitor.persistence` (module)

- `STATE_SCHEMA_VERSION`, `EVENT_SCHEMA_VERSION` — current versions.
- `StateVersionError` — raised when a snapshot cannot be safely loaded.
- `save_state(state, path)` / `load_state(path) -> dict` — dict-level file I/O.
- `migrate_state(state, *, current) -> dict` — validate + upgrade in place.
- `register_migration(from_version, fn) -> None` — register a 1-step migration.
- `INSTRUMENT_REGISTRY` / `instrument_from_state(state)` — name→class dispatch.
- `config_diff(instruments) -> dict` — manual inspection of stored-vs-default config.
- `EventLog(path)` — `.observe`, `.mark_boundary`, `.reset`, `.next_seq`.
- `read_events(path)` / `replay_events(target, events, *, last_seq)`.

## Versioning (conditioning & prevention)

`load_state` (and every `from_state`) runs `migrate_state`:

- `schema_version > STATE_SCHEMA_VERSION` → **refuse** (state from a newer build).
- `schema_version < STATE_SCHEMA_VERSION` → walk `register_migration` one version at a
  time; a missing step → **refuse** rather than silently mis-read the old shape.
- `schema_version == STATE_SCHEMA_VERSION` → load.

The version field is the prevention mechanism: no loader ever guesses at an
unrecognised shape. `register_migration` is the conditioning mechanism: known
upgrades are applied explicitly.

`config_diff` exists for *out-of-band* inspection of how stored config diverges from
current library defaults. It is deliberately **not** applied automatically — a
legitimate, explicit choice (e.g. the harness always forcing keyword-mode semantic
drift) would otherwise be reported as drift on every load.

## Known limitations

- The event tail recovers instrument *windows* and — for `DriftHarness.restore` —
  the harness counters. It does not recover `DriftMonitor`-level state beyond what
  `to_state` captures (there is no monitor-level counter distinct from the windows).
- `config_diff` is manual-only; automatic drift detection is out of scope for this
  phase (see the research item in [instruments.md](instruments.md)).
- Metadata values are serialized with `default=str`, so exotic non-string metadata
  would round-trip as strings. Metadata is strings in practice throughout the codebase.

## Test coverage

`tests/test_persistence.py` — instrument/dual-window/monitor/harness round-trips
(score-identical), file round-trips, future-version rejection, missing-migration
rejection, migration application, event-log seq + replay, `last_seq` filtering,
and snapshot→tail crash recovery (windows + counters + event-log resume).