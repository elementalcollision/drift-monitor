---
type: "Reference"
title: "CLI & Testing"
description: "The drift-monitor CLI subcommands, JSONL storage layer, synthetic drift simulation, the test suite, and the bundled validation data from RTX 5090 runs."
tags: [cli, testing, storage, simulation, validation]
generated: { by: "openwiki/0.7.2", at: "2026-10-19T08:52:43.736Z" }
---

# CLI & Testing

## CLI

**Source:** `drift_monitor/cli.py`

Entry point: `drift-monitor` (registered in `pyproject.toml` as `[project.scripts]`).

Also runnable as a module: `python -m drift_monitor`

### Commands

#### `drift-monitor run --pre anchor.jsonl --post recent.jsonl`

Runs all three instruments (GhostLexicon, BehavioralFootprint, SemanticDrift) on pre/post JSONL files and prints a composite `DriftReport` as JSON.

Common options for `run`, `ghost-lexicon`, `behavioral`, `semantic`, `hedging`:
- `--pre` (required): Pre-compression JSONL file path.
- `--post` (required): Post-compression JSONL file path.
- `--text-field` (default: `text`): JSON field containing the text content.
- `--tools-field` (default: `tools`): JSON field containing the tool list.

`run`-only options:
- `--permutations N` (default `0`): if > 0, run a permutation test with `N` reshuffles and add a `significance` block (p-value) to the JSON output. Uses keyword-mode semantics regardless of the instruments' mode.
- `--seed` (default `0`): random seed for the permutation test.

#### `drift-monitor ghost-lexicon --pre ... --post ...`

Runs only the Ghost Lexicon instrument and prints its `InstrumentReading` as JSON.

#### `drift-monitor behavioral --pre ... --post ...`

Runs only the Behavioral Footprint instrument.

#### `drift-monitor semantic --pre ... --post ...`

Runs only the Semantic Drift instrument (keyword overlap mode; embeddings not used in CLI).

#### `drift-monitor hedging --pre ... --post ...`

Runs only the opt-in Hedging Language instrument and prints its `InstrumentReading` as JSON. Not part of `run` — it is a standalone surface for the lexical confidence-erosion detector.

#### `drift-monitor demo`

Runs an end-to-end demo with synthetic data across all `DriftMode` values. Prints per-instrument scores, severity, composite score, and compression type for each mode.

#### `drift-monitor validate --trials 10 --seed 42`

Runs the validation suite: generates synthetic drift pairs across all modes, runs each instrument, and reports detection rates. For `framing` mode the surface trio (ghost lexicon, behavioral, semantic) must stay **undetected** — a detection rate ≥ 30% is a FAIL (the validator exits with code 1). The hedging instrument is expected to catch `framing`'s hedging qualifiers and is marked `ok` for that mode; other instruments are marked `ok` when their detection rate clears.

### JSONL Input Format

The CLI expects newline-delimited JSON with `text` and `tools` fields:

```json
{"text": "The idempotent handler retries with memoization.", "tools": ["read_file", "grep"]}
{"text": "Apply throttle to prevent backpressure.", "tools": ["grep", "edit_file"]}
```

### CLI Internal Flow

The CLI internal flow (see `_load_and_run()` in `cli.py`):
1. Reads pre/post JSONL via `read_jsonl()`; a missing-but-present file (permission error, directory, etc.) is converted to a friendly stderr message + `exit(1)`. Empty pre/post files also exit `1`.
2. Warns (but does not fail) if no record contains the requested `--text-field` — a wrong field name would otherwise silently yield empty text and zero drift.
3. Creates all three instruments with `SemanticDrift(use_embeddings=False)`.
4. Feeds pre-records to each instrument (extracting `text` and `tools` fields).
5. Calls `mark_boundary()` on each instrument.
6. Feeds post-records.
7. Collects readings, scores with `DriftScorer()`, returns `report.to_dict()`.
8. If `--permutations > 0`, runs `permutation_test()` on the first instrument's windows (all instruments share the same stream) and adds a `significance` block.

## Storage Layer

**Source:** `drift_monitor/storage.py`

All persistence uses JSONL (newline-delimited JSON) for interoperability.

| Function | Purpose |
|---|---|
| `atomic_write(path, content)` | Write file atomically via temp-file-then-rename |
| `atomic_append(path, line)` | Append a line to a file |
| `write_jsonl(path, records)` | Write a list of dicts as JSONL (atomic) |
| `append_jsonl(path, record)` | Append a single dict as a JSONL line |
| `read_jsonl(path)` | Read JSONL, skipping corrupt lines (common after crashes) |
| `load_texts_from_jsonl(path, text_field)` | Read JSONL, filtering to records with the text field |
| `save_drift_report(path, report_dict)` | Append a drift report with timestamp |

## Synthetic Drift Simulation

**Source:** `drift_monitor/simulate.py`

The simulation module generates pre/post compression text pairs with known drift characteristics to validate instrument behavior.

### DriftMode Enum

| Mode | Description |
|---|---|
| `VOCABULARY` | Replaces technical terms (e.g., "idempotent") with generic alternatives (e.g., "process") at 70% replacement rate |
| `TOPIC` | Replaces all text with generic filler sentences simulating complete focus loss |
| `TOOLCALL` | Shifts tool usage distribution from varied (`read_file`, `grep`, `edit_file`, `run_tests`) to narrow (`read_file`, `write_file`, `bash`) |
| `COMBINED` | Applies both vocabulary and toolcall drift simultaneously |
| `FRAMING` | Subtly reframes text (e.g., "must" → "could potentially") while preserving vocabulary and structure. **Intentionally invisible** to all three instruments — validates no false positives. |

### Key Functions

- `generate_drift_pair(mode, n_samples=5, seed=42) → (pre_records, post_records)`: Generate paired observation sets.
- `apply_vocabulary_drift(texts, replacement_rate=0.7, seed=None)`: Replace technical terms.
- `apply_topic_drift(texts, seed=None)`: Replace with generic filler.
- `apply_toolcall_drift(metadata_list, seed=None)`: Shift tool distributions.
- `apply_framing_drift(texts, seed=None)`: Subtle reframing.
- `validate_instruments(n_trials=10, seed=42) → dict`: Run full validation suite, returning detection rates and mean scores per mode per instrument.

### Technical Terms Used in Simulation

The simulation uses 20 technical terms (e.g., `idempotent`, `memoization`, `backpressure`, `linearizable`, `sharding`, `quorum`, `tombstone`, `compaction`, `vectorization`, `denormalization`, `deadlock`, `livelock`, `semaphore`, `mutex`, `coroutine`, `goroutine`, `serialization`, `marshalling`, `debounce`, `throttle`) and 20 generic replacements (e.g., `process`, `handle`, `manage`, `configure`, `operation`).

## Test Suite

**Source:** `tests/`

Run with: `pytest tests/ -v`

| File | Covers |
|---|---|
| `tests/test_ghost_lexicon.py` | Vocabulary extraction, decay scoring, stop word filtering, boundary behavior |
| `tests/test_behavioral.py` | Tool distribution distance, length shift, fingerprint computation, no-tools fallback |
| `tests/test_semantic.py` | Keyword overlap similarity, embedding fallback, method property |
| `tests/test_scoring.py` | Composite scoring, weight configuration, CompressionType classification for all firing patterns |
| `tests/test_harness.py` | DriftHarness lifecycle, boundary auto-setting, strategy classification, nudge generation, `analyze_tsv` |
| `tests/test_cli.py` | CLI subcommands, JSONL I/O, demo output, validate output |
| `tests/test_simulate.py` | Drift mode generation, drift application functions, validation suite |
| `tests/test_hedging.py` | Hedging increase detection, one-sidedness, framing-qualifier catch, multiword hedge counting |
| `tests/test_significance.py` | Permutation test power on real drift, null behaviour on identical windows, determinism, monitor wrapper |
| `tests/test_tokenizer_unicode.py` | ASCII token-stream identity, non-English tokenization, opt-in token length metric |

### Validation Results

The `validate` command reports detection rates:

| Drift Mode | Detection Rate | Notes |
|---|---|---|
| VOCABULARY | 100% | Specialized term replacement |
| TOPIC | 100% | Complete domain shift |
| TOOLCALL | 100% | Tool distribution change |
| COMBINED | 100% | All modes simultaneously |
| FRAMING | 0% | Intentionally undetectable (by design) |

The 0% on FRAMING validates that instruments don't produce false positives on subtle reframing that preserves vocabulary and tool usage.

## Validation Data

**Source:** `validation/`

The repository includes real validation data from RTX 5090 experiment runs:

- `validation/r1/` — Run 1: `results.tsv` (experiment log), `drift_readings.jsonl` (drift assessments), `full_output.log` (full run output)
- `validation/r3/` — Run 3: `results.tsv`, `drift_readings.jsonl`, `logs_r3.txt`

These are from autoresearch-unified runs where drift-monitor was integrated. The `drift_readings.jsonl` files show drift assessments over the course of ~100-experiment runs.

## Chart Generation

**Source:** `docs/generate_charts.py`

Generates validation charts (PNG) from the validation data in `docs/charts/`:
- `drift_trajectory.png` — Composite drift score over experiment count across runs
- `r1_strategy.png` / `r3_strategy.png` — Strategy distribution over time per run
- `r1_timeline.png` / `r3_timeline.png` — Drift timeline per run

## Examples

**Source:** `examples/`

| File | Purpose |
|---|---|
| `basic_usage.py` | Minimal example: create instruments, feed pre/post, get composite report |
| `jsonl_pipeline.py` | Generate synthetic JSONL and run the CLI on it |
| `autoresearch_adapter.py` | Standalone analysis of autoresearch `results.tsv` files |
| `local_harness_test.py` | Test the DriftHarness locally with simulated experiment results |
| `scan_all_branches.py` | Scan multiple autoresearch run branches for drift patterns |

## Where to Start When Modifying

- **Adding a CLI command:** Add a `cmd_*` function and register it in `main()` in `drift_monitor/cli.py`.
- **Adding a drift mode:** Add to `DriftMode` enum and implement an `apply_*` function + a branch in `generate_drift_pair()` in `drift_monitor/simulate.py`.
- **Adding tests:** Follow the pattern in `tests/` — one test file per component. Tests use no external fixtures; synthetic data is generated inline.
- **Changing storage format:** Edit `drift_monitor/storage.py`. The JSONL format is the canonical interchange format.
