# Harness & Live Monitoring

The `DriftHarness` provides plug-and-play drift monitoring for live experiment loops — particularly autonomous hyperparameter optimization runs like [autoresearch-unified](https://github.com/elementalcollision/autoresearch-unified). It wraps the three instruments, manages window boundaries automatically, writes drift readings to JSONL, and generates actionable nudge messages when drift thresholds are exceeded.

**Source:** `drift_monitor/harness.py`

## DriftConfig

```python
@dataclass
class DriftConfig:
    anchor_window: int = 25          # First N experiments form the baseline
    assessment_interval: int = 10    # Assess drift every N experiments
    ghost_nudge_threshold: float = 0.35
    behavioral_nudge_threshold: float = 0.30
    composite_nudge_threshold: float = 0.30
    output_filename: str = "drift_readings.jsonl"
    status_filename: str = ".drift_status.json"
    weights: dict[str, float] = field(default_factory=lambda: {
        "ghost_lexicon": 0.35,
        "behavioral_footprint": 0.35,
        "semantic_drift": 0.30,
    })
```

## DriftHarness API

### `observe_experiment(result: ExperimentResultLike)`

Feed a completed experiment to the drift instruments. Call this after each experiment completes.

- Skips experiments with `status == "baseline"`.
- Increments the experiment counter.
- **Auto-sets the boundary** after `anchor_window` experiments have been observed. All three instruments get `mark_boundary()` called simultaneously.
- Builds observation text from `result.description` + `result.notes`.
- Classifies the experiment's strategy category and appends to strategy history.
- Runs a full assessment every `assessment_interval` experiments (after boundary is set).

The `ExperimentResultLike` protocol requires: `exp: str`, `description: str`, `status: str`, `notes: str`.

### `get_drift_nudge() → str | None`

Returns a nudge message if drift thresholds are exceeded, or `None` if drift is within bounds. Call this from the orchestrator's prompt-building step.

Nudge priority (first match wins):

1. **Composite nudge** — fires when `composite_score >= composite_nudge_threshold` (0.30). Includes lost vocabulary terms, dominant strategy category, and a specific suggestion to break out of the current optimization basin.
2. **Vocabulary nudge** — fires when ghost lexicon score ≥ `ghost_nudge_threshold` (0.35). Lists lost terms and suggests alternative strategy categories.
3. **Behavioral nudge** — fires when behavioral score ≥ `behavioral_nudge_threshold` (0.30). Reports the dominant strategy percentage and suggests alternative levers.

### `wrap_callbacks(callbacks) → callbacks`

Wraps an autoresearch `OrchestratorCallbacks` object to intercept `on_experiment_complete`. The wrapped callbacks feed each experiment to the harness before delegating to the original handler.

```python
harness = DriftHarness(results_dir=".")
wrapped = harness.wrap(original_callbacks)
# Pass wrapped to ExperimentOrchestrator
```

### Properties

- `last_report: DriftReport | None` — most recent assessment report.
- `experiment_count: int` — total experiments observed (excluding baseline).
- `boundary_set: bool` — whether the anchor window is full and boundary has been marked.

## Strategy Classification

**Function:** `classify_strategy(description: str) → str`

Classifies an experiment description into one of seven strategy categories by keyword matching:

| Category | Pattern Keywords |
|---|---|
| `batch_size` | `batch_size`, `total_batch` |
| `architecture` | `depth`, `head_dim`, `window_pattern`, `mlp_ratio`, `aspect_ratio`, `num_heads` |
| `schedule` | `warmup`, `warmdown`, `final_lr_frac`, `schedule`, `cooldown` |
| `regularization` | `weight_decay`, `adam_beta`, `regularization` |
| `learning_rate` | `_lr`, `learning_rate`, `matrix_lr`, `scalar_lr`, `embedding_lr`, `unembedding_lr` |
| `infrastructure` | `activation_checkpointing`, `compile_mode` |
| `other` | (fallback) |

Strategy categories are used as "tools" in the behavioral footprint — the agent's choice of which hyperparameter lever to pull is its behavioral signature. This means behavioral drift in the harness context detects when the agent gets stuck in a strategy tunnel (e.g., only tweaking learning rates).

## Nudge Suggestion Map

Each strategy category maps to a concrete suggestion for what to try instead:

| Dominant Category | Suggestion |
|---|---|
| `learning_rate` | Architectural changes (DEPTH, WINDOW_PATTERN, MLP_RATIO, ASPECT_RATIO) |
| `architecture` | Schedule shape changes (WARMUP_RATIO, FINAL_LR_FRAC, WARMDOWN_RATIO) |
| `schedule` | Regularization tuning (WEIGHT_DECAY, ADAM_BETAS) |
| `regularization` | Learning rate exploration across parameter groups |
| `batch_size` | Architectural or schedule experiments |
| `other` | Systematic learning rate or architecture exploration |

## Output Files

### drift_readings.jsonl

Appended after each assessment. Each line is a JSON object:

```json
{
  "experiment_count": 30,
  "timestamp": 1714269106.0,
  "composite_score": 0.467,
  "compression_type": "full_boundary",
  "severity": "high",
  "readings": [...],
  "details": {...},
  "strategy_distribution": {"learning_rate": 0.6, "regularization": 0.4}
}
```

### .drift_status.json

Overwritten after each assessment with current status:

```json
{
  "experiment_count": 5,
  "composite_score": 0.8211,
  "compression_type": "full_boundary",
  "severity": "high",
  "nudge_active": true,
  "strategy_distribution": {"learning_rate": 1.0},
  "updated": "2026-04-04 10:44:58"
}
```

This file is written atomically (temp-file-then-rename) and failures are silently ignored (non-critical).

## TSV Analysis (Standalone)

**Function:** `analyze_tsv(tsv_path: str | Path, config: DriftConfig | None = None) → list[dict]`

Runs drift analysis on an existing autoresearch `results.tsv` file without needing a live experiment loop. Creates a `DriftHarness` internally, reads the TSV, and feeds each row as a `FakeResult` dataclass.

This is useful for post-hoc analysis of completed runs. See `examples/autoresearch_adapter.py` for a more detailed standalone analysis script.

## Integration with autoresearch-unified

drift-monitor integrates as an **optional enhancement** ([PR #53](https://github.com/elementalcollision/autoresearch-unified/pull/53)):

```bash
pip install -e ".[drift]"  # In autoresearch-unified repo
```

When installed, it replaces the built-in stagnation heuristic with multi-instrument drift analysis. The harness intercepts experiment completion events, feeds them to drift instruments, and injects nudge messages into the LLM prompt when drift thresholds are exceeded.

Design principles:
- **Zero modifications** to autoresearch-unified source code required.
- Attaches by wrapping existing callbacks at construction time via `wrap_callbacks()`.
- Operates on the experiment's `description` + `notes` (LLM-generated text) and the strategy category (which hyperparameter lever was pulled).

## Where to Start When Modifying

- **Changing nudge thresholds:** Edit `DriftConfig` defaults or pass a custom config.
- **Adding new nudge types:** Add templates in the `_NUDGE_*` section and logic in `get_drift_nudge()`.
- **Changing strategy classification:** Edit `_STRATEGY_PATTERNS` in `drift_monitor/harness.py`.
- **Changing assessment cadence:** Edit `DriftConfig.assessment_interval` or `anchor_window`.
- **Adding new output formats:** Extend `_run_assessment()` which writes both JSONL and status files.
