# drift-monitor — OpenWiki Quickstart

**Behavioral drift detection for AI agents across context compression boundaries.**

When an AI agent's context fills up and gets compressed (summarized, truncated, or window-shifted), it can silently lose specialized vocabulary, shift tool-call preferences, or narrow its conceptual framing. drift-monitor detects these changes using three complementary instruments and provides actionable nudges to course-correct.

- **Language:** Python 3.9+
- **Dependencies:** Zero required. Optional `sentence-transformers` for embedding-based semantic drift.
- **License:** MIT
- **Methodology:** Inspired by [morrow.run's compression-monitor](https://morrow.run) (v0.2.1, MIT). Independent implementation — no code copied.

## Installation

```bash
# From GitHub
pip install git+https://github.com/elementalcollision/drift-monitor.git

# With optional embedding support
pip install "drift-monitor[embeddings] @ git+https://github.com/elementalcollision/drift-monitor.git"

# Development
git clone https://github.com/elementalcollision/drift-monitor.git
cd drift-monitor
pip install -e ".[dev]"
pytest tests/ -v  # 60 tests
```

## Quick Usage

```python
from drift_monitor import GhostLexicon, BehavioralFootprint, SemanticDrift, DriftScorer

scorer = DriftScorer()

# Feed pre-compression observations (anchor window)
for text in pre_compression_outputs:
    scorer.observe(text, metadata={"tools_used": ["search", "code"]})

scorer.mark_boundary()  # Context was compressed here

# Feed post-compression observations (recent window)
for text in post_compression_outputs:
    scorer.observe(text, metadata={"tools_used": ["search"]})

report = scorer.score()
print(f"Drift: {report.composite:.3f} ({report.compression_type.name})")
```

For live experiment-loop monitoring with nudges:

```python
from drift_monitor.harness import DriftHarness, DriftConfig

config = DriftConfig(anchor_window=25, assessment_interval=10)
harness = DriftHarness(results_dir="./results", config=config)

for experiment in run_experiments():
    harness.observe_experiment(experiment)
    nudge = harness.get_drift_nudge()
    if nudge:
        prompt += nudge  # Inject into LLM prompt
```

## CLI

```bash
# Run all instruments on pre/post JSONL files
drift-monitor run --pre anchor.jsonl --post recent.jsonl

# Individual instruments
drift-monitor ghost-lexicon --pre anchor.jsonl --post recent.jsonl
drift-monitor behavioral --pre anchor.jsonl --post recent.jsonl
drift-monitor semantic --pre anchor.jsonl --post recent.jsonl

# Interactive demo with synthetic data
drift-monitor demo

# Validate detection rates against synthetic drift
drift-monitor validate --trials 50
```

## How It Works

The system compares an **anchor window** (pre-compression baseline) against a **recent window** (post-compression activity) using three instruments:

| Instrument | What It Detects | Source |
|---|---|---|
| **Ghost Lexicon** | Loss of low-frequency, high-precision vocabulary | `drift_monitor/instruments/ghost_lexicon.py` |
| **Behavioral Footprint** | Shifts in tool-call ratios and response shape | `drift_monitor/instruments/behavioral.py` |
| **Semantic Drift** | Movement of conceptual center-of-gravity (keyword overlap or embeddings) | `drift_monitor/instruments/semantic.py` |

A **DriftScorer** combines readings into a composite score (0.0–1.0) and classifies the compression event type based on which instruments fired. See [Architecture](architecture.md) for the full data flow.

## Compression Type Classification

| Type | Instruments Fired | Meaning |
|---|---|---|
| `NONE` | None | No drift detected |
| `VOCABULARY_ONLY` | Ghost Lexicon only | Surface compression, behavior intact |
| `OPERATIONAL` | Ghost Lexicon + Behavioral | Systemic context loss |
| `SEMANTIC_ONLY` | Semantic only | Topic changed, vocabulary intact |
| `INFRASTRUCTURE` | Behavioral only | Tool preferences shifted |
| `FULL_BOUNDARY` | All three | Major compression boundary crossed |

## Related Projects

- **[autoresearch-unified](https://github.com/elementalcollision/autoresearch-unified)** — Autonomous LLM-driven GPU pretraining research. drift-monitor integrates as an optional enhancement ([PR #53](https://github.com/elementalcollision/autoresearch-unified/pull/53)) that replaces the built-in stagnation heuristic.
- **[Benchwright](https://benchwright.polsia.app/)** — Production LLM monitoring and benchmarking. drift-monitor covers the experiment/research level; Benchwright covers production deployment level.

## Documentation Sections

- [Architecture](architecture.md) — DualWindow, Instrument base class, DriftScorer, CompressionType classification, data flow.
- [Instruments](instruments.md) — Detailed explanation of each of the three drift detection instruments, their algorithms, thresholds, and configuration.
- [Harness & Live Monitoring](harness.md) — DriftHarness for experiment loops, DriftConfig, nudge system, strategy classification, and TSV analysis.
- [CLI & Testing](cli-and-testing.md) — CLI commands, JSONL storage layer, synthetic drift simulation, test suite, and validation data.
