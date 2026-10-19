---
okf_version: "0.2"
---

# Files

- [Architecture](architecture.md) - Core architecture of drift-monitor — the dual-window system, instrument base class, weighted composite scoring, and compression-type classification from instrument firing patterns.
- [CLI & Testing](cli-and-testing.md) - The drift-monitor CLI subcommands, JSONL storage layer, synthetic drift simulation, the test suite, and the bundled validation data from RTX 5090 runs.
- [Harness & Live Monitoring](harness.md) - DriftHarness for live experiment loops — DriftConfig, the nudge system, edge-triggered alert hooks, strategy classification, TSV analysis, and autoresearch-unified integration.
- [Instruments](instruments.md) - The four drift instruments — Ghost Lexicon, Behavioral Footprint, Semantic Drift, and opt-in Hedging Language — their algorithms, thresholds, configuration, and the permutation-test significance layer.
- [Multi-Segment History](multi-segment.md) - SegmentHistory — a purely-additive layer over DualWindow that tracks drift across many compression boundaries and exposes the per-boundary trend.
- [drift-monitor — OpenWiki Quickstart](quickstart.md) - Quickstart for drift-monitor — install, the DriftMonitor facade, the live DriftHarness loop, the CLI, the three instruments, and compression-type classification.
- [State Persistence](state-persistence.md) - Snapshot + append-only event-tail persistence for DriftMonitor and DriftHarness — the to_state/from_state API, schema versioning, migration registry, and crash recovery.
