"""Adapter: analyze autoresearch experiment runs for behavioral drift.

Reads results.tsv files from autoresearch-unified and measures whether
the LLM agent's vocabulary, strategy diversity, and behavioral patterns
narrow over the course of a 100-experiment run.

Usage:
    python examples/autoresearch_adapter.py [--boundary N] [--files FILE ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.scoring import DriftScorer


def load_tsv(path: str | Path) -> list[dict[str, str]]:
    """Load an autoresearch results.tsv file."""
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(dict(row))
    return records


def record_to_observation(record: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """Convert a TSV record to (text, metadata) for drift-monitor.

    Text = description + notes (the agent-generated content).
    Metadata = structured fields that capture behavioral patterns.
    """
    description = record.get("description", "")
    notes = record.get("notes", "")
    text = f"{description}. {notes}" if notes else description

    # Extract the hyperparameter being modified as a "tool" analog
    # The agent's choice of which lever to pull is its behavioral signature
    tools = []
    for param in [
        "LEARNING_RATE", "MATRIX_LR", "SCALAR_LR", "EMBEDDING_LR",
        "UNEMBEDDING_LR", "WEIGHT_DECAY", "WARMUP_RATIO", "WARMDOWN_RATIO",
        "DEVICE_BATCH_SIZE", "ASPECT_RATIO", "MLP_RATIO", "WINDOW_PATTERN",
        "FINAL_LR_FRAC", "NUM_HEADS", "DEPTH",
    ]:
        if param.lower() in description.lower() or param in description:
            tools.append(param)

    # Status as a behavioral signal
    status = record.get("status", "")
    if status:
        tools.append(f"outcome:{status}")

    metadata = {
        "tools": tools,
        "exp": record.get("exp", ""),
        "status": status,
        "val_bpb": record.get("val_bpb", ""),
    }

    return text, metadata


def analyze_run(
    records: list[dict[str, str]],
    boundary: int = 50,
    label: str = "",
) -> dict[str, Any]:
    """Analyze a single run, splitting at the given experiment boundary.

    Compares experiments [0..boundary) vs [boundary..end).
    """
    if boundary >= len(records):
        boundary = len(records) // 2

    # Skip baseline and crash-only records for cleaner signal
    valid_records = [r for r in records if r.get("status") not in ("baseline",)]

    pre_records = valid_records[:boundary]
    post_records = valid_records[boundary:]

    if not pre_records or not post_records:
        return {"error": "Not enough records", "label": label}

    instruments = [
        GhostLexicon(min_freq=2, min_length=4),
        BehavioralFootprint(),
        SemanticDrift(use_embeddings=False),
    ]

    # Feed pre-boundary observations
    for record in pre_records:
        text, meta = record_to_observation(record)
        for instr in instruments:
            instr.observe(text, meta)

    # Mark boundary
    for instr in instruments:
        instr.mark_boundary()

    # Feed post-boundary observations
    for record in post_records:
        text, meta = record_to_observation(record)
        for instr in instruments:
            instr.observe(text, meta)

    # Collect readings
    readings = [instr.read() for instr in instruments]
    scorer = DriftScorer()
    report = scorer.score(readings)

    return {
        "label": label,
        "boundary": f"exp{boundary}",
        "pre_count": len(pre_records),
        "post_count": len(post_records),
        "report": report.to_dict(),
        "lost_terms": readings[0].details.get("lost_terms", []),
    }


def sliding_window_analysis(
    records: list[dict[str, str]],
    window_size: int = 25,
    step: int = 10,
    label: str = "",
) -> list[dict[str, Any]]:
    """Run drift analysis with a sliding window across the run.

    Compares each window against the first window (experiments 0..window_size).
    This shows how drift accumulates over the session.
    """
    valid = [r for r in records if r.get("status") not in ("baseline",)]

    if len(valid) < window_size * 2:
        return [{"error": "Not enough records for sliding analysis"}]

    anchor_records = valid[:window_size]
    results = []

    for start in range(step, len(valid) - window_size + 1, step):
        window_records = valid[start : start + window_size]

        instruments = [
            GhostLexicon(min_freq=2, min_length=3),
            BehavioralFootprint(),
            SemanticDrift(use_embeddings=False),
        ]

        for record in anchor_records:
            text, meta = record_to_observation(record)
            for instr in instruments:
                instr.observe(text, meta)

        for instr in instruments:
            instr.mark_boundary()

        for record in window_records:
            text, meta = record_to_observation(record)
            for instr in instruments:
                instr.observe(text, meta)

        readings = [instr.read() for instr in instruments]
        scorer = DriftScorer()
        report = scorer.score(readings)

        results.append({
            "label": label,
            "anchor": f"exp1-exp{window_size}",
            "window": f"exp{start+1}-exp{start+window_size}",
            "scores": {
                r.instrument: round(r.score, 4) for r in readings
            },
            "composite": round(report.composite_score, 4),
            "type": report.compression_type.value,
            "lost_terms": readings[0].details.get("lost_terms", [])[:10],
        })

    return results


def print_fixed_boundary(result: dict[str, Any]) -> None:
    """Pretty-print a fixed-boundary analysis result."""
    report = result["report"]
    print(f"\n{'='*60}")
    print(f"  {result['label']}")
    print(f"  Split at {result['boundary']}  "
          f"(pre={result['pre_count']}, post={result['post_count']})")
    print(f"{'='*60}")

    for reading in report["readings"]:
        marker = "*" if reading["severity"] != "low" else " "
        print(
            f"  {marker} {reading['instrument']:25s} "
            f"score={reading['score']:.4f}  "
            f"severity={reading['severity']}"
        )

    print(
        f"  >> composite={report['composite_score']:.4f}  "
        f"type={report['compression_type']}  "
        f"severity={report['severity']}"
    )

    if result.get("lost_terms"):
        print(f"  Lost terms: {', '.join(result['lost_terms'][:10])}")


def print_sliding(results: list[dict[str, Any]]) -> None:
    """Pretty-print sliding window results as a drift timeline."""
    if not results or "error" in results[0]:
        print("  Not enough data for sliding analysis.")
        return

    label = results[0].get("label", "")
    print(f"\n{'='*70}")
    print(f"  Sliding Window: {label}")
    print(f"  Anchor: {results[0]['anchor']}")
    print(f"{'='*70}")
    print(f"  {'Window':<18s} {'Ghost':>8s} {'Behav':>8s} {'Semantic':>8s} {'Composite':>10s}  Type")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*10}  {'-'*16}")

    for r in results:
        scores = r["scores"]
        gl = scores.get("ghost_lexicon", 0)
        bf = scores.get("behavioral_footprint", 0)
        sd = scores.get("semantic_drift", 0)
        print(
            f"  {r['window']:<18s} "
            f"{gl:>8.4f} {bf:>8.4f} {sd:>8.4f} "
            f"{r['composite']:>10.4f}  {r['type']}"
        )

    # Show vocabulary loss progression
    print(f"\n  Vocabulary loss over time:")
    for r in results:
        if r.get("lost_terms"):
            print(f"    {r['window']}: -{len(r['lost_terms'])} terms: {', '.join(r['lost_terms'][:5])}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze autoresearch experiment runs for behavioral drift"
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="Path(s) to results.tsv files"
    )
    parser.add_argument(
        "--boundary", type=int, default=50,
        help="Experiment number to split at (default: 50)"
    )
    parser.add_argument(
        "--sliding", action="store_true",
        help="Run sliding window analysis"
    )
    parser.add_argument(
        "--window-size", type=int, default=25,
        help="Sliding window size (default: 25)"
    )
    parser.add_argument(
        "--step", type=int, default=10,
        help="Sliding window step (default: 10)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted text"
    )

    args = parser.parse_args()

    all_results = []

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: {filepath} not found, skipping", file=sys.stderr)
            continue

        records = load_tsv(path)
        label = path.stem

        if args.sliding:
            results = sliding_window_analysis(
                records,
                window_size=args.window_size,
                step=args.step,
                label=label,
            )
            if args.json:
                all_results.extend(results)
            else:
                print_sliding(results)
        else:
            result = analyze_run(records, boundary=args.boundary, label=label)
            if args.json:
                all_results.append(result)
            else:
                print_fixed_boundary(result)

    if args.json:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
