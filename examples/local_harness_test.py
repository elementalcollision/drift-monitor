"""Local test runner for the drift harness.

Simulates an autoresearch experiment loop by replaying a real results.tsv
file through the harness, showing exactly what the orchestrator would see
at each assessment point.

Can also run in "live simulation" mode, printing each experiment as it
would appear during a real run.

Usage:
    # Replay a real TSV
    python examples/local_harness_test.py --tsv /path/to/results.tsv

    # Replay with live simulation (1s delay between experiments)
    python examples/local_harness_test.py --tsv /path/to/results.tsv --live

    # Use synthetic data (no TSV needed)
    python examples/local_harness_test.py --synthetic
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drift_monitor.harness import DriftHarness, DriftConfig, classify_strategy


@dataclass
class FakeResult:
    """Minimal result object matching the ExperimentResultLike protocol."""
    exp: str
    description: str
    status: str
    notes: str


def load_results_from_tsv(path: str) -> list[FakeResult]:
    results = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(FakeResult(
                exp=row.get("exp", ""),
                description=row.get("description", ""),
                status=row.get("status", ""),
                notes=row.get("notes", ""),
            ))
    return results


# ---------------------------------------------------------------------------
# Synthetic experiment generator
# ---------------------------------------------------------------------------

def generate_synthetic_run(n: int = 100, seed: int = 42) -> list[FakeResult]:
    """Generate a synthetic experiment run that exhibits natural drift.

    Simulates the real pattern: broad exploration → LR tunnel → stagnation.
    """
    import random
    rng = random.Random(seed)

    templates = {
        "architecture": [
            ("Decrease DEPTH from {a} to {b}", "Reducing model depth to test shallower architectures"),
            ("Increase ASPECT_RATIO from {a} to {b}", "Wider model may capture more features"),
            ("Change WINDOW_PATTERN from {a} to {b}", "Different attention pattern for sequence modeling"),
            ("Decrease MLP_RATIO from {a} to {b}", "Smaller feedforward blocks reduce overfitting"),
            ("Increase NUM_HEADS from {a} to {b}", "More attention heads for finer-grained patterns"),
        ],
        "learning_rate": [
            ("Decrease MATRIX_LR from {a} to {b}", "Lower matrix learning rate for stability"),
            ("Decrease SCALAR_LR from {a} to {b}", "Fine-tune scalar parameters more carefully"),
            ("Decrease EMBEDDING_LR from {a} to {b}", "Reduce embedding learning rate"),
            ("Decrease UNEMBEDDING_LR from {a} to {b}", "Fine-tune output layer learning"),
        ],
        "schedule": [
            ("Decrease WARMDOWN_RATIO from {a} to {b}", "More gradual learning rate decay"),
            ("Increase FINAL_LR_FRAC from {a} to {b}", "Maintain higher learning rate at end"),
            ("Increase WARMUP_RATIO from {a} to {b}", "Longer warmup for stable early training"),
        ],
        "regularization": [
            ("Decrease WEIGHT_DECAY from {a} to {b}", "Less regularization may help"),
            ("Increase WEIGHT_DECAY from {a} to {b}", "Stronger regularization to prevent overfitting"),
        ],
        "batch_size": [
            ("Decrease DEVICE_BATCH_SIZE by ~{a}%", "More gradient steps per fixed time"),
        ],
    }

    results = []
    # Phase 1 (exp0-24): Broad exploration
    # Phase 2 (exp25-49): Narrowing to LR + schedule
    # Phase 3 (exp50-74): Heavy LR tunnel
    # Phase 4 (exp75-99): Deep LR stagnation

    results.append(FakeResult("exp0", "baseline (no modifications)", "baseline", "depth=8, NVIDIA GeForce RTX 5090"))

    for i in range(1, n):
        # Choose category based on phase
        if i < 25:
            weights = {"architecture": 3, "learning_rate": 3, "schedule": 2, "regularization": 1, "batch_size": 1}
        elif i < 50:
            weights = {"architecture": 1, "learning_rate": 5, "schedule": 2, "regularization": 1, "batch_size": 1}
        elif i < 75:
            weights = {"architecture": 0, "learning_rate": 7, "schedule": 2, "regularization": 1, "batch_size": 0}
        else:
            weights = {"architecture": 0, "learning_rate": 9, "schedule": 1, "regularization": 0, "batch_size": 0}

        categories = []
        for cat, w in weights.items():
            categories.extend([cat] * w)
        category = rng.choice(categories)

        tmpl_desc, tmpl_notes = rng.choice(templates[category])
        a = round(rng.uniform(0.01, 0.5), 3)
        b = round(a * rng.uniform(0.6, 0.95), 3)
        desc = tmpl_desc.format(a=a, b=b)
        notes = tmpl_notes

        # Status distribution shifts over phases
        if i < 25:
            status = rng.choices(["keep", "discard", "crash"], weights=[3, 5, 2])[0]
        elif i < 50:
            status = rng.choices(["keep", "discard", "crash"], weights=[2, 7, 1])[0]
        else:
            status = rng.choices(["keep", "discard", "crash"], weights=[1, 8, 1])[0]

        results.append(FakeResult(f"exp{i}", desc, status, notes))

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_header():
    print(f"{'Exp':<6s} {'Strategy':<16s} {'Status':<8s} {'Description':<60s}")
    print("-" * 95)


def print_experiment(result: FakeResult):
    strategy = classify_strategy(result.description)
    desc = result.description[:58] + ".." if len(result.description) > 60 else result.description
    print(f"{result.exp:<6s} {strategy:<16s} {result.status:<8s} {desc:<60s}")


def print_assessment(harness: DriftHarness, exp_count: int):
    report = harness.last_report
    if not report:
        return

    print(f"\n{'─'*95}")
    print(f"  DRIFT ASSESSMENT @ exp{exp_count}")
    print(f"{'─'*95}")

    for reading in report.readings:
        bar_len = int(reading.score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = "!" if reading.severity.value != "low" else " "
        print(f"  {marker} {reading.instrument:<25s} [{bar}] {reading.score:.3f}  ({reading.severity.value})")

    print(f"  >> composite={report.composite_score:.3f}  type={report.compression_type.value}")

    # Strategy distribution
    dist = harness._strategy_distribution()
    if dist:
        dist_str = "  ".join(f"{k}={v:.0%}" for k, v in sorted(dist.items(), key=lambda x: -x[1]))
        print(f"  strategies: {dist_str}")

    # Nudge
    nudge = harness.get_drift_nudge()
    if nudge:
        # Truncate for display
        nudge_short = nudge.strip()[:200]
        print(f"\n  NUDGE: {nudge_short}...")

    print(f"{'─'*95}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local drift harness test runner")
    parser.add_argument("--tsv", help="Path to results.tsv to replay")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--live", action="store_true", help="Simulate live run with delays")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between experiments in live mode (seconds)")
    parser.add_argument("--anchor", type=int, default=25, help="Anchor window size")
    parser.add_argument("--interval", type=int, default=10, help="Assessment interval")
    parser.add_argument("--json", action="store_true", help="Output JSON assessments")

    args = parser.parse_args()

    if not args.tsv and not args.synthetic:
        print("Error: specify --tsv or --synthetic", file=sys.stderr)
        sys.exit(1)

    # Load experiments
    if args.tsv:
        results = load_results_from_tsv(args.tsv)
        label = Path(args.tsv).stem
    else:
        results = generate_synthetic_run(100, seed=42)
        label = "synthetic"

    print(f"Loaded {len(results)} experiments from {label}\n")

    # Configure harness
    config = DriftConfig(
        anchor_window=args.anchor,
        assessment_interval=args.interval,
        output_filename=f"drift_{label}.jsonl",
        status_filename=f".drift_{label}_status.json",
    )
    harness = DriftHarness(results_dir="/tmp/drift-test", config=config)

    if not args.json:
        print_header()

    assessments = []

    for result in results:
        if not args.json:
            print_experiment(result)

        harness.observe_experiment(result)

        # Check if an assessment just happened
        if (
            harness.last_report
            and harness.experiment_count > 0
            and harness.experiment_count % config.assessment_interval == 0
        ):
            if args.json:
                assessments.append({
                    "experiment_count": harness.experiment_count,
                    "report": harness.last_report.to_dict(),
                    "nudge": harness.get_drift_nudge(),
                })
            else:
                print_assessment(harness, harness.experiment_count)

        if args.live and not args.json:
            time.sleep(args.delay)

    if args.json:
        print(json.dumps(assessments, indent=2))
    else:
        # Final summary
        print(f"\n{'='*95}")
        print(f"  FINAL SUMMARY: {label}")
        print(f"{'='*95}")
        print(f"  Total experiments: {harness.experiment_count}")
        if harness.last_report:
            print(f"  Final composite drift: {harness.last_report.composite_score:.3f}")
            print(f"  Compression type: {harness.last_report.compression_type.value}")
            nudge = harness.get_drift_nudge()
            if nudge:
                print(f"  Active nudge: YES")
            else:
                print(f"  Active nudge: none (within bounds)")

        print(f"  Output: /tmp/drift-test/drift_{label}.jsonl")


if __name__ == "__main__":
    main()
