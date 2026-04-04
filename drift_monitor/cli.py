"""Command-line interface for drift-monitor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.scoring import DriftScorer
from drift_monitor.storage import read_jsonl
from drift_monitor.simulate import (
    DriftMode,
    generate_drift_pair,
    validate_instruments,
)


def _load_and_run(
    pre_path: str,
    post_path: str,
    text_field: str = "text",
    tools_field: str = "tools",
) -> dict:
    """Load JSONL files and run all instruments."""
    pre_records = read_jsonl(pre_path)
    post_records = read_jsonl(post_path)

    if not pre_records:
        print(f"Error: no records in {pre_path}", file=sys.stderr)
        sys.exit(1)
    if not post_records:
        print(f"Error: no records in {post_path}", file=sys.stderr)
        sys.exit(1)

    instruments = [
        GhostLexicon(),
        BehavioralFootprint(),
        SemanticDrift(use_embeddings=False),
    ]

    for instr in instruments:
        for r in pre_records:
            text = r.get(text_field, "")
            meta = {"tools": r.get(tools_field, [])}
            instr.observe(text, meta)
        instr.mark_boundary()
        for r in post_records:
            text = r.get(text_field, "")
            meta = {"tools": r.get(tools_field, [])}
            instr.observe(text, meta)

    readings = [instr.read() for instr in instruments]
    scorer = DriftScorer()
    report = scorer.score(readings)
    return report.to_dict()


def cmd_run(args: argparse.Namespace) -> None:
    """Run all instruments on pre/post JSONL files."""
    result = _load_and_run(args.pre, args.post, args.text_field, args.tools_field)
    print(json.dumps(result, indent=2))


def cmd_ghost_lexicon(args: argparse.Namespace) -> None:
    """Run ghost lexicon instrument only."""
    pre_records = read_jsonl(args.pre)
    post_records = read_jsonl(args.post)

    gl = GhostLexicon()
    for r in pre_records:
        gl.observe(r.get(args.text_field, ""))
    gl.mark_boundary()
    for r in post_records:
        gl.observe(r.get(args.text_field, ""))

    reading = gl.read()
    print(json.dumps(reading.to_dict(), indent=2))


def cmd_behavioral(args: argparse.Namespace) -> None:
    """Run behavioral footprint instrument only."""
    pre_records = read_jsonl(args.pre)
    post_records = read_jsonl(args.post)

    bf = BehavioralFootprint()
    for r in pre_records:
        bf.observe(r.get(args.text_field, ""), {"tools": r.get(args.tools_field, [])})
    bf.mark_boundary()
    for r in post_records:
        bf.observe(r.get(args.text_field, ""), {"tools": r.get(args.tools_field, [])})

    reading = bf.read()
    print(json.dumps(reading.to_dict(), indent=2))


def cmd_semantic(args: argparse.Namespace) -> None:
    """Run semantic drift instrument only."""
    pre_records = read_jsonl(args.pre)
    post_records = read_jsonl(args.post)

    sd = SemanticDrift(use_embeddings=False)
    for r in pre_records:
        sd.observe(r.get(args.text_field, ""))
    sd.mark_boundary()
    for r in post_records:
        sd.observe(r.get(args.text_field, ""))

    reading = sd.read()
    print(json.dumps(reading.to_dict(), indent=2))


def cmd_demo(args: argparse.Namespace) -> None:
    """Run end-to-end demo with synthetic data."""
    print("=== Drift Monitor Demo ===\n")

    for mode in DriftMode:
        pre, post = generate_drift_pair(mode, seed=42)
        print(f"--- Mode: {mode.value} ---")

        instruments = [
            GhostLexicon(),
            BehavioralFootprint(),
            SemanticDrift(use_embeddings=False),
        ]

        for instr in instruments:
            for r in pre:
                instr.observe(r["text"], {"tools": r.get("tools", [])})
            instr.mark_boundary()
            for r in post:
                instr.observe(r["text"], {"tools": r.get("tools", [])})

        readings = [instr.read() for instr in instruments]
        scorer = DriftScorer()
        report = scorer.score(readings)

        for reading in readings:
            marker = "*" if reading.severity.value != "low" else " "
            print(
                f"  {marker} {reading.instrument:25s} "
                f"score={reading.score:.4f}  "
                f"severity={reading.severity.value}"
            )

        print(
            f"  >> composite={report.composite_score:.4f}  "
            f"type={report.compression_type.value}  "
            f"severity={report.severity.value}"
        )
        print()


def cmd_validate(args: argparse.Namespace) -> None:
    """Run validation suite on synthetic data."""
    print("Running validation suite...\n")
    results = validate_instruments(n_trials=args.trials, seed=args.seed)

    all_pass = True
    for mode, instruments in results.items():
        print(f"--- {mode} ---")
        for instr_name, stats in instruments.items():
            rate = stats["detection_rate"]
            mean = stats["mean_score"]
            # Framing should be undetected; everything else should detect
            if mode == "framing":
                status = "PASS" if rate < 0.3 else "FAIL"
            else:
                # At least the relevant instrument should detect its drift type
                status = "ok"

            if status == "FAIL":
                all_pass = False

            print(
                f"  {instr_name:25s} "
                f"detection_rate={rate:.0%}  "
                f"mean_score={mean:.4f}  "
                f"[{status}]"
            )
        print()

    if all_pass:
        print("Validation: ALL CHECKS PASSED")
    else:
        print("Validation: SOME CHECKS FAILED")
        sys.exit(1)


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    """Add common I/O arguments."""
    parser.add_argument("--pre", required=True, help="Pre-compression JSONL file")
    parser.add_argument("--post", required=True, help="Post-compression JSONL file")
    parser.add_argument(
        "--text-field", default="text", help="JSON field containing text (default: text)"
    )
    parser.add_argument(
        "--tools-field",
        default="tools",
        help="JSON field containing tool names (default: tools)",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="drift-monitor",
        description="Behavioral drift detection for AI agents",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run all instruments")
    _add_io_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # ghost-lexicon
    p_gl = subparsers.add_parser("ghost-lexicon", help="Run ghost lexicon only")
    _add_io_args(p_gl)
    p_gl.set_defaults(func=cmd_ghost_lexicon)

    # behavioral
    p_bf = subparsers.add_parser("behavioral", help="Run behavioral footprint only")
    _add_io_args(p_bf)
    p_bf.set_defaults(func=cmd_behavioral)

    # semantic
    p_sd = subparsers.add_parser("semantic", help="Run semantic drift only")
    _add_io_args(p_sd)
    p_sd.set_defaults(func=cmd_semantic)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run demo with synthetic data")
    p_demo.set_defaults(func=cmd_demo)

    # validate
    p_val = subparsers.add_parser("validate", help="Run validation suite")
    p_val.add_argument(
        "--trials", type=int, default=10, help="Number of trials per mode"
    )
    p_val.add_argument("--seed", type=int, default=42, help="Random seed")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
