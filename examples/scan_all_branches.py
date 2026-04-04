"""Scan all branches of autoresearch-unified for results.tsv files
and run drift analysis across every run found.

Usage:
    python examples/scan_all_branches.py --repo /path/to/autoresearch-unified
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drift_monitor.instruments.ghost_lexicon import GhostLexicon
from drift_monitor.instruments.behavioral import BehavioralFootprint
from drift_monitor.instruments.semantic import SemanticDrift
from drift_monitor.scoring import DriftScorer


@dataclass
class BranchRun:
    branch: str
    tsv_path: str  # path within repo
    gpu: str
    model: str
    dataset: str
    n_experiments: int
    records: list[dict[str, str]]


def git_cmd(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip()


def list_branches(repo: Path) -> list[str]:
    output = git_cmd(repo, "branch", "-r", "--list", "origin/*")
    branches = []
    for line in output.splitlines():
        b = line.strip()
        if "->" in b:
            continue
        branches.append(b)
    return branches


def find_tsvs_in_branch(repo: Path, branch: str) -> list[str]:
    """List all results*.tsv files in a branch."""
    output = git_cmd(repo, "ls-tree", "-r", "--name-only", branch)
    return [
        f for f in output.splitlines()
        if f.endswith(".tsv") and "results" in f.lower()
    ]


def read_tsv_from_branch(repo: Path, branch: str, filepath: str) -> str:
    """Read file contents from a specific branch without checkout."""
    return git_cmd(repo, "show", f"{branch}:{filepath}")


def parse_tsv(content: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    return [dict(row) for row in reader]


def infer_metadata(branch: str, tsv_path: str, records: list[dict[str, str]]) -> dict[str, str]:
    """Infer GPU, model, and dataset from branch name and record content."""
    branch_lower = branch.lower()

    # GPU
    gpu = "unknown"
    gpu_map = {
        "5090": "RTX-5090", "5070ti": "RTX-5070Ti", "5070-ti": "RTX-5070Ti",
        "mi300x": "MI300X", "rtxpro6000": "RTX-Pro-6000", "rtx-pro-6000": "RTX-Pro-6000",
        "4000ada": "RTX-4000-Ada", "m5": "M5-Mac",
    }
    for key, val in gpu_map.items():
        if key in branch_lower or key in tsv_path.lower():
            gpu = val
            break
    # Fallback: check records
    if gpu == "unknown" and records:
        gpu_name = records[0].get("gpu_name", "")
        if gpu_name:
            gpu = gpu_name.split("NVIDIA ")[-1].strip() if "NVIDIA" in gpu_name else gpu_name

    # Model
    model = "unknown"
    model_map = {
        "gpt41": "GPT-4.1", "gpt-41": "GPT-4.1",
        "sonnet": "Sonnet-4.6", "claude": "Claude",
        "qwen": "Qwen-3.5", "gemma": "Gemma-4",
        "kimi": "Kimi-K2.5",
    }
    for key, val in model_map.items():
        if key in branch_lower or key in tsv_path.lower():
            model = val
            break

    # Dataset
    dataset = "unknown"
    dataset_map = {
        "pubmed": "pubmed", "climbmix": "climbmix",
        "fineweb-edu-high": "fineweb-edu-high",
        "fineweb-edu": "fineweb-edu", "fineweb": "fineweb",
        "cosmopedia": "cosmopedia-v2", "slimpajama": "slimpajama",
        "github-code": "github-code-python",
    }
    for key, val in dataset_map.items():
        if key in branch_lower or key in tsv_path.lower():
            dataset = val
            break

    return {"gpu": gpu, "model": model, "dataset": dataset}


def record_to_observation(record: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """Same logic as autoresearch_adapter.py."""
    description = record.get("description", "")
    notes = record.get("notes", "")
    text = f"{description}. {notes}" if notes else description

    tools = []
    for param in [
        "LEARNING_RATE", "MATRIX_LR", "SCALAR_LR", "EMBEDDING_LR",
        "UNEMBEDDING_LR", "WEIGHT_DECAY", "WARMUP_RATIO", "WARMDOWN_RATIO",
        "DEVICE_BATCH_SIZE", "ASPECT_RATIO", "MLP_RATIO", "WINDOW_PATTERN",
        "FINAL_LR_FRAC", "NUM_HEADS", "DEPTH",
    ]:
        if param.lower() in description.lower() or param in description:
            tools.append(param)

    status = record.get("status", "")
    if status:
        tools.append(f"outcome:{status}")

    return text, {"tools": tools, "status": status}


def analyze_run_drift(records: list[dict[str, str]], n_windows: int = 4) -> dict[str, Any]:
    """Analyze drift across evenly-spaced windows in a run."""
    valid = [r for r in records if r.get("status") not in ("baseline",)]
    if len(valid) < 10:
        return {"error": "too few records", "n_valid": len(valid)}

    window_size = max(5, len(valid) // n_windows)
    anchor = valid[:window_size]

    timeline = []
    for i in range(1, n_windows):
        start = i * window_size
        end = min(start + window_size, len(valid))
        if start >= len(valid):
            break
        window = valid[start:end]

        instruments = [
            GhostLexicon(min_freq=2, min_length=3),
            BehavioralFootprint(),
            SemanticDrift(use_embeddings=False),
        ]

        for r in anchor:
            text, meta = record_to_observation(r)
            for instr in instruments:
                instr.observe(text, meta)
        for instr in instruments:
            instr.mark_boundary()
        for r in window:
            text, meta = record_to_observation(r)
            for instr in instruments:
                instr.observe(text, meta)

        readings = [instr.read() for instr in instruments]
        scorer = DriftScorer()
        report = scorer.score(readings)

        timeline.append({
            "quarter": f"Q{i+1}",
            "window": f"exp{start+1}-exp{end}",
            "ghost": round(readings[0].score, 4),
            "behavioral": round(readings[1].score, 4),
            "semantic": round(readings[2].score, 4),
            "composite": round(report.composite_score, 4),
            "type": report.compression_type.value,
            "lost_count": readings[0].details.get("lost_count", 0),
        })

    # Compute overall drift trajectory (Q2 vs last Q)
    if len(timeline) >= 2:
        drift_acceleration = timeline[-1]["composite"] - timeline[0]["composite"]
    else:
        drift_acceleration = 0.0

    return {
        "n_experiments": len(valid),
        "timeline": timeline,
        "final_composite": timeline[-1]["composite"] if timeline else 0.0,
        "drift_acceleration": round(drift_acceleration, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Scan all branches for drift analysis")
    parser.add_argument("--repo", required=True, help="Path to autoresearch-unified repo")
    parser.add_argument("--json", action="store_true", help="Output JSON for further analysis")
    args = parser.parse_args()

    repo = Path(args.repo)
    branches = list_branches(repo)

    print(f"Found {len(branches)} branches, scanning for results.tsv files...\n",
          file=sys.stderr)

    all_runs: list[dict[str, Any]] = []
    skipped = 0

    for branch in sorted(branches):
        tsvs = find_tsvs_in_branch(repo, branch)
        if not tsvs:
            skipped += 1
            continue

        for tsv_path in tsvs:
            content = read_tsv_from_branch(repo, branch, tsv_path)
            if not content or "description" not in content.split("\n")[0]:
                continue

            records = parse_tsv(content)
            if len(records) < 10:
                continue

            meta = infer_metadata(branch, tsv_path, records)
            analysis = analyze_run_drift(records)

            if "error" in analysis:
                continue

            run_info = {
                "branch": branch.replace("origin/", ""),
                "tsv": tsv_path,
                "gpu": meta["gpu"],
                "model": meta["model"],
                "dataset": meta["dataset"],
                **analysis,
            }
            all_runs.append(run_info)

    # Sort by final composite drift (most drift first)
    all_runs.sort(key=lambda r: r.get("final_composite", 0), reverse=True)

    if args.json:
        print(json.dumps(all_runs, indent=2))
        return

    # Pretty print summary table
    print(f"{'Branch':<52s} {'GPU':<16s} {'Model':<10s} {'Dataset':<18s} "
          f"{'N':>4s} {'Q2':>6s} {'Q3':>6s} {'Q4':>6s} {'Final':>6s} {'Accel':>7s}")
    print("-" * 145)

    for run in all_runs:
        tl = run.get("timeline", [])
        q2 = tl[0]["composite"] if len(tl) > 0 else 0
        q3 = tl[1]["composite"] if len(tl) > 1 else 0
        q4 = tl[2]["composite"] if len(tl) > 2 else 0

        print(
            f"{run['branch']:<52s} "
            f"{run['gpu']:<16s} "
            f"{run['model']:<10s} "
            f"{run['dataset']:<18s} "
            f"{run['n_experiments']:>4d} "
            f"{q2:>6.3f} "
            f"{q3:>6.3f} "
            f"{q4:>6.3f} "
            f"{run['final_composite']:>6.3f} "
            f"{run['drift_acceleration']:>+7.3f}"
        )

    # Summary stats
    print(f"\n{'='*145}")
    print(f"Total runs analyzed: {len(all_runs)}  |  Branches skipped (no TSV): {skipped}")

    if all_runs:
        composites = [r["final_composite"] for r in all_runs]
        print(f"Final composite drift — min: {min(composites):.3f}  "
              f"mean: {sum(composites)/len(composites):.3f}  "
              f"max: {max(composites):.3f}")

        accels = [r["drift_acceleration"] for r in all_runs]
        print(f"Drift acceleration  — min: {min(accels):+.3f}  "
              f"mean: {sum(accels)/len(accels):+.3f}  "
              f"max: {max(accels):+.3f}")

        # Breakdown by GPU
        gpus = sorted(set(r["gpu"] for r in all_runs))
        print(f"\nBy GPU:")
        for gpu in gpus:
            gpu_runs = [r for r in all_runs if r["gpu"] == gpu]
            avg = sum(r["final_composite"] for r in gpu_runs) / len(gpu_runs)
            print(f"  {gpu:<20s} n={len(gpu_runs):>3d}  avg_drift={avg:.3f}")

        # Breakdown by model
        models = sorted(set(r["model"] for r in all_runs))
        print(f"\nBy Model:")
        for model in models:
            model_runs = [r for r in all_runs if r["model"] == model]
            avg = sum(r["final_composite"] for r in model_runs) / len(model_runs)
            print(f"  {model:<20s} n={len(model_runs):>3d}  avg_drift={avg:.3f}")

        # Breakdown by dataset
        datasets = sorted(set(r["dataset"] for r in all_runs))
        print(f"\nBy Dataset:")
        for ds in datasets:
            ds_runs = [r for r in all_runs if r["dataset"] == ds]
            avg = sum(r["final_composite"] for r in ds_runs) / len(ds_runs)
            print(f"  {ds:<20s} n={len(ds_runs):>3d}  avg_drift={avg:.3f}")


if __name__ == "__main__":
    main()
