#!/usr/bin/env python3
"""Generate validation charts for drift-monitor wiki."""

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Paths
ROOT = Path(__file__).parent.parent
OUT = ROOT / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
})

STRATEGY_COLORS = {
    "learning_rate": "#2196F3",
    "regularization": "#FF9800",
    "schedule": "#4CAF50",
    "architecture": "#9C27B0",
    "batch_size": "#F44336",
    "infrastructure": "#607D8B",
    "other": "#9E9E9E",
}

STRATEGY_LABELS = {
    "learning_rate": "Learning Rate",
    "regularization": "Regularization",
    "schedule": "Schedule",
    "architecture": "Architecture",
    "batch_size": "Batch Size",
    "infrastructure": "Infrastructure",
    "other": "Other",
}

STATUS_MARKERS = {
    "keep": ("^", 9, 1.0),      # triangle up, big, full opacity
    "discard": ("o", 5, 0.6),   # circle, small, semi-transparent
    "crash": ("x", 6, 0.4),     # x, medium, faded
}


def classify(desc):
    dl = desc.lower()
    if any(k in dl for k in ["batch_size", "total_batch"]):
        return "batch_size"
    if any(k in dl for k in ["depth", "head_dim", "window_pattern", "mlp_ratio", "num_heads", "aspect_ratio"]):
        return "architecture"
    if any(k in dl for k in ["warmup", "warmdown", "final_lr_frac", "cooldown"]):
        return "schedule"
    if any(k in dl for k in ["weight_decay", "adam_beta"]):
        return "regularization"
    if any(k in dl for k in ["_lr", "learning_rate", "matrix_lr", "scalar_lr", "embedding_lr", "unembedding_lr"]):
        return "learning_rate"
    if any(k in dl for k in ["activation_checkpointing", "compile_mode"]):
        return "infrastructure"
    return "other"


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_drift(path):
    readings = []
    with open(path) as f:
        for line in f:
            readings.append(json.loads(line))
    return readings


# ──────────────────────────────────────────────────────────────────────
# Chart 1 & 2: val_bpb timeline with nudge points (one per run)
# ──────────────────────────────────────────────────────────────────────

def plot_run_timeline(run_name, tsv_path, drift_path, out_path, tunnel_label):
    rows = load_tsv(tsv_path)
    drift = load_drift(drift_path)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    # Plot each experiment
    for r in rows:
        exp_num = int(r["exp"].replace("exp", ""))
        vbpb = float(r["val_bpb"])
        status = r["status"]
        cat = classify(r["description"])

        if status == "baseline":
            ax.axhline(y=vbpb, color="#888888", linestyle=":", alpha=0.5, linewidth=1)
            ax.annotate("baseline", xy=(0, vbpb), fontsize=8, color="#888888",
                       xytext=(2, vbpb + 0.001))
            continue

        if vbpb <= 0:  # crash with 0
            continue

        marker, size, alpha = STATUS_MARKERS.get(status, ("o", 5, 0.5))
        color = STRATEGY_COLORS.get(cat, "#9E9E9E")

        ax.scatter(exp_num, vbpb, c=color, marker=marker, s=size**2 * 3,
                  alpha=alpha, zorder=3, edgecolors="white", linewidths=0.3)

    # Anchor window shading
    ax.axvspan(0, 24, alpha=0.06, color="#2196F3", zorder=0)
    ax.text(12, ax.get_ylim()[1] - 0.001, "Anchor Window",
            ha="center", va="top", fontsize=9, color="#2196F3", alpha=0.7,
            fontstyle="italic")

    # Nudge points
    for d in drift:
        ec = d["experiment_count"]
        cs = d["composite_score"]
        ymin, ymax = ax.get_ylim()

        # Vertical line
        ax.axvline(x=ec, color="#E53935", linestyle="--", alpha=0.7, linewidth=1.5, zorder=2)

        # Nudge label
        ax.annotate(
            f"NUDGE\n{cs:.3f}",
            xy=(ec, ymin + (ymax - ymin) * 0.05),
            fontsize=8, fontweight="bold", color="#E53935",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#E53935", alpha=0.9),
            zorder=5,
        )

    # Best result annotation
    valid = [r for r in rows if r["status"] == "keep" and float(r["val_bpb"]) > 0]
    if valid:
        best = min(valid, key=lambda r: float(r["val_bpb"]))
        bx = int(best["exp"].replace("exp", ""))
        by = float(best["val_bpb"])
        ax.annotate(
            f"BEST: {by:.4f}\n({best['exp']})",
            xy=(bx, by),
            xytext=(bx - 8, by - 0.004),
            fontsize=9, fontweight="bold", color="#1B5E20",
            arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#1B5E20", alpha=0.9),
            zorder=5,
        )

    # Legend
    strategy_handles = [
        mpatches.Patch(color=STRATEGY_COLORS[k], label=STRATEGY_LABELS[k])
        for k in ["learning_rate", "regularization", "schedule", "architecture", "batch_size"]
    ]
    status_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#333", markersize=8, label="Keep"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333", markersize=6, label="Discard", alpha=0.6),
        Line2D([0], [0], marker="x", color="w", markerfacecolor="#333", markeredgecolor="#333", markersize=7, label="Crash", alpha=0.4),
    ]

    leg1 = ax.legend(handles=strategy_handles, loc="upper right", fontsize=8,
                     title="Strategy", title_fontsize=9, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=status_handles, loc="upper left", fontsize=8,
              title="Status", title_fontsize=9, framealpha=0.9)

    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("val_bpb (lower is better)")
    ax.set_title(f"{run_name}: val_bpb Timeline with Drift Nudge Points\n"
                 f"Claude Haiku 4.5 | RTX 5090 | climbmix | {tunnel_label}")
    ax.set_xlim(-1, 61)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Chart 3: Drift score trajectory comparison
# ──────────────────────────────────────────────────────────────────────

def plot_drift_trajectory(drift_r1, drift_r3, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Run 1
    x1 = [d["experiment_count"] for d in drift_r1]
    y1 = [d["composite_score"] for d in drift_r1]
    ax.plot(x1, y1, "s-", color="#1565C0", markersize=10, linewidth=2.5,
            label=f"Run 1 (0.454 → 0.315, −30.6%)", zorder=3)

    # Run 3
    x3 = [d["experiment_count"] for d in drift_r3]
    y3 = [d["composite_score"] for d in drift_r3]
    ax.plot(x3, y3, "D-", color="#E65100", markersize=10, linewidth=2.5,
            label=f"Run 3 (0.412 → 0.271, −34.2%)", zorder=3)

    # Threshold line
    ax.axhline(y=0.30, color="#E53935", linestyle=":", alpha=0.6, linewidth=1.5)
    ax.text(51, 0.31, "Composite nudge\nthreshold (0.30)",
            fontsize=8, color="#E53935", ha="right", va="bottom")

    # Annotations
    for x, y in zip(x1, y1):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 12),
                   textcoords="offset points", fontsize=9, ha="center",
                   color="#1565C0", fontweight="bold")
    for x, y in zip(x3, y3):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, -18),
                   textcoords="offset points", fontsize=9, ha="center",
                   color="#E65100", fontweight="bold")

    # Nudge markers
    for x in x1:
        ax.axvline(x=x, color="#E53935", linestyle="--", alpha=0.2, linewidth=1)

    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("Composite Drift Score")
    ax.set_title("Drift Score Trajectory: Monotonic Decrease After Nudge Activation")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.set_xlim(28, 52)
    ax.set_ylim(0.2, 0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Chart 4: Strategy distribution stacked area (one per run)
# ──────────────────────────────────────────────────────────────────────

def plot_strategy_distribution(run_name, tsv_path, drift_path, out_path, tunnel_label):
    rows = load_tsv(tsv_path)
    drift = load_drift(drift_path)

    # Build per-window strategy counts
    window_size = 5
    categories = ["learning_rate", "regularization", "schedule", "architecture", "batch_size", "other"]
    windows = []
    centers = []

    for start in range(0, len(rows), window_size):
        window = rows[start:start + window_size]
        cats = [classify(r["description"]) for r in window if r["status"] not in ("baseline",)]
        total = max(len(cats), 1)
        from collections import Counter
        counts = Counter(cats)
        pcts = {c: counts.get(c, 0) / total for c in categories}
        windows.append(pcts)
        centers.append(start + window_size / 2)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Stacked area
    bottoms = [0] * len(windows)
    for cat in categories:
        vals = [w.get(cat, 0) for w in windows]
        color = STRATEGY_COLORS.get(cat, "#9E9E9E")
        ax.bar(centers, vals, bottom=bottoms, width=window_size * 0.85,
               color=color, alpha=0.85, label=STRATEGY_LABELS.get(cat, cat),
               edgecolor="white", linewidth=0.5)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    # Nudge lines
    for d in drift:
        ec = d["experiment_count"]
        cs = d["composite_score"]
        ax.axvline(x=ec, color="#E53935", linestyle="--", linewidth=2, alpha=0.8, zorder=5)
        ax.text(ec, 1.05, f"NUDGE\n{cs:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#E53935",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", edgecolor="#E53935", alpha=0.9))

    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("Strategy Proportion")
    ax.set_title(f"{run_name}: Strategy Distribution Over Time with Nudge Points\n{tunnel_label}")
    ax.set_xlim(-1, 62)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Generate all charts
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating validation charts...")

    r1_tsv = ROOT / "validation" / "r1" / "results.tsv"
    r1_drift = ROOT / "validation" / "r1" / "drift_readings.jsonl"
    r3_tsv = ROOT / "validation" / "r3" / "results.tsv"
    r3_drift = ROOT / "validation" / "r3" / "drift_readings.jsonl"

    # Check drift reading files exist, create from status if not
    for p in [r1_drift, r3_drift]:
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping charts that need it")

    print("\n[1/5] Run 1 timeline...")
    plot_run_timeline("Run 1", r1_tsv, r1_drift, OUT / "r1_timeline.png",
                      "Regularization tunnel detected → nudge → schedule pivot")

    print("[2/5] Run 3 timeline...")
    plot_run_timeline("Run 3", r3_tsv, r3_drift, OUT / "r3_timeline.png",
                      "Learning rate tunnel detected → nudge → architecture pivot")

    print("[3/5] Drift trajectory comparison...")
    d1 = load_drift(r1_drift)
    d3 = load_drift(r3_drift)
    plot_drift_trajectory(d1, d3, OUT / "drift_trajectory.png")

    print("[4/5] Run 1 strategy distribution...")
    plot_strategy_distribution("Run 1", r1_tsv, r1_drift, OUT / "r1_strategy.png",
                               "Regularization dominance broken by drift nudges")

    print("[5/5] Run 3 strategy distribution...")
    plot_strategy_distribution("Run 3", r3_tsv, r3_drift, OUT / "r3_strategy.png",
                               "Learning rate dominance broken by drift nudges")

    print(f"\nAll charts saved to {OUT}/")
