"""Example: CLI pipeline with JSONL files.

This script generates sample JSONL files and shows how to use the CLI
to analyze them.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from drift_monitor.simulate import DriftMode, generate_drift_pair


def main():
    # Generate synthetic data
    pre_records, post_records = generate_drift_pair(
        DriftMode.COMBINED,
        n_samples=5,
        seed=42,
    )

    # Write to temp JSONL files
    pre_path = Path(tempfile.mktemp(suffix="_pre.jsonl"))
    post_path = Path(tempfile.mktemp(suffix="_post.jsonl"))

    with open(pre_path, "w") as f:
        for r in pre_records:
            f.write(json.dumps(r) + "\n")

    with open(post_path, "w") as f:
        for r in post_records:
            f.write(json.dumps(r) + "\n")

    print(f"Pre-compression:  {pre_path}")
    print(f"Post-compression: {post_path}")
    print()

    # Run via CLI
    result = subprocess.run(
        [sys.executable, "-m", "drift_monitor", "run", "--pre", str(pre_path), "--post", str(post_path)],
        capture_output=True,
        text=True,
    )
    print("=== CLI Output ===")
    print(result.stdout)

    # Cleanup
    pre_path.unlink()
    post_path.unlink()


if __name__ == "__main__":
    main()
