"""JSONL storage with atomic write operations.

All drift readings and observations are stored as JSONL for maximum
interoperability with other tools and pipelines.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_write(path: str | Path, content: str) -> None:
    """Write content atomically using temp-file-then-rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_append(path: str | Path, line: str) -> None:
    """Append a line atomically (as atomic as append gets on POSIX)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line if line.endswith("\n") else line + "\n")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write a list of records as JSONL, atomically."""
    lines = [json.dumps(r, default=str) for r in records]
    atomic_write(path, "\n".join(lines) + "\n" if lines else "")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single record to a JSONL file."""
    atomic_append(path, json.dumps(record, default=str))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping corrupt lines."""
    path = Path(path)
    if not path.exists():
        return []

    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip corrupt trailing lines (common after crashes)
                continue
    return records


def load_texts_from_jsonl(
    path: str | Path,
    text_field: str = "text",
) -> list[dict[str, Any]]:
    """Load records from JSONL, ensuring each has a text field."""
    records = read_jsonl(path)
    return [r for r in records if text_field in r]


def save_drift_report(
    path: str | Path,
    report_dict: dict[str, Any],
) -> None:
    """Save a drift report, adding a timestamp."""
    report_dict["timestamp"] = time.time()
    append_jsonl(path, report_dict)
