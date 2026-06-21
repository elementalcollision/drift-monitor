"""Tests for the JSONL storage helpers in drift_monitor.storage."""

from __future__ import annotations

import json

from drift_monitor.storage import read_jsonl, write_jsonl


def test_write_jsonl_then_read_jsonl_round_trip(tmp_path):
    """Records written with write_jsonl come back identically via read_jsonl."""
    records = [
        {"id": 1, "text": "hello world", "score": 0.5},
        {"id": 2, "text": "second entry", "nested": {"a": 1}},
    ]
    path = tmp_path / "readings.jsonl"

    write_jsonl(path, records)
    loaded = read_jsonl(path)

    assert loaded == records


def test_read_jsonl_returns_empty_for_missing_file(tmp_path):
    """read_jsonl on a path that does not exist returns []."""
    missing = tmp_path / "does_not_exist.jsonl"
    assert not missing.exists()
    assert read_jsonl(missing) == []


def test_read_jsonl_skips_blank_and_corrupt_lines(tmp_path):
    """Blank lines and non-JSON lines are silently skipped."""
    path = tmp_path / "mixed.jsonl"
    good_first = {"id": 1, "text": "first"}
    good_last = {"id": 2, "text": "last"}

    raw = "\n".join(
        [
            json.dumps(good_first),
            "",                      # blank line
            "   ",                   # whitespace-only line
            "not valid json at all", # corrupt
            "{also not: json",       # corrupt
            json.dumps(good_last),
            "",
        ]
    )
    path.write_text(raw, encoding="utf-8")

    loaded = read_jsonl(path)
    assert loaded == [good_first, good_last]
