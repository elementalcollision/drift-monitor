"""Tests for the CLI."""

import json
import tempfile
from pathlib import Path

from drift_monitor.cli import main
from drift_monitor.storage import write_jsonl


def _write_test_jsonl(records: list[dict]) -> str:
    """Write records to a temp JSONL file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def test_demo_runs(capsys):
    main(["demo"])
    captured = capsys.readouterr()
    assert "Drift Monitor Demo" in captured.out
    assert "vocabulary" in captured.out
    assert "composite=" in captured.out


def test_validate_runs(capsys):
    main(["validate", "--trials", "3", "--seed", "42"])
    captured = capsys.readouterr()
    assert "vocabulary" in captured.out
    assert "framing" in captured.out


def test_run_with_files(capsys):
    pre = [
        {"text": "The idempotent handler uses memoization.", "tools": ["read"]},
        {"text": "Apply idempotent logic with memoization.", "tools": ["read"]},
        {"text": "Check idempotent guarantee and memoization.", "tools": ["read"]},
    ]
    post = [
        {"text": "The process handler uses caching.", "tools": ["write"]},
        {"text": "Apply process logic.", "tools": ["write"]},
        {"text": "Check the guarantee.", "tools": ["write"]},
    ]
    pre_path = _write_test_jsonl(pre)
    post_path = _write_test_jsonl(post)

    main(["run", "--pre", pre_path, "--post", post_path])
    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert "composite_score" in result
    assert "compression_type" in result
    assert result["composite_score"] > 0.1


def test_ghost_lexicon_command(capsys):
    pre = [
        {"text": "semaphore mutex deadlock"},
        {"text": "semaphore mutex livelock"},
        {"text": "semaphore coroutine goroutine"},
    ]
    post = [
        {"text": "lock handle process"},
        {"text": "lock handle action"},
        {"text": "lock task worker"},
    ]
    pre_path = _write_test_jsonl(pre)
    post_path = _write_test_jsonl(post)

    main(["ghost-lexicon", "--pre", pre_path, "--post", post_path])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["instrument"] == "ghost_lexicon"
    assert result["score"] > 0.0
