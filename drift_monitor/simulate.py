"""Synthetic drift generator for validation.

Produces pre/post compression text pairs with known drift characteristics
to validate that instruments detect what they should (and stay silent when
they should).
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any


class DriftMode(Enum):
    VOCABULARY = "vocabulary"
    TOPIC = "topic"
    TOOLCALL = "toolcall"
    COMBINED = "combined"
    FRAMING = "framing"  # Intentionally invisible to surface instruments


# Domain-specific technical vocabulary that compression tends to strip
TECHNICAL_TERMS = [
    "idempotent",
    "memoization",
    "backpressure",
    "linearizable",
    "sharding",
    "quorum",
    "tombstone",
    "compaction",
    "vectorization",
    "denormalization",
    "deadlock",
    "livelock",
    "semaphore",
    "mutex",
    "coroutine",
    "goroutine",
    "serialization",
    "marshalling",
    "debounce",
    "throttle",
]

# Generic replacements that compression substitutes in
GENERIC_REPLACEMENTS = [
    "process",
    "handle",
    "manage",
    "configure",
    "operation",
    "function",
    "method",
    "approach",
    "system",
    "component",
    "module",
    "service",
    "handler",
    "worker",
    "task",
    "step",
    "action",
    "item",
    "element",
    "object",
]

# Sample technical responses (anchor window content)
SAMPLE_PRE_RESPONSES = [
    "The idempotent retry handler uses memoization to cache previous results. "
    "When backpressure exceeds the threshold, we apply throttle logic to prevent "
    "the downstream semaphore from deadlocking.",
    "We need linearizable reads here because the sharding strategy distributes "
    "tombstone records across quorum nodes. The compaction process must serialize "
    "these before the goroutine pool exhausts its coroutine budget.",
    "Apply debounce to the vectorization pipeline input. The denormalization step "
    "requires careful marshalling of the intermediate state to avoid livelock "
    "conditions in the mutex acquisition path.",
    "The backpressure mechanism triggers when the semaphore count drops below "
    "the quorum threshold. Each goroutine handles serialization of tombstone "
    "records through the memoization cache.",
    "After compaction, the linearizable guarantee ensures idempotent writes "
    "across sharding boundaries. The throttle prevents coroutine exhaustion "
    "during vectorization of denormalized data.",
]

# Tool usage patterns for behavioral drift
TOOL_SETS = {
    "pre": ["read_file", "grep", "edit_file", "run_tests", "read_file", "grep"],
    "post_shifted": ["read_file", "read_file", "read_file", "write_file", "bash"],
}


def apply_vocabulary_drift(
    texts: list[str],
    replacement_rate: float = 0.7,
    seed: int | None = None,
) -> list[str]:
    """Replace technical terms with generic alternatives."""
    rng = random.Random(seed)
    result = []
    for text in texts:
        words = text.split()
        new_words = []
        for word in words:
            clean = word.strip(".,;:!?()[]{}\"'")
            if clean.lower() in [t.lower() for t in TECHNICAL_TERMS]:
                if rng.random() < replacement_rate:
                    replacement = rng.choice(GENERIC_REPLACEMENTS)
                    new_words.append(word.replace(clean, replacement))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        result.append(" ".join(new_words))
    return result


def apply_topic_drift(
    texts: list[str],
    seed: int | None = None,
) -> list[str]:
    """Replace with generic filler content simulating focus loss."""
    rng = random.Random(seed)
    fillers = [
        "We should consider the overall approach to this problem carefully.",
        "The system needs to handle this in a robust way.",
        "Let me think about the best way to implement this functionality.",
        "We can use a standard approach to process this data.",
        "The component should be designed to handle various inputs.",
        "This operation needs to be performed efficiently.",
        "We should ensure the solution is maintainable and scalable.",
        "The implementation should follow best practices.",
    ]
    return [rng.choice(fillers) for _ in texts]


def apply_toolcall_drift(
    metadata_list: list[dict[str, Any]],
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Shift tool usage patterns to simulate behavioral drift."""
    rng = random.Random(seed)
    shifted_tools = TOOL_SETS["post_shifted"]
    result = []
    for meta in metadata_list:
        new_meta = dict(meta)
        # Replace tools with shifted distribution
        n_tools = len(meta.get("tools", []))
        new_meta["tools"] = [rng.choice(shifted_tools) for _ in range(max(1, n_tools))]
        result.append(new_meta)
    return result


def apply_framing_drift(texts: list[str], seed: int | None = None) -> list[str]:
    """Shift implicit priors while preserving surface metrics.

    This mode is intentionally INVISIBLE to all three surface instruments.
    It validates that the instruments correctly report no drift when only
    deep framing changes.
    """
    # Keep exact vocabulary and structure, but subtly reframe
    # In practice this means the texts are nearly identical
    result = []
    for text in texts:
        # Preserve all technical terms but add hedging qualifiers
        reframed = text.replace("must", "could potentially")
        reframed = reframed.replace("requires", "might benefit from")
        reframed = reframed.replace("ensures", "attempts to provide")
        result.append(reframed)
    return result


def generate_drift_pair(
    mode: DriftMode,
    n_samples: int = 5,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate pre/post compression observation pairs with known drift.

    Returns (pre_records, post_records) where each record has "text" and
    optional "tools" metadata.
    """
    rng = random.Random(seed)

    # Build pre-compression records
    pre_texts = SAMPLE_PRE_RESPONSES[:n_samples]
    pre_tools = TOOL_SETS["pre"]
    pre_records = [
        {"text": t, "tools": [pre_tools[i % len(pre_tools)]]}
        for i, t in enumerate(pre_texts)
    ]

    # Apply drift based on mode
    if mode == DriftMode.VOCABULARY:
        post_texts = apply_vocabulary_drift(pre_texts, seed=seed)
        post_records = [
            {"text": t, "tools": r["tools"]}
            for t, r in zip(post_texts, pre_records)
        ]

    elif mode == DriftMode.TOPIC:
        post_texts = apply_topic_drift(pre_texts, seed=seed)
        post_records = [
            {"text": t, "tools": r["tools"]}
            for t, r in zip(post_texts, pre_records)
        ]

    elif mode == DriftMode.TOOLCALL:
        post_meta = apply_toolcall_drift(pre_records, seed=seed)
        post_records = [
            {"text": r["text"], "tools": m["tools"]}
            for r, m in zip(pre_records, post_meta)
        ]

    elif mode == DriftMode.COMBINED:
        post_texts = apply_vocabulary_drift(pre_texts, seed=seed)
        post_meta = apply_toolcall_drift(pre_records, seed=seed)
        post_records = [
            {"text": t, "tools": m["tools"]}
            for t, m in zip(post_texts, post_meta)
        ]

    elif mode == DriftMode.FRAMING:
        post_texts = apply_framing_drift(pre_texts, seed=seed)
        post_records = [
            {"text": t, "tools": r["tools"]}
            for t, r in zip(post_texts, pre_records)
        ]

    else:
        post_records = list(pre_records)

    return pre_records, post_records


def validate_instruments(
    n_trials: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Run validation suite across all drift modes.

    Returns detection rates for each mode and instrument.
    """
    from drift_monitor.instruments.ghost_lexicon import GhostLexicon
    from drift_monitor.instruments.behavioral import BehavioralFootprint
    from drift_monitor.instruments.semantic import SemanticDrift

    results: dict[str, dict[str, list[float]]] = {}

    for mode in DriftMode:
        mode_results: dict[str, list[float]] = {
            "ghost_lexicon": [],
            "behavioral_footprint": [],
            "semantic_drift": [],
        }

        for trial in range(n_trials):
            trial_seed = seed + trial
            pre, post = generate_drift_pair(mode, seed=trial_seed)

            for InstrClass, key in [
                (GhostLexicon, "ghost_lexicon"),
                (BehavioralFootprint, "behavioral_footprint"),
                (SemanticDrift, "semantic_drift"),
            ]:
                instr = InstrClass(window_size=20)
                # Feed anchor
                for r in pre:
                    instr.observe(r["text"], {"tools": r.get("tools", [])})
                instr.mark_boundary()
                # Feed recent
                for r in post:
                    instr.observe(r["text"], {"tools": r.get("tools", [])})

                mode_results[key].append(instr.score())

        results[mode.value] = {
            instrument: {
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "detection_rate": (
                    sum(1 for s in scores if s > 0.1) / len(scores)
                    if scores
                    else 0.0
                ),
                "scores": [round(s, 4) for s in scores],
            }
            for instrument, scores in mode_results.items()
        }

    return results
