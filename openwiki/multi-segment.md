---
type: "Reference"
title: "Multi-Segment History"
description: "SegmentHistory — a purely-additive layer over DualWindow that tracks drift across many compression boundaries and exposes the per-boundary trend."
tags: [history, multi-segment, trend, persistence]
---

# Multi-Segment History

**Status:** Accepted (Phase 5)
**Source:** `drift_monitor/history.py` (`SegmentHistory`)

## What it adds

`DualWindow` compares exactly **two** windows — anchor (before a compression
boundary) vs recent (after) — which is the right model for a single boundary but
cannot answer "is drift *accumulating* or *recovering* across a session with many
compression events?"

`SegmentHistory` layers on top of `DualWindow` to track an arbitrary number of
boundaries and expose the drift **trend** over a session. It is purely additive:
`DualWindow`, the instruments, and their existing tests are untouched. It reuses
the same instrument trio + keyword-mode semantics as `significance.py`, so a trend
is directly comparable against a single-boundary reading.

## Model

A **segment** is the sequence of observations between two consecutive boundaries.
`observe()` appends to the current (open) segment; `mark_boundary()` closes it and
opens a new one.

```python
from drift_monitor import SegmentHistory

history = SegmentHistory(window_size=50)

for text in pre_compression_1:
    history.observe(text)
history.mark_boundary()          # boundary 0
for text in post_1:
    history.observe(text)
history.mark_boundary()          # boundary 1
# ...
```

Per-boundary drift compares **consecutive closed segments** with the standard
trio (`GhostLexicon` + `BehavioralFootprint` + keyword-mode `SemanticDrift`), so
the `k`-th report is "segment `k-1` vs segment `k`".

## The one-boundary lag

The drift at boundary `k` compares the segment *before* it against the segment
*after* it — but the after-segment needs observations before it can be compared.
So boundary `k` is only reportable once segment `k+1` has also been closed (one
boundary later). `boundary_reports()` therefore returns `max(0, len(closed) - 1)`
reports; the most recent boundary is always one behind.

This is the natural consequence of comparing two *complete* things: you cannot
score drift against a segment that has not happened yet.

## API reference

- `observe(text, metadata=None)` — record an observation in the open segment.
- `mark_boundary()` — close the current segment and open a new one.
- `reset()` — drop all segments.
- `segment_count` — closed segments + the open one.
- `boundary_count` — number of `mark_boundary()` calls.
- `boundary_reports() -> list[DriftReport]` — one full report per completed boundary,
  in chronological order (each embeds per-instrument readings + compression type).
- `trend() -> list[float]` — composite drift score per boundary; the headline
  "is drift accumulating or recovering?" number.
- `to_state()` / `from_state()` — persistence (reports are recomputed on load, since
  they are deterministic functions of the segments).

## Comparison horizon vs history

Segments are stored **in full** (for audit), but each pair is compared over its
last `window_size` observations — the same `window_size` used by the instruments.
A long-lived process should snapshot or trim old segments as needed; the class
does not impose a bound.

## Example — detecting a drift-relapse pattern

```python
trend = history.trend()
# e.g. [0.01, 0.68, 0.02, 0.02]  -> no drift, then a spike, then re-stabilised
if trend and trend[-1] > 0.3:
    print("current boundary drifted")
if len(trend) >= 2 and trend[-1] > trend[-2]:
    print("drift is worsening across boundaries")
```

## Limitation

`SegmentHistory` is a *replay/comparison* facility — it computes per-boundary drift
on demand by re-feeding segments into fresh instruments. It is not yet wired into
the live `DriftHarness` assessment loop (which still uses a single `DualWindow` per
instrument); that integration, if wanted, is a follow-on.