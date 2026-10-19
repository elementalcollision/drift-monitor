"""Tests for multi-segment drift history (SegmentHistory, layered over DualWindow)."""

from drift_monitor import SegmentHistory

RICH = "The idempotent retry handler uses memoization to cache previous results."
POOR = "The handler caches results."


def _feed_segments(sh: SegmentHistory, segments: list[list[str]]) -> None:
    for i, seg in enumerate(segments):
        for text in seg:
            sh.observe(text)
        if i < len(segments) - 1:
            sh.mark_boundary()


def test_counts_segments_and_boundaries():
    sh = SegmentHistory()
    sh.observe("one")
    assert sh.segment_count == 1
    assert sh.boundary_count == 0

    sh.mark_boundary()
    sh.observe("two")
    assert sh.segment_count == 2
    assert sh.boundary_count == 1


def test_no_boundaries_yields_no_reports():
    sh = SegmentHistory()
    sh.observe("only one segment")
    assert sh.boundary_reports() == []
    assert sh.trend() == []


def test_one_boundary_yields_no_report_yet_lag():
    # The segment AFTER the boundary has no observations to compare against, so
    # a single boundary produces zero reports until a second boundary closes
    # that segment off.
    sh = SegmentHistory()
    for _ in range(3):
        sh.observe(RICH)
    sh.mark_boundary()
    for _ in range(3):
        sh.observe(RICH)  # still open, not closed
    assert sh.boundary_reports() == []


def test_trend_reflects_drift_introduction():
    # Four segments: rich -> rich (no drift) -> poor -> poor. The drift appears
    # at the rich->poor boundary (index 1); the rich/rich and poor/poor adjacencies
    # are stable.
    sh = SegmentHistory()
    _feed_segments(sh, [[RICH] * 3, [RICH] * 3, [POOR] * 3, [POOR] * 3])

    assert sh.boundary_count == 3
    trend = sh.trend()
    assert len(trend) == 2

    assert trend[0] < 0.05  # rich vs rich == no drift
    assert trend[1] > trend[0]  # rich vs poor == drift introduced
    assert trend[1] > 0.1


def test_trend_is_chronological_and_stabilizes():
    # rich, rich, poor, poor, poor -> drift appears at boundary 1, then stabilises.
    sh = SegmentHistory()
    _feed_segments(sh, [[RICH] * 3, [RICH] * 3, [POOR] * 3, [POOR] * 3, [POOR] * 3])
    trend = sh.trend()
    assert len(trend) == 3
    assert trend[0] < 0.05  # stable start
    assert trend[1] > trend[0]  # drift appears
    assert trend[2] < trend[1]  # poor == poor stabilises again
    assert trend[2] < 0.05


def test_reset_clears_history():
    sh = SegmentHistory()
    _feed_segments(sh, [[RICH] * 3, [POOR] * 3])
    sh.reset()
    assert sh.boundary_count == 0
    assert sh.segment_count == 1
    assert sh.trend() == []


def test_persistence_roundtrip():
    sh = SegmentHistory()
    _feed_segments(sh, [[RICH] * 3, [POOR] * 3, [POOR] * 3])

    restored = SegmentHistory.from_state(sh.to_state())

    assert restored.boundary_count == sh.boundary_count
    assert restored.segment_count == sh.segment_count
    assert restored.trend() == sh.trend()


def test_reports_are_full_drift_reports():
    sh = SegmentHistory()
    _feed_segments(sh, [[RICH] * 3, [POOR] * 3, [POOR] * 3])
    reports = sh.boundary_reports()
    assert reports
    # each report embeds per-instrument readings + a classification
    assert len(reports[0].readings) == 3
    assert {r.instrument for r in reports[0].readings} == {
        "ghost_lexicon",
        "behavioral_footprint",
        "semantic_drift",
    }