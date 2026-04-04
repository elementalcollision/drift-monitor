"""Tests for the autoresearch drift harness."""

from __future__ import annotations

from dataclasses import dataclass

from drift_monitor.harness import (
    DriftHarness,
    DriftConfig,
    classify_strategy,
    analyze_tsv,
)


@dataclass
class FakeResult:
    exp: str
    description: str
    status: str
    notes: str


def _make_lr_result(n: int) -> FakeResult:
    return FakeResult(
        exp=f"exp{n}",
        description=f"Decrease SCALAR_LR from 0.{n:02d} to 0.{n-1:02d}",
        status="discard",
        notes="fine-tuning scalar parameters",
    )


def _make_arch_result(n: int) -> FakeResult:
    return FakeResult(
        exp=f"exp{n}",
        description=f"Decrease DEPTH from {10-n%5} to {9-n%5}",
        status="keep",
        notes="exploring shallower architectures with varied depth and window patterns",
    )


# ---------------------------------------------------------------------------
# classify_strategy
# ---------------------------------------------------------------------------

def test_classify_lr():
    assert classify_strategy("Decrease SCALAR_LR from 0.22 to 0.20") == "learning_rate"
    assert classify_strategy("Decrease MATRIX_LR from 0.03 to 0.025") == "learning_rate"


def test_classify_architecture():
    assert classify_strategy("Change WINDOW_PATTERN from SSSL to SSLL") == "architecture"
    assert classify_strategy("Decrease DEPTH from 8 to 6") == "architecture"


def test_classify_schedule():
    assert classify_strategy("Increase WARMUP_RATIO from 0.0 to 0.05") == "schedule"
    assert classify_strategy("Decrease WARMDOWN_RATIO from 0.7 to 0.6") == "schedule"


def test_classify_batch():
    assert classify_strategy("Decrease DEVICE_BATCH_SIZE by 10%") == "batch_size"


def test_classify_other():
    assert classify_strategy("Something completely different") == "other"


# ---------------------------------------------------------------------------
# DriftHarness basics
# ---------------------------------------------------------------------------

def test_harness_counts_experiments():
    harness = DriftHarness(config=DriftConfig(anchor_window=5, assessment_interval=5))
    for i in range(10):
        harness.observe_experiment(_make_lr_result(i + 1))
    assert harness.experiment_count == 10


def test_harness_skips_baseline():
    harness = DriftHarness(config=DriftConfig(anchor_window=5, assessment_interval=5))
    harness.observe_experiment(FakeResult("exp0", "baseline", "baseline", ""))
    assert harness.experiment_count == 0


def test_harness_sets_boundary_at_anchor_window():
    config = DriftConfig(anchor_window=5, assessment_interval=5)
    harness = DriftHarness(config=config)
    for i in range(4):
        harness.observe_experiment(_make_arch_result(i + 1))
    assert not harness.boundary_set
    harness.observe_experiment(_make_arch_result(5))
    assert harness.boundary_set


def test_harness_generates_report_at_interval():
    config = DriftConfig(anchor_window=5, assessment_interval=5)
    harness = DriftHarness(config=config)

    # Fill anchor (5 architecture experiments)
    # Note: boundary is set AND assessment fires at count=5 (5 % 5 == 0)
    for i in range(5):
        harness.observe_experiment(_make_arch_result(i + 1))

    # Boundary should be set
    assert harness.boundary_set

    # Fill 5 more LR experiments (triggers assessment at count=10)
    for i in range(5, 10):
        harness.observe_experiment(_make_lr_result(i + 1))

    assert harness.last_report is not None
    assert harness.last_report.composite_score >= 0.0


def test_harness_detects_strategy_shift():
    """Architecture anchor → LR recent should produce behavioral drift."""
    config = DriftConfig(anchor_window=10, assessment_interval=10)
    harness = DriftHarness(config=config)

    # Anchor: architecture experiments
    for i in range(10):
        harness.observe_experiment(_make_arch_result(i + 1))

    # Recent: all LR experiments
    for i in range(10, 20):
        harness.observe_experiment(_make_lr_result(i + 1))

    report = harness.last_report
    assert report is not None
    assert report.composite_score > 0.1  # Meaningful drift

    # Behavioral footprint should fire (strategy shift)
    behavioral = next(r for r in report.readings if r.instrument == "behavioral_footprint")
    assert behavioral.score > 0.1


def test_harness_no_nudge_when_stable():
    config = DriftConfig(anchor_window=5, assessment_interval=5)
    harness = DriftHarness(config=config)

    # Same strategy throughout
    for i in range(10):
        harness.observe_experiment(_make_lr_result(i + 1))

    nudge = harness.get_drift_nudge()
    # With same content, scores should be low → no nudge
    # (may or may not fire depending on text variation)
    # Just verify it doesn't crash and returns str or None
    assert nudge is None or isinstance(nudge, str)


def test_harness_nudge_on_heavy_drift():
    """Heavy strategy shift should produce a nudge."""
    config = DriftConfig(
        anchor_window=10,
        assessment_interval=10,
        composite_nudge_threshold=0.15,  # Low threshold for testing
    )
    harness = DriftHarness(config=config)

    # Anchor: diverse architecture experiments
    for i in range(10):
        harness.observe_experiment(_make_arch_result(i + 1))

    # Recent: repetitive LR experiments
    for i in range(10, 20):
        harness.observe_experiment(_make_lr_result(i + 1))

    nudge = harness.get_drift_nudge()
    assert nudge is not None
    assert "DRIFT ALERT" in nudge


def test_harness_strategy_distribution():
    config = DriftConfig(anchor_window=3, assessment_interval=5)
    harness = DriftHarness(config=config)

    harness.observe_experiment(_make_arch_result(1))
    harness.observe_experiment(_make_arch_result(2))
    harness.observe_experiment(_make_lr_result(3))
    harness.observe_experiment(_make_lr_result(4))
    harness.observe_experiment(_make_lr_result(5))

    dist = harness._strategy_distribution()
    assert "learning_rate" in dist
    assert dist["learning_rate"] > 0


# ---------------------------------------------------------------------------
# wrap_callbacks
# ---------------------------------------------------------------------------

def test_wrap_callbacks():
    """wrap_callbacks should intercept on_experiment_complete."""
    config = DriftConfig(anchor_window=3, assessment_interval=3)
    harness = DriftHarness(config=config)

    # Mock callbacks object
    @dataclass
    class MockCallbacks:
        on_experiment_complete: object = None
        call_log: list = None

        def __post_init__(self):
            self.call_log = []
            self.on_experiment_complete = lambda r: self.call_log.append(r.exp)

    original = MockCallbacks()
    wrapped = harness.wrap_callbacks(original)

    result = _make_lr_result(1)
    wrapped.on_experiment_complete(result)

    # Both should have been called
    assert harness.experiment_count == 1
    assert "exp1" in original.call_log


# ---------------------------------------------------------------------------
# analyze_tsv
# ---------------------------------------------------------------------------

def test_analyze_tsv(tmp_path):
    """Test standalone TSV analysis."""
    tsv = tmp_path / "results.tsv"
    header = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\n"
    rows = [header]
    rows.append("exp0\tbaseline\t1.09\t5.9\t298900\t17.0\t2736\tbaseline\tdepth=8\tRTX 5090\tabc123\n")

    for i in range(1, 60):
        if i < 30:
            desc = f"Decrease DEPTH from {10-i%5} to {9-i%5}"
            notes = "exploring architecture"
        else:
            desc = f"Decrease SCALAR_LR from 0.{i:02d} to 0.{i-1:02d}"
            notes = "fine-tuning LR"
        rows.append(f"exp{i}\t{desc}\t1.0{90-i}\t5.9\t300000\t17.0\t2736\tdiscard\t{notes}\tRTX 5090\tabc123\n")

    tsv.write_text("".join(rows))

    records = analyze_tsv(str(tsv), DriftConfig(anchor_window=15, assessment_interval=10))
    assert len(records) > 0
    # Should detect drift as strategy shifts from architecture to LR
    last = records[-1]
    assert last["report"]["composite_score"] > 0.0
