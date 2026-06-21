"""Tests for ObservationWindow and DualWindow in drift_monitor.window.

Includes faithfulness mutants that must be killed:
- int 50->51 (anchor.max_size, recent.max_size defaults)
- bool False->True (_post_boundary init default)
- bool True->False (mark_boundary sets _post_boundary)
- bool False->True (reset clears _post_boundary)
- int 3->4 (min_observations default)
- compare GtE->Lt (anchor len check in has_enough_data)
- compare GtE->Lt (recent len check in has_enough_data)
"""

import pytest

from drift_monitor.window import DualWindow, ObservationWindow


# ---------------------------------------------------------------------------
# ObservationWindow
# ---------------------------------------------------------------------------


def test_observationwindow_add_then_texts_returns_texts_in_order() -> None:
    win = ObservationWindow(max_size=5)
    win.add("alpha")
    win.add("beta")
    win.add("gamma")
    assert win.texts == ["alpha", "beta", "gamma"]


def test_observationwindow_len_reflects_count() -> None:
    win = ObservationWindow(max_size=5)
    assert len(win) == 0
    win.add("a")
    win.add("b")
    assert len(win) == 2
    win.add("c")
    assert len(win) == 3


def test_observationwindow_clear_empties_window() -> None:
    win = ObservationWindow(max_size=5)
    win.add("keep")
    win.add("these")
    assert len(win) == 2
    win.clear()
    assert len(win) == 0
    assert win.texts == []


def test_observationwindow_exceeding_max_size_evicts_oldest() -> None:
    win = ObservationWindow(max_size=3)
    win.add("one")
    win.add("two")
    win.add("three")
    # Now full; adding more should evict the oldest ("one")
    win.add("four")
    assert len(win) == 3
    assert win.texts == ["two", "three", "four"]
    # Evicts "two" next
    win.add("five")
    assert win.texts == ["three", "four", "five"]


def test_observationwindow_metadata_defaults_is_empty_dict() -> None:
    win = ObservationWindow(max_size=4)
    win.add("hello")
    win.add("hi", metadata={"tag": "greeting"})
    obs = win.observations
    assert obs[0].metadata == {}
    assert obs[1].metadata == {"tag": "greeting"}


def test_observationwindow_default_max_size_is_50() -> None:
    """Pins ObservationWindow default max_size to 50 (kills int 50->51)."""
    win = ObservationWindow()
    assert win.max_size == 50


# ---------------------------------------------------------------------------
# DualWindow — kills faithfulness mutants
# ---------------------------------------------------------------------------


def _fill(window, n: int, prefix: str = "t") -> None:
    for i in range(n):
        window.add(f"{prefix}{i}")


@pytest.fixture
def dual() -> DualWindow:
    return DualWindow()


def test_dualwindow_default_window_size_is_50(dual: DualWindow) -> None:
    """Pins anchor.max_size and recent.max_size to 50 (kills int 50->51 x2)."""
    assert dual.anchor.max_size == 50
    assert dual.recent.max_size == 50


def test_dualwindow_before_boundary_marked_is_false(dual: DualWindow) -> None:
    """Pins _post_boundary init default to False (kills bool False->True)."""
    assert dual.boundary_marked is False


def test_dualwindow_mark_boundary_sets_marked_true(dual: DualWindow) -> None:
    """Pins mark_boundary() to set _post_boundary True (kills bool True->False)."""
    dual.mark_boundary()
    assert dual.boundary_marked is True


def test_dualwindow_reset_clears_boundary_marked(dual: DualWindow) -> None:
    """Pins reset() to clear _post_boundary to False (kills bool False->True)."""
    dual.mark_boundary()
    dual.reset()
    assert dual.boundary_marked is False


def test_dualwindow_add_routes_to_anchor_before_boundary(dual: DualWindow) -> None:
    dual.add("a")
    dual.add("b")
    assert dual.anchor.texts == ["a", "b"]
    assert dual.recent.texts == []


def test_dualwindow_add_routes_to_recent_after_boundary(dual: DualWindow) -> None:
    dual.mark_boundary()
    dual.add("x")
    dual.add("y")
    assert dual.anchor.texts == []
    assert dual.recent.texts == ["x", "y"]


def test_dualwindow_reset_clears_windows(dual: DualWindow) -> None:
    dual.add("a")
    dual.mark_boundary()
    dual.add("b")
    dual.reset()
    assert len(dual.anchor) == 0
    assert len(dual.recent) == 0
    assert dual.boundary_marked is False
    # After reset, adds should go back to anchor.
    dual.add("c")
    assert dual.anchor.texts == ["c"]
    assert dual.recent.texts == []


def test_dualwindow_has_enough_data_default_threshold_is_3() -> None:
    """With exactly 3 anchor + 3 recent, default threshold of 3 is met.
    Kills int 3->4 (would fail at 4) and both GtE->Lt (3<3 is False)."""
    dw = DualWindow(window_size=10)
    _fill(dw.anchor, 3)
    dw.mark_boundary()
    _fill(dw.recent, 3)
    assert dw.has_enough_data() is True


def test_dualwindow_has_enough_data_anchor_below_threshold() -> None:
    """anchor=2 (<3), recent=3 -> False.
    Kills first-GtE->Lt: 2<3 is True and 3>=3 is True -> mutant returns True.
    """
    dw = DualWindow(window_size=10)
    _fill(dw.anchor, 2)
    dw.mark_boundary()
    _fill(dw.recent, 3)
    assert dw.has_enough_data() is False


def test_dualwindow_has_enough_data_recent_below_threshold() -> None:
    """anchor=3, recent=2 (<3) -> False.
    Kills second-GtE->Lt: 3>=3 is True and 2<3 is True -> mutant returns True.
    """
    dw = DualWindow(window_size=10)
    _fill(dw.anchor, 3)
    dw.mark_boundary()
    _fill(dw.recent, 2)
    assert dw.has_enough_data() is False


def test_dualwindow_has_enough_data_both_below_threshold() -> None:
    dw = DualWindow(window_size=10)
    _fill(dw.anchor, 2)
    dw.mark_boundary()
    _fill(dw.recent, 2)
    assert dw.has_enough_data() is False


def test_dualwindow_has_enough_data_explicit_threshold() -> None:
    dw = DualWindow(window_size=10)
    _fill(dw.anchor, 4)
    dw.mark_boundary()
    _fill(dw.recent, 4)
    # Below explicit threshold
    assert dw.has_enough_data(min_observations=5) is False
    # At exact threshold
    dw.anchor.add("extra_anchor")  # now 5
    dw.recent.add("extra_recent")  # now 5
    assert dw.has_enough_data(min_observations=5) is True
