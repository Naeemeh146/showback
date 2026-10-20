"""Tests for showback.loops: LoopDetector and loop_to_budget_exceeded."""

from __future__ import annotations

import pytest

from showback.guard import BudgetExceeded, BudgetGuard
from showback.loops import LoopDetected, LoopDetector, loop_to_budget_exceeded
from showback.tracker import SpendTracker


def test_exact_repeat_triggers_at_max_repeats() -> None:
    detector = LoopDetector()
    detector.observe("search_web", "q=cats")
    detector.observe("search_web", "q=cats")
    with pytest.raises(LoopDetected) as exc_info:
        detector.observe("search_web", "q=cats")
    assert exc_info.value.kind == "repeat"
    assert exc_info.value.pattern == ("search_web|q=cats",)
    assert len(exc_info.value.history) == 3


def test_two_repeats_do_not_trigger_default() -> None:
    detector = LoopDetector()
    detector.observe("search_web", "q=cats")
    detector.observe("search_web", "q=cats")
    assert detector.history == ("search_web|q=cats", "search_web|q=cats")


def test_cycle_abab_triggers() -> None:
    detector = LoopDetector(cycle_lengths=(2,))
    detector.observe("a", "1")
    detector.observe("b", "2")
    detector.observe("a", "1")
    with pytest.raises(LoopDetected) as exc_info:
        detector.observe("b", "2")
    assert exc_info.value.kind == "cycle"
    assert exc_info.value.pattern == ("a|1", "b|2")
    assert exc_info.value.history == ["a|1", "b|2", "a|1", "b|2"]


def test_no_false_positive_on_abcd() -> None:
    detector = LoopDetector()
    for action in ("a", "b", "c", "d"):
        detector.observe(action, "x")
    assert detector.history == ("a|x", "b|x", "c|x", "d|x")


def test_normalization_counts_as_repeat() -> None:
    detector = LoopDetector()
    detector.observe("Search ", "  Q=Cats  ")
    detector.observe("search", "q=cats")
    with pytest.raises(LoopDetected) as exc_info:
        detector.observe("SEARCH", "  Q=CATS")
    assert exc_info.value.kind == "repeat"
    assert exc_info.value.pattern == ("search|q=cats",)


def test_reset_clears_history() -> None:
    detector = LoopDetector()
    detector.observe("ping", "")
    detector.observe("ping", "")
    detector.reset()
    assert detector.history == ()
    # Same actions can repeat again after reset without triggering.
    detector.observe("ping", "")
    detector.observe("ping", "")
    assert detector.history == ("ping|", "ping|")


def test_message_is_human_readable() -> None:
    detector = LoopDetector()
    detector.observe("search_web", "q=dogs")
    detector.observe("search_web", "q=dogs")
    try:
        detector.observe("search_web", "q=dogs")
    except LoopDetected as exc:
        message = str(exc)
    else:
        pytest.fail("expected LoopDetected")
    assert "Loop detected (repeat)" in message
    assert "'search_web|q=dogs' x3" in message


def test_custom_max_repeats_two() -> None:
    detector = LoopDetector(max_repeats=2)
    detector.observe("ping", "")
    with pytest.raises(LoopDetected) as exc_info:
        detector.observe("ping", "")
    assert exc_info.value.kind == "repeat"


def test_cycle_length_one_defers_to_repeat() -> None:
    # A length-1 "cycle" of identical actions is a repeat-in-progress, so it
    # never trips the cycle check — it is left to the repeat check.
    detector = LoopDetector(max_repeats=99, cycle_lengths=(1,))
    detector.observe("a", "1")
    detector.observe("a", "1")
    detector.observe("a", "1")
    assert detector.history == ("a|1", "a|1", "a|1")

    # With a small max_repeats, the same stream trips the repeat detector.
    detector = LoopDetector(max_repeats=2, cycle_lengths=(1,))
    detector.observe("a", "1")
    with pytest.raises(LoopDetected) as exc_info:
        detector.observe("a", "1")
    assert exc_info.value.kind == "repeat"


def test_repeat_checked_before_cycle() -> None:
    # Exact repeat should win even when a length-1 cycle would also match.
    detector = LoopDetector(max_repeats=2, cycle_lengths=(1,))
    detector.observe("a", "1")
    with pytest.raises(LoopDetected) as exc_info:
        detector.observe("a", "1")
    assert exc_info.value.kind == "repeat"


def test_invalid_constructor_args() -> None:
    with pytest.raises(ValueError):
        LoopDetector(max_repeats=1)
    with pytest.raises(ValueError):
        LoopDetector(cycle_lengths=(0,))


def test_loop_to_budget_exceeded() -> None:
    tracker = SpendTracker()
    tracker.record_tool("web", 0.50, run="demo")
    guard = BudgetGuard(10.0, tracker, match_tags={"run": "demo"})
    detector = LoopDetector()
    detector.observe("a", "")
    detector.observe("a", "")
    try:
        detector.observe("a", "")
    except LoopDetected as exc:
        converted = loop_to_budget_exceeded(exc, guard)
    else:
        pytest.fail("expected LoopDetected")
    assert isinstance(converted, BudgetExceeded)
    assert converted.budget_usd == 10.0
    assert converted.spent_usd == pytest.approx(0.50)
    assert converted.match_tags == {"run": "demo"}
