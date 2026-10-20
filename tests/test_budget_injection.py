"""Budget injection (BATS-style prompt signal) + per-tool call caps. Offline."""

from __future__ import annotations

import subprocess
import sys

import pytest

from showback import BudgetExceeded, BudgetGuard, SpendTracker


def _guard() -> BudgetGuard:
    return BudgetGuard(0.50, SpendTracker())


# -- budget_prompt_snippet --


def test_snippet_fresh_guard_exact_format():
    snippet = _guard().budget_prompt_snippet()
    assert snippet == (
        "Remaining budget: $0.5000 of $0.5000 (100% left). Spend carefully; prefer cheaper actions."
    )


def test_snippet_updates_as_spend_grows():
    guard = _guard()
    before = guard.budget_prompt_snippet()
    guard.tracker.record_tool("search", 0.18)  # $0.18 of $0.50
    after = guard.budget_prompt_snippet()
    assert after != before
    assert "$0.3200" in after
    assert "$0.5000" in after
    assert "(64% left)" in after


def test_snippet_clamps_at_zero_when_over_budget():
    guard = _guard()
    guard.tracker.record_tool("search", 0.75)  # over the $0.50 budget
    snippet = guard.budget_prompt_snippet()
    assert "$0.0000" in snippet
    assert "(0% left)" in snippet


# -- spent_ratio --


def test_spent_ratio_zero_on_fresh_guard():
    assert _guard().spent_ratio == 0.0


def test_spent_ratio_fraction_after_spend():
    guard = _guard()
    guard.tracker.record_tool("search", 0.125)  # $0.125 of $0.50
    assert guard.spent_ratio == pytest.approx(0.25)


def test_spent_ratio_capped_at_one_when_over_budget():
    guard = _guard()
    guard.tracker.record_tool("search", 0.75)  # $0.75 of $0.50
    assert guard.spent_ratio == 1.0


# -- per-tool call caps --


def test_record_tool_under_cap_ok():
    tracker = SpendTracker()
    tracker.set_tool_cap("search", 0.10)
    entry = tracker.record_tool("search", 0.09)
    assert entry.cost_usd == pytest.approx(0.09)
    assert len(tracker.entries) == 1


def test_record_tool_at_exact_cap_ok():
    tracker = SpendTracker()
    tracker.set_tool_cap("search", 0.10)
    tracker.record_tool("search", 0.10)  # cost == cap: allowed (cost <= b)
    assert len(tracker.entries) == 1


def test_record_tool_over_cap_raises_and_appends_nothing():
    tracker = SpendTracker()
    tracker.set_tool_cap("search", 0.10)
    before = len(tracker.entries)
    with pytest.raises(BudgetExceeded) as exc_info:
        tracker.record_tool("search", 0.25)
    assert exc_info.value.budget_usd == pytest.approx(0.10)
    assert exc_info.value.spent_usd == pytest.approx(0.25)
    assert len(tracker.entries) == before  # the rejected call is not recorded


def test_caps_are_per_tool_name():
    tracker = SpendTracker()
    tracker.set_tool_cap("search", 0.10)
    tracker.record_tool("lookup", 1.00)  # no cap on "lookup": fine
    assert len(tracker.entries) == 1
    with pytest.raises(BudgetExceeded):
        tracker.record_tool("search", 0.11)


def test_clear_tool_cap_removes_enforcement():
    tracker = SpendTracker()
    tracker.set_tool_cap("search", 0.10)
    tracker.clear_tool_cap("search")
    tracker.record_tool("search", 1.00)  # cap gone: recorded
    assert len(tracker.entries) == 1


def test_clear_tool_cap_absent_is_noop():
    SpendTracker().clear_tool_cap("never-set")  # must not raise


def test_set_tool_cap_negative_raises():
    with pytest.raises(ValueError):
        SpendTracker().set_tool_cap("search", -0.01)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_set_tool_cap_blank_name_raises(blank: str):
    with pytest.raises(ValueError):
        SpendTracker().set_tool_cap(blank, 0.10)


def test_set_tool_cap_zero_is_a_valid_cap():
    tracker = SpendTracker()
    tracker.set_tool_cap("search", 0.0)
    tracker.record_tool("search", 0.0)  # free calls still allowed
    with pytest.raises(BudgetExceeded):
        tracker.record_tool("search", 0.01)


# -- import order: guard.py imports tracker.py, so tracker.py must not import
#    guard.py at module top (that would cycle). Both orders must work in a
#    fresh interpreter, and the function-local import must fire at runtime. --


@pytest.mark.parametrize("first", ["showback.tracker", "showback.guard", "showback"])
def test_no_import_cycle_in_either_order(first: str):
    code = (
        f"import {first}\n"
        "import showback.guard\n"
        "import showback.tracker\n"
        "t = showback.tracker.SpendTracker()\n"
        "t.set_tool_cap('search', 0.10)\n"
        "try:\n"
        "    t.record_tool('search', 0.25)\n"
        "except showback.guard.BudgetExceeded:\n"
        "    print('cap enforced')\n"
        "else:\n"
        "    raise SystemExit('cap NOT enforced')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "cap enforced" in result.stdout
