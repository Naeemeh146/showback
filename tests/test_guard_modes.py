"""Tests for BudgetGuard's on_exceed modes: stop / warn / ask, plus warn_at."""

import warnings

import pytest

from showback import BudgetAlert, BudgetExceeded, BudgetGuard, BudgetWarning, SpendTracker


def _spend(tracker: SpendTracker, calls: int = 1) -> None:
    for _ in range(calls):
        # $0.75 per call at default gpt-4o-mini pricing
        tracker.record_llm("gpt-4o-mini", 1_000_000, 1_000_000)


def test_default_mode_is_stop():
    tracker = SpendTracker()
    guard = BudgetGuard(1.00, tracker)
    assert guard.on_exceed == "stop"
    _spend(tracker, 2)  # $1.50
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_warn_mode_does_not_raise_and_alerts_once():
    tracker = SpendTracker()
    seen: list[BudgetAlert] = []
    guard = BudgetGuard(1.00, tracker, on_exceed="warn", on_warn=seen.append)
    _spend(tracker, 2)  # $1.50, over budget
    alert = guard.check()
    assert alert is not None
    assert alert.kind == "exceeded"
    assert alert.spent_usd == pytest.approx(1.50)
    assert seen == [alert]
    # Repeated checks do not spam: no new alert, still no raise.
    assert guard.check() is None
    assert guard.check() is None
    assert len(seen) == 1
    assert guard.warned is True
    assert guard.alerts == (alert,)


def test_warn_mode_without_callback_uses_warnings_module():
    tracker = SpendTracker()
    guard = BudgetGuard(1.00, tracker, on_exceed="warn")
    _spend(tracker, 2)
    with pytest.warns(BudgetWarning, match="Budget exceeded"):
        guard.check()
    # ...and only once.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        guard.check()


def test_warn_mode_context_manager_allows_over_budget_entry():
    tracker = SpendTracker()
    _spend(tracker, 2)  # $1.50 already spent
    seen: list[BudgetAlert] = []
    with BudgetGuard(1.00, tracker, on_exceed="warn", on_warn=seen.append) as guard:
        assert guard.spent_usd == pytest.approx(1.50)
    assert len(seen) == 1


def test_warn_at_fires_once_before_the_cap_then_stop_still_stops():
    tracker = SpendTracker()
    seen: list[BudgetAlert] = []
    guard = BudgetGuard(1.00, tracker, warn_at=0.7, on_warn=seen.append)  # warn at $0.70
    guard.check()  # nothing spent: no alert
    _spend(tracker, 1)  # $0.75 — past 70%, under cap
    alert = guard.check()
    assert alert is not None and alert.kind == "warn_threshold"
    assert guard.check() is None  # threshold warning fires only once
    _spend(tracker, 1)  # $1.50 — over cap, default mode stops
    with pytest.raises(BudgetExceeded):
        guard.check()
    assert len(seen) == 1  # only the threshold alert; stop raises, does not alert


def test_warn_at_plus_warn_mode_fires_threshold_then_exceeded():
    tracker = SpendTracker()
    seen: list[BudgetAlert] = []
    guard = BudgetGuard(1.00, tracker, on_exceed="warn", warn_at=0.7, on_warn=seen.append)
    _spend(tracker, 1)  # $0.75
    guard.check()
    _spend(tracker, 1)  # $1.50
    guard.check()
    assert [a.kind for a in seen] == ["warn_threshold", "exceeded"]


def test_ask_mode_approved_continues_and_asks_only_once():
    tracker = SpendTracker()
    asked: list[BudgetAlert] = []

    def approve(alert: BudgetAlert) -> bool:
        asked.append(alert)
        return True

    guard = BudgetGuard(1.00, tracker, on_exceed="ask", on_ask=approve)
    _spend(tracker, 2)  # $1.50
    alert = guard.check()
    assert alert is not None and alert.kind == "approval_requested"
    assert guard.human_approved_overrun is True
    # Later checks do not ask again.
    assert guard.check() is None
    assert len(asked) == 1


def test_ask_mode_denied_stops():
    tracker = SpendTracker()
    guard = BudgetGuard(1.00, tracker, on_exceed="ask", on_ask=lambda alert: False)
    _spend(tracker, 2)
    with pytest.raises(BudgetExceeded):
        guard.check()
    assert guard.human_approved_overrun is False


def test_ask_mode_without_handler_stops_safely():
    tracker = SpendTracker()
    guard = BudgetGuard(1.00, tracker, on_exceed="ask")  # no on_ask handler
    _spend(tracker, 2)
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_invalid_mode_and_warn_at_rejected():
    with pytest.raises(ValueError):
        BudgetGuard(1.00, SpendTracker(), on_exceed="shrug")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BudgetGuard(1.00, SpendTracker(), warn_at=0)
    with pytest.raises(ValueError):
        BudgetGuard(1.00, SpendTracker(), warn_at=1.0)
    with pytest.raises(ValueError):
        BudgetGuard(1.00, SpendTracker(), warn_at=1.5)
