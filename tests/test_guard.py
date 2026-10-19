import pytest

from agent_spend_guard import BudgetExceeded, BudgetGuard, SpendTracker


def _tracker_with_spend(amount_calls: int = 1) -> SpendTracker:
    tracker = SpendTracker()
    for _ in range(amount_calls):
        # $0.75 per call at default gpt-4o-mini pricing
        tracker.record_llm("gpt-4o-mini", 1_000_000, 1_000_000)
    return tracker


def test_context_manager_allows_under_budget():
    with BudgetGuard(10.0, _tracker_with_spend()) as guard:
        assert guard.remaining_usd == pytest.approx(9.25)


def test_context_manager_pref_light_refuses_over_budget():
    tracker = _tracker_with_spend(2)  # $1.50 spent
    with pytest.raises(BudgetExceeded) as exc_info:
        with BudgetGuard(1.00, tracker):
            pass
    assert exc_info.value.spent_usd == pytest.approx(1.50)
    assert exc_info.value.budget_usd == pytest.approx(1.00)
    assert "$1.5000" in str(exc_info.value)


def test_check_raises_mid_run():
    guard = BudgetGuard(1.00, SpendTracker())
    guard.check()  # nothing spent yet: fine
    guard.tracker.record_llm("gpt-4o-mini", 1_000_000, 1_000_000)  # $0.75
    guard.check()  # still fine
    guard.tracker.record_llm("gpt-4o-mini", 1_000_000, 1_000_000)  # $1.50 total
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_check_trips_exactly_at_budget():
    guard = BudgetGuard(0.75, SpendTracker())
    guard.tracker.record_llm("gpt-4o-mini", 1_000_000, 1_000_000)  # exactly $0.75
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_decorator_blocks_call_when_over_budget():
    guard = BudgetGuard(1.00, _tracker_with_spend(2))
    calls = []

    @guard
    def run():
        calls.append(1)

    with pytest.raises(BudgetExceeded):
        run()
    assert calls == []


def test_decorator_allows_call_under_budget():
    guard = BudgetGuard(10.00, SpendTracker())

    @guard
    def run(x):
        return x * 2

    assert run(21) == 42
    assert run.__name__ == "run"  # functools.wraps preserved


def test_scoped_guards_share_one_ledger():
    tracker = SpendTracker()
    guard_a = BudgetGuard(1.00, tracker, match_tags={"task_id": "a"})
    guard_b = BudgetGuard(1.00, tracker, match_tags={"task_id": "b"})
    tracker.record_llm("gpt-4o-mini", 1_000_000, 1_000_000, task_id="a")  # $0.75
    tracker.record_llm("gpt-4o-mini", 1_000_000, 1_000_000, task_id="a")  # $1.50 on a
    with pytest.raises(BudgetExceeded) as exc_info:
        guard_a.check()
    assert exc_info.value.match_tags == {"task_id": "a"}
    guard_b.check()  # b's slice is empty: fine
    assert guard_b.spent_usd == 0.0


def test_remaining_never_negative():
    guard = BudgetGuard(0.10, _tracker_with_spend())
    assert guard.remaining_usd == 0.0


def test_non_positive_budget_rejected():
    with pytest.raises(ValueError):
        BudgetGuard(0, SpendTracker())
    with pytest.raises(ValueError):
        BudgetGuard(-5, SpendTracker())


def test_guard_creates_own_tracker_when_omitted():
    guard = BudgetGuard(5.0)
    assert isinstance(guard.tracker, SpendTracker)
    guard.tracker.record_llm("gpt-4o-mini", 1_000_000, 0)
    assert guard.spent_usd == pytest.approx(0.15)
