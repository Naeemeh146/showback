import threading

import pytest

from showback import PricingTable, SpendTracker


def test_record_llm_computes_cost():
    tracker = SpendTracker()
    entry = tracker.record_llm("gpt-4o-mini", 1_000_000, 1_000_000, agent="a1")
    assert entry.kind == "llm"
    assert entry.model == "gpt-4o-mini"
    assert entry.cost_usd == pytest.approx(0.75)
    assert entry.tags == {"agent": "a1"}
    assert entry.id and entry.timestamp


def test_tags_coerced_to_strings():
    tracker = SpendTracker()
    entry = tracker.record_llm("gpt-4o-mini", 10, 10, step=3)
    assert entry.tags == {"step": "3"}


def test_record_tool_uses_explicit_cost():
    tracker = SpendTracker()
    entry = tracker.record_tool("web_search", 0.0025, agent="a1", note="serp call")
    assert entry.kind == "tool"
    assert entry.name == "web_search"
    assert entry.model is None
    assert entry.cost_usd == pytest.approx(0.0025)
    assert entry.note == "serp call"


def test_record_tool_rejects_negative_cost():
    tracker = SpendTracker()
    with pytest.raises(ValueError):
        tracker.record_tool("x", -0.1)


def test_total_cost_sums_kinds():
    tracker = SpendTracker()
    tracker.record_llm("gpt-4o-mini", 1_000_000, 0, agent="a1")  # 0.15
    tracker.record_tool("t", 0.10, agent="a1")
    assert tracker.total_cost() == pytest.approx(0.25)


def test_entries_returns_immutable_snapshot():
    tracker = SpendTracker()
    tracker.record_llm("gpt-4o-mini", 10, 10)
    entries = tracker.entries
    assert isinstance(entries, tuple)
    assert len(entries) == 1
    tracker.record_llm("gpt-4o-mini", 10, 10)
    assert len(entries) == 1  # snapshot unaffected
    assert len(tracker.entries) == 2


def test_cost_for_tags_filters():
    tracker = SpendTracker()
    tracker.record_llm("gpt-4o-mini", 1_000_000, 0, task_id="t1")  # 0.15
    tracker.record_llm("gpt-4o-mini", 1_000_000, 0, task_id="t2")  # 0.15
    assert tracker.cost_for_tags({"task_id": "t1"}) == pytest.approx(0.15)
    assert tracker.cost_for_tags({"task_id": "nope"}) == 0.0


def test_clear_empties_ledger():
    tracker = SpendTracker()
    tracker.record_llm("gpt-4o-mini", 10, 10)
    tracker.clear()
    assert tracker.entries == ()
    assert tracker.total_cost() == 0.0


def test_thread_safe_recording():
    tracker = SpendTracker(PricingTable.from_dict({"m": {"input": 1.0, "output": 1.0}}))
    threads = [
        threading.Thread(target=lambda: tracker.record_llm("m", 1000, 1000)) for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(tracker.entries) == 20
    assert tracker.total_cost() == pytest.approx(20 * 0.002)
