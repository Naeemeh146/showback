"""Tests for showback.cascade (FrugalGPT-style model cascading). Offline only."""

from __future__ import annotations

from functools import partial

import pytest

from showback import BudgetExceeded, BudgetGuard, PricingTable, SpendTracker
from showback.cascade import (
    CascadeResult,
    CascadeRouter,
    keyword_scorer,
    nonempty_scorer,
)


def _tracker() -> SpendTracker:
    pricing = PricingTable(
        {
            "cheap": {"input": 1.0, "output": 2.0},
            "mid": {"input": 10.0, "output": 20.0},
            "strong": {"input": 100.0, "output": 200.0},
        }
    )
    return SpendTracker(pricing)


def _call_fn_factory(answers: dict[str, tuple[str, int, int]]):
    """Build a fake call_fn(model, prompt) -> (answer, in_tokens, out_tokens)."""
    seen: list[str] = []

    def call_fn(model: str, prompt: str) -> tuple[str, int, int]:
        seen.append(model)
        return answers[model]

    call_fn.seen = seen  # type: ignore[attr-defined]
    return call_fn


def _pass(prompt: str, answer: str) -> float:
    return 1.0


def _fail(prompt: str, answer: str) -> float:
    return 0.0


# --- constructor validation -------------------------------------------------


def test_constructor_rejects_empty_tiers():
    with pytest.raises(ValueError, match="non-empty"):
        CascadeRouter([], _tracker())


def test_constructor_rejects_bad_model_names():
    for bad in ("", "   ", 123, None):
        with pytest.raises(ValueError, match="non-empty string"):
            CascadeRouter([(bad, _pass, 0.5)], _tracker())  # type: ignore[list-item]


def test_constructor_rejects_non_callable_scorer():
    with pytest.raises(ValueError, match="callable"):
        CascadeRouter([("cheap", "not-a-scorer", 0.5)], _tracker())  # type: ignore[list-item]


def test_constructor_rejects_bad_threshold():
    for bad in (-0.1, 1.1, "0.5", None, True):
        with pytest.raises(ValueError, match=r"threshold"):
            CascadeRouter([("cheap", _pass, bad)], _tracker())  # type: ignore[list-item]


def test_constructor_rejects_malformed_tier_tuple():
    with pytest.raises(ValueError, match="must be a"):
        CascadeRouter([("cheap", _pass)], _tracker())  # type: ignore[list-item]


def test_constructor_accepts_threshold_boundaries():
    router = CascadeRouter([("cheap", _pass, 0.0), ("strong", _pass, 1.0)], _tracker())
    assert len(router.tiers) == 2


# --- routing behaviour ------------------------------------------------------


def test_accepts_cheapest_tier_when_scorer_passes():
    tracker = _tracker()
    router = CascadeRouter([("cheap", _pass, 0.5), ("strong", _pass, 0.5)], tracker)
    call_fn = _call_fn_factory({"cheap": ("cheap answer", 100, 50)})
    result = router.route("what is 2+2?", call_fn, task_id="t1")

    assert result.model == "cheap"
    assert result.answer == "cheap answer"
    assert result.tiers_tried == ["cheap"]
    assert result.scores == [1.0]
    assert call_fn.seen == ["cheap"]
    expected = tracker.pricing.cost("cheap", 100, 50)
    assert result.total_cost == pytest.approx(expected)
    assert tracker.total_cost() == pytest.approx(expected)
    entry = tracker.entries[0]
    assert entry.tags == {"task_id": "t1"}
    assert entry.note == "cascade tier=cheap score=1.0"


def test_escalates_exactly_one_tier_and_records_both_costs():
    tracker = _tracker()
    router = CascadeRouter(
        [("cheap", _fail, 0.5), ("strong", _pass, 0.5)],
        tracker,
    )
    call_fn = _call_fn_factory(
        {
            "cheap": ("cheap guess", 100, 50),
            "strong": ("strong answer", 200, 100),
        }
    )
    result = router.route("hard question", call_fn)

    assert result.model == "strong"
    assert result.answer == "strong answer"
    assert result.tiers_tried == ["cheap", "strong"]
    assert result.scores == [0.0, 1.0]
    assert call_fn.seen == ["cheap", "strong"]
    cheap_cost = tracker.pricing.cost("cheap", 100, 50)
    strong_cost = tracker.pricing.cost("strong", 200, 100)
    assert result.total_cost == pytest.approx(cheap_cost + strong_cost)
    assert tracker.total_cost() == pytest.approx(cheap_cost + strong_cost)


def test_fallback_accepts_last_tier_despite_low_score():
    tracker = _tracker()
    router = CascadeRouter(
        [("cheap", _fail, 0.5), ("mid", _fail, 0.5), ("strong", _fail, 0.5)],
        tracker,
    )
    call_fn = _call_fn_factory(
        {
            "cheap": ("c", 10, 5),
            "mid": ("m", 20, 10),
            "strong": ("best effort", 30, 15),
        }
    )
    result = router.route("impossible question", call_fn)

    assert result.model == "strong"
    assert result.answer == "best effort"
    assert result.tiers_tried == ["cheap", "mid", "strong"]
    assert result.scores == [0.0, 0.0, 0.0]
    assert call_fn.seen == ["cheap", "mid", "strong"]


def test_guard_raises_before_any_call_when_already_over_budget():
    tracker = _tracker()
    tracker.record_llm("strong", 10_000_000, 10_000_000)  # $3000, past the cap
    guard = BudgetGuard(1.0, tracker)
    router = CascadeRouter([("cheap", _pass, 0.5)], tracker, guard)
    call_fn = _call_fn_factory({"cheap": ("x", 1, 1)})
    with pytest.raises(BudgetExceeded):
        router.route("anything", call_fn)
    assert call_fn.seen == []  # no model was ever called
    assert len(tracker.entries) == 1  # only the pre-fill, no cascade records


def test_guard_mid_cascade_aborts_before_next_tier():
    tracker = _tracker()
    cheap_cost = tracker.pricing.cost("cheap", 100_000, 0)
    guard = BudgetGuard(cheap_cost, tracker)
    router = CascadeRouter(
        [("cheap", _fail, 0.9), ("strong", _pass, 0.5)],
        tracker,
        guard,
    )
    call_fn = _call_fn_factory(
        {
            "cheap": ("cheap guess", 100_000, 0),
            "strong": ("strong answer", 10, 5),
        }
    )
    with pytest.raises(BudgetExceeded):
        router.route("hard question", call_fn)
    assert call_fn.seen == ["cheap"]  # strong tier never ran


def test_scorer_returning_out_of_range_raises_value_error():
    tracker = _tracker()
    router = CascadeRouter([("cheap", lambda p, a: 1.5, 0.5)], tracker)
    call_fn = _call_fn_factory({"cheap": ("x", 1, 1)})
    with pytest.raises(ValueError, match="out of \\[0, 1\\]"):
        router.route("q", call_fn)


def test_call_fn_exception_propagates():
    tracker = _tracker()
    router = CascadeRouter([("cheap", _pass, 0.5), ("strong", _pass, 0.5)], tracker)

    def boom(model: str, prompt: str) -> tuple[str, int, int]:
        raise RuntimeError("upstream exploded")

    with pytest.raises(RuntimeError, match="upstream exploded"):
        router.route("q", boom)
    assert len(tracker.entries) == 0  # failed call is not recorded


def test_result_fields_populated_correctly():
    tracker = _tracker()
    router = CascadeRouter(
        [("cheap", lambda p, a: 0.7, 0.7), ("strong", _pass, 0.5)],
        tracker,
    )
    call_fn = _call_fn_factory({"cheap": ("ok", 10, 5)})
    result = router.route("q", call_fn)
    assert isinstance(result, CascadeResult)
    assert result.answer == "ok"
    assert result.model == "cheap"
    assert result.tiers_tried == ["cheap"]
    assert result.scores == [pytest.approx(0.7)]  # boundary: score == threshold accepts
    assert result.total_cost == pytest.approx(tracker.pricing.cost("cheap", 10, 5))


def test_total_cost_excludes_prior_tracker_spend():
    tracker = _tracker()
    tracker.record_llm("strong", 1_000_000, 1_000_000)
    before = tracker.total_cost()
    router = CascadeRouter([("cheap", _pass, 0.5)], tracker)
    call_fn = _call_fn_factory({"cheap": ("x", 1_000, 500)})
    result = router.route("q", call_fn)
    assert result.total_cost == pytest.approx(tracker.pricing.cost("cheap", 1_000, 500))
    assert tracker.total_cost() == pytest.approx(before + result.total_cost)


# --- built-in scorers -------------------------------------------------------


def test_nonempty_scorer():
    assert nonempty_scorer("p", "some answer") == 1.0
    assert nonempty_scorer("p", "   ") == 0.0
    assert nonempty_scorer("p", "") == 0.0
    assert nonempty_scorer("p", " hi ", min_length=2) == 1.0
    assert nonempty_scorer("p", "h", min_length=2) == 0.0


def test_nonempty_scorer_usable_directly_as_scorer():
    tracker = _tracker()
    router = CascadeRouter([("cheap", nonempty_scorer, 0.5)], tracker)
    call_fn = _call_fn_factory({"cheap": ("yes", 1, 1)})
    result = router.route("q", call_fn)
    assert result.scores == [1.0]


def test_nonempty_scorer_via_partial_with_min_length():
    tracker = _tracker()
    router = CascadeRouter([("cheap", partial(nonempty_scorer, min_length=10), 0.5)], tracker)
    call_fn = _call_fn_factory({"cheap": ("short", 1, 1)})
    result = router.route("q", call_fn)  # single tier is the fallback, accepted anyway
    assert result.scores == [0.0]
    assert result.tiers_tried == ["cheap"]


def test_keyword_scorer():
    assert keyword_scorer("p", "the cat sat on the mat", ["cat", "mat"]) == 1.0
    assert keyword_scorer("p", "the cat sat", ["cat", "mat"]) == 0.5
    assert keyword_scorer("p", "nothing here", ["cat", "mat"]) == 0.0
    assert keyword_scorer("p", "CAT and MAT", ("cat", "mat")) == 1.0  # case-insensitive


def test_keyword_scorer_rejects_empty_keywords():
    with pytest.raises(ValueError, match="non-empty"):
        keyword_scorer("p", "some answer", [])
    with pytest.raises(ValueError, match="non-empty"):
        keyword_scorer("p", "some answer", ())


def test_keyword_scorer_via_partial_as_scorer():
    tracker = _tracker()
    scorer = partial(keyword_scorer, keywords=["refund", "policy"])
    router = CascadeRouter([("cheap", scorer, 0.5), ("strong", _pass, 0.5)], tracker)
    call_fn = _call_fn_factory(
        {
            "cheap": ("no info available", 10, 10),
            "strong": ("per our refund policy", 10, 10),
        }
    )
    result = router.route("what is the refund policy?", call_fn)
    assert result.model == "strong"
    assert result.scores == [0.0, 1.0]
