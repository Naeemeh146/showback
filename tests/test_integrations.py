"""Tests for the optional LiteLLM integration.

litellm is never a required dependency: the module must import fine without
it, and constructing the callback must raise a helpful error. A fake
``litellm`` module stands in for the positive path so these tests stay
offline.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from showback import BudgetExceeded, BudgetGuard, SpendTracker


def _block_litellm(monkeypatch):
    """Simulate litellm not being installed."""
    monkeypatch.setitem(sys.modules, "litellm", None)


def _fake_litellm(monkeypatch):
    """Simulate litellm being installed (we never call into it)."""
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))


def _kwargs(model="gpt-4o-mini"):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def _response(prompt_tokens=1000, completion_tokens=500, as_dict=False):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if as_dict:
        return SimpleNamespace(usage=usage)
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    )


def test_missing_litellm_raises_helpful_error(monkeypatch):
    _block_litellm(monkeypatch)
    from showback.integrations.litellm import SpendGuardCallback

    with pytest.raises(ImportError, match="pip install showback\\[litellm\\]"):
        SpendGuardCallback(SpendTracker())


def test_callback_records_call(monkeypatch):
    _fake_litellm(monkeypatch)
    from showback.integrations.litellm import SpendGuardCallback

    tracker = SpendTracker()
    cb = SpendGuardCallback(tracker, tags={"agent": "researcher"})
    cb.log_success_event(_kwargs(), _response(), None, None)

    entries = tracker.entries
    assert len(entries) == 1
    assert entries[0].model == "gpt-4o-mini"
    assert entries[0].input_tokens == 1000
    assert entries[0].output_tokens == 500
    # 1000/1e6*0.15 + 500/1e6*0.60 = 0.00045
    assert entries[0].cost_usd == pytest.approx(0.00045)
    assert entries[0].tags == {"agent": "researcher"}


def test_callback_handles_dict_usage(monkeypatch):
    _fake_litellm(monkeypatch)
    from showback.integrations.litellm import SpendGuardCallback

    tracker = SpendTracker()
    cb = SpendGuardCallback(tracker)
    cb.log_success_event(_kwargs(), _response(as_dict=True), None, None)
    assert tracker.entries[0].input_tokens == 1000


def test_callback_enforces_guard_mid_run(monkeypatch):
    _fake_litellm(monkeypatch)
    from showback.integrations.litellm import SpendGuardCallback

    tracker = SpendTracker()
    guard = BudgetGuard(0.0001, tracker)  # less than one call costs
    cb = SpendGuardCallback(tracker, guard)
    with pytest.raises(BudgetExceeded):
        cb.log_success_event(_kwargs(), _response(), None, None)


def test_tag_from_kwargs(monkeypatch):
    _fake_litellm(monkeypatch)
    from showback.integrations.litellm import SpendGuardCallback

    tracker = SpendTracker()
    cb = SpendGuardCallback(
        tracker,
        tag_from_kwargs=lambda kw: {"model_used": kw["model"]},
    )
    cb.log_success_event(_kwargs("gpt-4o"), _response(), None, None)
    assert tracker.entries[0].tags == {"model_used": "gpt-4o"}
