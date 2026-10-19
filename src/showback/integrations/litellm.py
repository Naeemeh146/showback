"""LiteLLM integration (optional extra: ``pip install showback[litellm]``).

Drop :class:`SpendGuardCallback` into ``litellm.success_callback`` and every
completed call is recorded in your ledger — with the budget guard tripping
mid-run if a cap is breached::

    import litellm
    from showback import BudgetGuard, SpendTracker
    from showback.integrations.litellm import SpendGuardCallback

    tracker = SpendTracker()
    guard = BudgetGuard(1.00, tracker)
    litellm.success_callback = [SpendGuardCallback(tracker, guard)]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from showback.guard import BudgetGuard
from showback.tracker import SpendTracker


def _usage_value(usage: Any, *names: str) -> int:
    """Read a token count from LiteLLM's usage object or dict."""
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


class SpendGuardCallback:
    """Custom callback for ``litellm.success_callback``.

    Records each successful call and then runs ``guard.check()``, so a
    multi-step agent loop aborts as soon as the budget is hit.
    """

    def __init__(
        self,
        tracker: SpendTracker | None = None,
        guard: BudgetGuard | None = None,
        *,
        tags: dict[str, str] | None = None,
        tag_from_kwargs: Callable[[dict[str, Any]], dict[str, str]] | None = None,
    ) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The LiteLLM integration requires the 'litellm' package: "
                "pip install showback[litellm]"
            ) from exc
        self.tracker = tracker if tracker is not None else SpendTracker()
        self.guard = guard
        self.tags = dict(tags) if tags else {}
        self.tag_from_kwargs = tag_from_kwargs

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """LiteLLM success-callback signature. Records the call, then enforces."""
        model = str(kwargs.get("model", "unknown"))
        usage = getattr(response_obj, "usage", None)
        input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")

        call_tags = dict(self.tags)
        if self.tag_from_kwargs is not None:
            extra = self.tag_from_kwargs(kwargs) or {}
            call_tags.update({str(k): str(v) for k, v in extra.items()})

        self.tracker.record_llm(model, input_tokens, output_tokens, **call_tags)
        if self.guard is not None:
            self.guard.check()
