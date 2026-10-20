"""BudgetGuard: hard caps for agent spend. Pre-flight and mid-run enforcement."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from showback.pricing import PricingTable
from showback.tracker import SpendTracker

F = TypeVar("F", bound=Callable[..., Any])


class BudgetExceeded(Exception):
    """Raised when spend reaches or crosses a hard budget cap."""

    def __init__(
        self,
        budget_usd: float,
        spent_usd: float,
        match_tags: dict[str, str] | None = None,
    ) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self.match_tags = dict(match_tags) if match_tags else None
        scope = f" for tags {self.match_tags}" if self.match_tags else ""
        super().__init__(
            f"Budget exceeded{scope}: spent ${spent_usd:.4f} of ${budget_usd:.4f} budget. "
            "Run aborted."
        )


class BudgetGuard:
    """Enforce a hard USD budget over a :class:`SpendTracker`.

    Two enforcement points:

    * **Pre-flight** — entering the context manager (or calling a decorated
      function) raises :class:`BudgetExceeded` immediately if the budget is
      already spent. A run that cannot afford its first step never starts.
    * **Mid-run** — call :meth:`check` between agent steps (after each LLM or
      tool call). The moment spend reaches the cap, the run aborts instead of
      drifting on.

    Pass ``match_tags`` to scope the budget to a slice of a shared tracker,
    e.g. one guard per ``task_id`` while several tasks share one ledger.

    Vocabulary (after BATS, Liu et al., arXiv:2511.17006):

    * **BUDGET** — the hard upper limit in USD this guard enforces. Crossing
      it aborts the run.
    * **COST** — realized spend recorded in the ledger. The guard trips when
      COST reaches the BUDGET.
    """

    def __init__(
        self,
        budget_usd: float,
        tracker: SpendTracker | None = None,
        *,
        match_tags: dict[str, str] | None = None,
        pricing: PricingTable | None = None,
    ) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        self.budget_usd = float(budget_usd)
        self.tracker = tracker if tracker is not None else SpendTracker(pricing)
        self.match_tags = {str(k): str(v) for k, v in match_tags.items()} if match_tags else None

    @property
    def spent_usd(self) -> float:
        """Spend counted against this guard's budget (scoped by match_tags)."""
        if self.match_tags:
            return self.tracker.cost_for_tags(self.match_tags)
        return self.tracker.total_cost()

    @property
    def remaining_usd(self) -> float:
        """Budget left before the cap trips. Never negative."""
        return max(0.0, round(self.budget_usd - self.spent_usd, 6))

    @property
    def spent_ratio(self) -> float:
        """Fraction of the BUDGET consumed by realized COST, capped at 1.0.

        This is the continuous signal the BATS budget tracker feeds back to
        the agent (BATS, Liu et al., arXiv:2511.17006): 0.0 on a fresh run,
        1.0 once the hard upper limit is reached or crossed.
        """
        return min(1.0, self.spent_usd / self.budget_usd)

    def budget_prompt_snippet(self) -> str:
        """One-line budget signal to inject into the agent's prompt each step.

        A lightweight, always-on reminder of the remaining budget, in the
        spirit of the BATS budget tracker (BATS, Liu et al.,
        arXiv:2511.17006): a continuous signal that nudges the agent to
        prefer cheaper actions as COST approaches the BUDGET.
        """
        remaining = self.remaining_usd
        percent = int(remaining / self.budget_usd * 100)
        return (
            f"Remaining budget: ${remaining:.4f} of ${self.budget_usd:.4f} "
            f"({percent}% left). Spend carefully; prefer cheaper actions."
        )

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` if the cap is reached. Call between steps."""
        spent = self.spent_usd
        if spent >= self.budget_usd:
            raise BudgetExceeded(self.budget_usd, spent, self.match_tags)

    # -- context manager: `with BudgetGuard(0.50, tracker):` --
    def __enter__(self) -> BudgetGuard:
        self.check()  # pre-flight: refuse to start an already-busted run
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False  # never swallow exceptions

    # -- decorator: `@BudgetGuard(0.50, tracker)` on a function --
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.check()  # pre-flight before the call
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]
