"""BudgetGuard: budget enforcement for agent spend, with a choice at the cap.

By default the guard is a hard cap: pre-flight and mid-run checks raise
:class:`BudgetExceeded` the moment spend reaches the budget. Two softer
modes are available via ``on_exceed`` for teams that want a signal instead
of a stop, or a human decision instead of an automatic one:

* ``"stop"`` (default) — raise :class:`BudgetExceeded`. The run aborts.
* ``"warn"`` — fire a :class:`BudgetAlert` exactly once (via the ``on_warn``
  callback if given, otherwise Python's ``warnings`` machinery) and let the
  run continue. Every later dollar is still recorded in the ledger.
* ``"ask"`` — call the ``on_ask`` handler with a :class:`BudgetAlert` and
  let its return value decide: ``True`` approves continuing (asked only
  once per guard), ``False`` raises :class:`BudgetExceeded`. With no
  handler, ``"ask"`` behaves like ``"stop"`` — a run is never allowed to
  continue past its budget without an explicit human yes.

Independently of the mode, ``warn_at`` (a fraction of the budget, e.g.
``0.8``) fires one early alert when spend crosses that share, so a
warn-then-stop setup is simply ``warn_at=0.8, on_exceed="stop"``.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from showback.pricing import PricingTable
from showback.tracker import SpendTracker

F = TypeVar("F", bound=Callable[..., Any])

OnExceed = Literal["stop", "warn", "ask"]


class BudgetWarning(UserWarning):
    """Warning category emitted by BudgetGuard in ``warn`` mode / at ``warn_at``."""


@dataclass(frozen=True)
class BudgetAlert:
    """A single budget alert event, passed to ``on_warn`` / ``on_ask`` handlers.

    ``kind`` is ``"warn_threshold"`` (spend crossed the ``warn_at`` share of
    the budget), ``"exceeded"`` (spend reached the budget in ``warn``
    mode), or ``"approval_requested"`` (spend reached the budget in
    ``ask`` mode and a human decision is being requested).
    """

    kind: str
    budget_usd: float
    spent_usd: float
    match_tags: dict[str, str] | None = None

    @property
    def message(self) -> str:
        scope = f" for tags {self.match_tags}" if self.match_tags else ""
        if self.kind == "warn_threshold":
            return (
                f"Budget warning{scope}: spent ${self.spent_usd:.4f} "
                f"of ${self.budget_usd:.4f} budget."
            )
        if self.kind == "approval_requested":
            return (
                f"Budget approval requested{scope}: spent ${self.spent_usd:.4f} "
                f"of ${self.budget_usd:.4f} budget. Human approval required to continue."
            )
        return (
            f"Budget exceeded{scope}: spent ${self.spent_usd:.4f} "
            f"of ${self.budget_usd:.4f} budget. Run continuing in warn mode."
        )


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
    """Enforce a USD budget over a :class:`SpendTracker`, your way.

    Two enforcement points:

    * **Pre-flight** — entering the context manager (or calling a decorated
      function) applies the guard's ``on_exceed`` behaviour immediately if
      the budget is already spent. In ``stop`` mode a run that cannot
      afford its first step never starts.
    * **Mid-run** — call :meth:`check` between agent steps (after each LLM
      or tool call). What happens when spend reaches the cap is chosen by
      ``on_exceed``: ``"stop"`` aborts the run, ``"warn"`` alerts once and
      continues, ``"ask"`` asks a human via the ``on_ask`` handler.

    Pass ``match_tags`` to scope the budget to a slice of a shared tracker,
    e.g. one guard per ``task_id`` while several tasks share one ledger.

    Alerts fired by this guard are kept in :attr:`alerts`, in order, so the
    report for a run can show "warned at $0.40, approved to continue at
    $0.50, finished at $0.73".

    Vocabulary (after BATS, Liu et al., arXiv:2511.17006):

    * **BUDGET** — the hard upper limit in USD this guard enforces. What
      crossing it *does* is chosen by ``on_exceed``.
    * **COST** — realized spend recorded in the ledger. The guard acts when
      COST reaches the BUDGET.
    """

    def __init__(
        self,
        budget_usd: float,
        tracker: SpendTracker | None = None,
        *,
        match_tags: dict[str, str] | None = None,
        pricing: PricingTable | None = None,
        on_exceed: OnExceed = "stop",
        warn_at: float | None = None,
        on_warn: Callable[[BudgetAlert], None] | None = None,
        on_ask: Callable[[BudgetAlert], bool] | None = None,
    ) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        if on_exceed not in ("stop", "warn", "ask"):
            raise ValueError('on_exceed must be one of "stop", "warn", "ask"')
        if warn_at is not None and not 0.0 < warn_at < 1.0:
            raise ValueError("warn_at must be a fraction of the budget between 0 and 1")
        if on_warn is not None and not callable(on_warn):
            raise ValueError("on_warn must be callable")
        if on_ask is not None and not callable(on_ask):
            raise ValueError("on_ask must be callable")
        self.budget_usd = float(budget_usd)
        self.tracker = tracker if tracker is not None else SpendTracker(pricing)
        self.match_tags = {str(k): str(v) for k, v in match_tags.items()} if match_tags else None
        self.on_exceed = on_exceed
        self.warn_at = warn_at
        self.on_warn = on_warn
        self.on_ask = on_ask
        self._alerts: list[BudgetAlert] = []
        self._warned_threshold = False
        self._warned_exceeded = False
        self._asked = False
        #: True once a human has approved continuing past the budget in ask mode.
        self.human_approved_overrun = False

    @property
    def alerts(self) -> tuple[BudgetAlert, ...]:
        """Every alert this guard has fired, in order (threshold, exceeded, approval)."""
        return tuple(self._alerts)

    @property
    def warned(self) -> bool:
        """True once any alert (threshold or exceeded) has fired."""
        return bool(self._alerts)

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

    def _fire_alert(self, kind: str, spent: float) -> BudgetAlert:
        alert = BudgetAlert(
            kind=kind,
            budget_usd=self.budget_usd,
            spent_usd=spent,
            match_tags=dict(self.match_tags) if self.match_tags else None,
        )
        self._alerts.append(alert)
        if self.on_warn is not None:
            self.on_warn(alert)
        else:
            warnings.warn(alert.message, BudgetWarning, stacklevel=3)
        return alert

    def check(self) -> BudgetAlert | None:
        """Apply the guard's policy. Call between steps.

        Returns the :class:`BudgetAlert` fired by this call, if any.
        In ``stop`` mode (and in ``ask`` mode when the human says no, or no
        handler is set) raises :class:`BudgetExceeded` at the cap instead.
        """
        spent = self.spent_usd

        # Early warning: fires once, in every mode, when warn_at is set.
        if (
            self.warn_at is not None
            and not self._warned_threshold
            and spent < self.budget_usd
            and spent >= self.warn_at * self.budget_usd
        ):
            self._warned_threshold = True
            return self._fire_alert("warn_threshold", spent)

        if spent < self.budget_usd:
            return None

        # At or over the cap: the mode decides.
        if self.on_exceed == "stop":
            raise BudgetExceeded(self.budget_usd, spent, self.match_tags)

        if self.on_exceed == "warn":
            if not self._warned_exceeded:
                self._warned_exceeded = True
                return self._fire_alert("exceeded", spent)
            return None

        # on_exceed == "ask": ask once; an approval covers the rest of the run.
        if self.human_approved_overrun:
            return None
        if not self._asked:
            self._asked = True
            alert = BudgetAlert(
                kind="approval_requested",
                budget_usd=self.budget_usd,
                spent_usd=spent,
                match_tags=dict(self.match_tags) if self.match_tags else None,
            )
            self._alerts.append(alert)
            if self.on_ask is not None and self.on_ask(alert):
                self.human_approved_overrun = True
                return alert
        raise BudgetExceeded(self.budget_usd, spent, self.match_tags)

    # -- context manager: `with BudgetGuard(0.50, tracker):` --
    def __enter__(self) -> BudgetGuard:
        self.check()  # pre-flight: applies on_exceed if already over budget
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
