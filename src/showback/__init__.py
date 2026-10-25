"""showback: hard budget enforcement for AI agent spend.

The enforcement layer cost dashboards don't give you: record every LLM and
tool call in a local ledger, refuse to start work that is already over budget,
and abort a run mid-flight when spend crosses a hard cap. Fully offline —
no network calls, no required dependencies.
"""

from showback.attribution import export_csv, export_json, summarize_by_tag
from showback.cascade import (
    CallFn,
    CascadeResult,
    CascadeRouter,
    Scorer,
    keyword_scorer,
    nonempty_scorer,
)
from showback.guard import BudgetAlert, BudgetExceeded, BudgetGuard, BudgetWarning
from showback.loops import LoopDetected, LoopDetector, loop_to_budget_exceeded
from showback.pricing import DEFAULT_PRICES, PricingTable, UnknownModelError
from showback.tracker import LedgerEntry, SpendTracker

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_PRICES",
    "BudgetAlert",
    "BudgetExceeded",
    "BudgetGuard",
    "BudgetWarning",
    "CallFn",
    "CascadeResult",
    "CascadeRouter",
    "LedgerEntry",
    "LoopDetected",
    "LoopDetector",
    "PricingTable",
    "Scorer",
    "SpendTracker",
    "UnknownModelError",
    "__version__",
    "export_csv",
    "export_json",
    "keyword_scorer",
    "loop_to_budget_exceeded",
    "nonempty_scorer",
    "summarize_by_tag",
]
