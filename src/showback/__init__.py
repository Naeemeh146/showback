"""showback: hard budget enforcement for AI agent spend.

The enforcement layer cost dashboards don't give you: record every LLM and
tool call in a local ledger, refuse to start work that is already over budget,
and abort a run mid-flight when spend crosses a hard cap. Fully offline —
no network calls, no required dependencies.
"""

from showback.attribution import export_csv, export_json, summarize_by_tag
from showback.guard import BudgetExceeded, BudgetGuard
from showback.pricing import DEFAULT_PRICES, PricingTable, UnknownModelError
from showback.tracker import LedgerEntry, SpendTracker

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PRICES",
    "BudgetExceeded",
    "BudgetGuard",
    "LedgerEntry",
    "PricingTable",
    "SpendTracker",
    "UnknownModelError",
    "__version__",
    "export_csv",
    "export_json",
    "summarize_by_tag",
]
