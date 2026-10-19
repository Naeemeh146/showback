"""Simulated multi-step agent run with a hard budget.

Run from the repo root::

    python examples/agent_loop.py

The agent plans 6 steps but only has a $0.004 budget. After each step the
guard is checked, so the run aborts mid-flight instead of drifting over
budget. No network calls — token counts are simulated.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_spend_guard import (  # noqa: E402
    BudgetExceeded,
    BudgetGuard,
    SpendTracker,
    export_csv,
    export_json,
    summarize_by_tag,
)

MODEL = "gpt-4o-mini"  # $0.15 / $0.60 per 1M tokens
BUDGET_USD = 0.004
PLANNED_STEPS = 6

# Fixed, deterministic "simulated" usage per step: (input_tokens, output_tokens, tool, tool_cost)
STEPS = [
    (4000, 800, "web_search", 0.0005),
    (4200, 900, "web_search", 0.0005),
    (3800, 1200, "calculator", 0.0002),
    (4500, 700, "web_search", 0.0005),
    (4100, 850, "code_exec", 0.0010),
    (3900, 950, "web_search", 0.0005),
]


def main() -> None:
    tracker = SpendTracker()
    guard = BudgetGuard(BUDGET_USD, tracker)

    print(f"Agent starting: {PLANNED_STEPS} planned steps, budget ${BUDGET_USD:.4f}\n")

    completed = 0
    try:
        with guard:  # pre-flight: refuse to start if already over budget
            for i, (in_tok, out_tok, tool, tool_cost) in enumerate(STEPS, start=1):
                guard.check()  # mid-run enforcement point between steps
                entry = tracker.record_llm(
                    MODEL, in_tok, out_tok, agent="demo-agent", task_id="demo-1", step=str(i)
                )
                tracker.record_tool(
                    tool,
                    tool_cost,
                    agent="demo-agent",
                    task_id="demo-1",
                    step=str(i),
                    note=f"step {i} tool call",
                )
                completed += 1
                print(
                    f"step {i}: llm ${entry.cost_usd:.6f} + tool ${tool_cost:.6f} "
                    f"| running total ${tracker.total_cost():.6f}"
                )
    except BudgetExceeded as exc:
        print(f"\nABORTED mid-run after {completed}/{PLANNED_STEPS} steps.")
        print(f"BudgetExceeded: {exc}")

    print("\n--- ledger ---")
    for entry in tracker.entries:
        label = entry.model if entry.kind == "llm" else f"tool:{entry.name}"
        print(f"{entry.timestamp} {entry.kind:4} {label:22} ${entry.cost_usd:.6f} {entry.tags}")

    print("\n--- spend by step (showback) ---")
    for step, cost in summarize_by_tag(tracker, "step").items():
        print(f"  step {step}: ${cost:.6f}")
    print(f"\nTOTAL: ${tracker.total_cost():.6f} of ${BUDGET_USD:.4f} budget")

    out_dir = Path(tempfile.mkdtemp(prefix="spend-guard-demo-"))
    json_path = export_json(tracker, out_dir / "ledger.json")
    csv_path = export_csv(tracker, out_dir / "ledger.csv")
    print(f"\nLedger exported to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
