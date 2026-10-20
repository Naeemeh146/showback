# showback

[![PyPI version](https://img.shields.io/pypi/v/showback)](https://pypi.org/project/showback/)
[![Python versions](https://img.shields.io/pypi/pyversions/showback)](https://pypi.org/project/showback/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Naeemeh146/showback/blob/main/LICENSE)

Hard budgets for AI agents, with the receipts.

Every observability tool logs what your agents *already spent* — `showback` is the
layer that **stops the run** when a budget is breached, and hands finance the
itemized bill: refuse to start an over-budget run (pre-flight), abort it mid-flight
between steps, and attribute every dollar to agent, task, team, or customer tags —
exportable as JSON/CSV for showback and chargeback. Fully offline, zero required
dependencies.

```python
from showback import BudgetExceeded, BudgetGuard, SpendTracker

tracker = SpendTracker()  # local, append-only cost ledger
guard = BudgetGuard(0.50, tracker)  # hard $0.50 cap

try:
    with guard:  # pre-flight: raises immediately if already over budget
        for step in plan:
            guard.check()  # mid-run: raises BudgetExceeded the moment the cap is hit
            tracker.record_llm("gpt-4o-mini", in_tok, out_tok, agent="researcher", task_id="t-42")
            tracker.record_tool("web_search", 0.0005, agent="researcher", task_id="t-42")
except BudgetExceeded as e:
    print(f"aborted: {e}")
```

## Why this exists

Cost dashboards answer "what did we spend?" after the money is gone. Agent runs are
different from single API calls: a loop that was supposed to cost cents can spiral into
dollars while nobody is watching. What was missing is **enforcement inside the loop** —
a check between steps that actually halts the run, plus cost attribution per agent,
task, team, or customer so the budget means something. That's this library.

## Features

- **Pricing table** — per-model USD-per-1M-token prices, editable via JSON/YAML file or
  Python API. Sensible built-in defaults; works fully offline.
- **SpendTracker** — thread-safe, append-only ledger of LLM calls (model, input/output
  tokens, free-form tags) with costs computed from the pricing table.
- **BudgetGuard** — context manager *and* decorator. Pre-flight check on entry, plus a
  `check()` method to call between agent steps that raises `BudgetExceeded` mid-run.
  Hard caps, not soft alerts. Guards can be scoped with `match_tags` so several tasks
  share one ledger while each gets its own budget.
- **Attribution & export** — per-tag spend summaries for showback/chargeback, ledger
  export to JSON and CSV.
- **Tool-call attribution** — record tool/MCP-style calls with explicit costs against the
  same budget and the same tags.
- **LiteLLM integration** (optional extra) — drop-in `success_callback` that records
  every call and enforces the guard.

## Install

```bash
pip install showback            # core: zero required dependencies
pip install "showback[litellm]" # LiteLLM callback integration
pip install "showback[yaml]"    # YAML pricing files (JSON works out of the box)
```

## Architecture

```
                    +----------------+
                    |  PricingTable  |  USD per 1M tokens, editable (JSON/YAML/API)
                    +-------+--------+
                            | cost(model, in_tok, out_tok)
                            v
                    +----------------+       +------------------+
  record_llm() ---> |  SpendTracker  | <---> |   BudgetGuard    |
  record_tool() --> |  (ledger)      |       | pre-flight +     |
                    +-------+--------+       | check() -> raise |
                            |                +------------------+
                            v
                 summarize_by_tag / export_json / export_csv
                            |
                    integrations/litellm.py (optional)
```

`SpendTracker` owns the facts (what was spent, tagged how). `BudgetGuard` owns the
policy (how much may be spent, over which slice of the ledger). Attribution and export
turn the ledger into reports. Integrations feed real calls into the tracker.

## Pricing

```python
from showback import PricingTable

table = PricingTable.from_json_file("pricing.json")  # or .from_yaml_file(...)
table.set_price("my-fine-tune", input_per_1m=3.0, output_per_1m=12.0)
table.set_price("*", 1.0, 4.0)  # wildcard default for unlisted models
table.to_json_file("pricing.json")  # version-control it with your app
```

The built-in defaults are starting points — providers change pricing, so pin your own
table in production. Unknown models raise `UnknownModelError` unless a `"*"` wildcard
is registered; `openai/gpt-4o`-style prefixed names resolve to their base model.

## LiteLLM

```python
import litellm
from showback import BudgetGuard, SpendTracker
from showback.integrations.litellm import SpendGuardCallback

tracker = SpendTracker()
guard = BudgetGuard(2.00, tracker, match_tags={"agent": "researcher"})
litellm.success_callback = [SpendGuardCallback(tracker, guard, tags={"agent": "researcher"})]
```

Every completed call is recorded with its real token usage; `guard.check()` runs after
each one, so a multi-step loop aborts as soon as the cap is hit.

## Paper-backed features

Three techniques from the academic literature, implemented as small offline modules:

### CascadeRouter — FrugalGPT cascades

*FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving
Performance* (Chen, Zaharia, Zou; arXiv:2305.05176, TMLR 2024): order models
cheap → expensive, score each answer's reliability, and escalate only on low scores —
reported up to 98% cost reduction at GPT-4 parity.

```python
from showback import BudgetGuard, CascadeRouter, SpendTracker, nonempty_scorer

tracker = SpendTracker()
guard = BudgetGuard(1.00, tracker)
router = CascadeRouter(
    [("gpt-4o-mini", nonempty_scorer, 0.7), ("gpt-4o", nonempty_scorer, 0.0)], tracker, guard
)
result = router.route("Summarize this report.", call_llm)  # cheap first; escalates on low score
```

`call_llm(model, prompt)` returns `(answer, in_tokens, out_tokens)`; every tier tried
is recorded in the ledger, escalation aborts via `BudgetExceeded` instead of
overspending, and the last tier is the fallback. Plug a real reliability classifier
(e.g. a fine-tuned DistilBERT, as in the paper) as the scorer for production use.

### LoopDetector — MAST failure monitors

*Why Do Multi-Agent LLM Systems Fail?* (MAST) (Cemri et al.; arXiv:2503.13657,
NeurIPS 2025) found step repetition (~15.7%)
and failure-to-terminate (~12.4%) dominate agent failures; the implementable primitive
is a repetition detector over the action stream (see also Dubey, *Real-Time
Detection and Repair of LLM Agent Failures*, arXiv:2608.02464).

```python
from showback import BudgetGuard, LoopDetected, LoopDetector, SpendTracker, loop_to_budget_exceeded

tracker = SpendTracker()
guard, detector = (
    BudgetGuard(1.00, tracker),
    LoopDetector(),
)  # raises LoopDetected on repeats/cycles
try:
    for tool, sig in agent_steps():
        detector.observe(tool, sig)
except LoopDetected as e:
    raise loop_to_budget_exceeded(e, guard) from e  # abort via the budget path
```

Deterministic and offline — no model calls. ML-based telemetry (e.g. CUSUM drift
detection) is roadmap, not v1.

### Budget injection + per-tool caps — BATS

*BATS: Budget-Aware Tool-Use Enables Effective Agent Scaling* (Liu et al.;
arXiv:2511.17006): give the agent a continuous signal of remaining budget, enforce
per-tool call caps, and keep the vocabulary straight — BUDGET is the hard upper
limit, COST is realized spend.

```python
tracker = SpendTracker()
tracker.set_tool_cap("web_search", 0.01)  # per-call cap b_i
guard = BudgetGuard(0.50, tracker)
prompt = task + "\n" + guard.budget_prompt_snippet()  # inject remaining budget each step
```

`record_tool` now refuses any single call over its cap (`BudgetExceeded`, nothing
recorded), and `guard.spent_ratio` reports the fraction of budget consumed.

## Honest positioning

| Tool | What it does | What it doesn't do |
|---|---|---|
| **LiteLLM** | Gateway-level budget caps per key/team, blocks requests at the proxy | Nothing inside your agent loop; no per-task attribution |
| **Langfuse / LangSmith** | Excellent per-trace cost logging and dashboards | Logging only — no hard caps, no mid-run abort |
| **Portkey / Helicone-style gateways** | Spend visibility, some with request blocking | Proxy-bound; your in-process agent code is on its own |
| **showback** | In-process hard caps: pre-flight + mid-run abort, per-tag attribution | Not a dashboard, not a proxy — pairs with the tools above |

Use them together: a gateway cap as the outer fence, this library as the inner guard
that stops *this specific run* before it burns *this specific task's* budget.

## Example

`examples/agent_loop.py` simulates a 6-step agent with a $0.004 budget. It aborts after
3 steps, then prints the ledger, per-step showback, and exports JSON + CSV:

```bash
python examples/agent_loop.py
```

## Roadmap

- [ ] Async support (`asyncio`-safe tracker, async guard checks)
- [ ] Native middleware for the OpenAI and Anthropic SDKs (like the LiteLLM callback)
- [ ] Streaming token accounting (record partial usage as streams progress)
- [ ] Budget webhooks / callbacks (notify PagerDuty/Slack *and* abort)
- [ ] SQLite ledger backend for long-running services
- [ ] Prometheus exporter for the ledger
- [ ] `spend-guard` CLI: `spend-guard report ledger.json --by team`

## Contributing

See [CONTRIBUTING.md](https://github.com/Naeemeh146/showback/blob/main/CONTRIBUTING.md). Bug reports and PRs welcome — especially
pricing corrections and new integrations.

## License

MIT — see [LICENSE](https://github.com/Naeemeh146/showback/blob/main/LICENSE). Copyright 2026 Naeemeh Yadollahpour.
