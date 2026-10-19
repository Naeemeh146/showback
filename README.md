# agent-spend-guard

[![PyPI version](https://img.shields.io/pypi/v/agent-spend-guard)](https://pypi.org/project/agent-spend-guard/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-spend-guard)](https://pypi.org/project/agent-spend-guard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Naeemeh146/agent-spend-guard/blob/main/LICENSE)

Hard budget enforcement for AI agent spend. Every observability tool logs what your
agents *already spent* — `agent-spend-guard` is the layer that **stops the run** when a
budget is breached: refuse to start an over-budget run (pre-flight), and abort it
mid-flight between steps. Fully offline, zero required dependencies.

```python
from agent_spend_guard import BudgetExceeded, BudgetGuard, SpendTracker

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
pip install agent-spend-guard            # core: zero required dependencies
pip install "agent-spend-guard[litellm]" # LiteLLM callback integration
pip install "agent-spend-guard[yaml]"    # YAML pricing files (JSON works out of the box)
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
from agent_spend_guard import PricingTable

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
from agent_spend_guard import BudgetGuard, SpendTracker
from agent_spend_guard.integrations.litellm import SpendGuardCallback

tracker = SpendTracker()
guard = BudgetGuard(2.00, tracker, match_tags={"agent": "researcher"})
litellm.success_callback = [SpendGuardCallback(tracker, guard, tags={"agent": "researcher"})]
```

Every completed call is recorded with its real token usage; `guard.check()` runs after
each one, so a multi-step loop aborts as soon as the cap is hit.

## Honest positioning

| Tool | What it does | What it doesn't do |
|---|---|---|
| **LiteLLM** | Gateway-level budget caps per key/team, blocks requests at the proxy | Nothing inside your agent loop; no per-task attribution |
| **Langfuse / LangSmith** | Excellent per-trace cost logging and dashboards | Logging only — no hard caps, no mid-run abort |
| **Portkey / Helicone-style gateways** | Spend visibility, some with request blocking | Proxy-bound; your in-process agent code is on its own |
| **agent-spend-guard** | In-process hard caps: pre-flight + mid-run abort, per-tag attribution | Not a dashboard, not a proxy — pairs with the tools above |

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

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and PRs welcome — especially
pricing corrections and new integrations.

## License

MIT — see [LICENSE](LICENSE). Copyright 2026 Naeemeh Yadollahpour.
