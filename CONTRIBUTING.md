# Contributing to agent-spend-guard

Thanks for helping build the enforcement layer agent spend is missing. Bug reports,
pricing corrections, and new integrations are all welcome.

## Setup

```bash
git clone https://github.com/Naeemeh146/agent-spend-guard.git
cd agent-spend-guard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks (all must pass)

```bash
pytest            # full test suite — must be green
ruff check .      # lint — must be clean
ruff format .     # formatting — run before committing
python examples/agent_loop.py   # the example must abort mid-run and print its ledger
```

## Ground rules

- **Core stays dependency-free.** The `agent_spend_guard` package itself must never
  gain a required runtime dependency. Integrations live under
  `src/agent_spend_guard/integrations/` and are optional extras; importing the top-level
  package must never import them.
- **Fully offline.** No network calls in the library or the test suite. Tests that need
  a third-party package must fake it (see `tests/test_integrations.py`).
- **No secrets, ever.** Never commit tokens, API keys, credentials, or `.env` files.
  Grep your diff before pushing.
- **Hard caps, not alerts.** New features should enforce; reporting alone belongs in
  dashboards, not here.
- Keep public API changes backward-compatible within a minor version; document
  breaking changes clearly in the PR.

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Add tests for any behavior change; update the README if the public API changes.
3. Run the checks above.
4. Open the PR with a short description of the problem, the fix, and how you tested it.
