"""The example is a runnable spec: it must abort mid-run and print its ledger."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_agent_loop_aborts_mid_run():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "agent_loop.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "ABORTED mid-run after 3/6 steps" in out
    assert "Budget exceeded" in out
    assert "--- ledger ---" in out
    assert "TOTAL:" in out
    assert "ledger.json" in out and "ledger.csv" in out
