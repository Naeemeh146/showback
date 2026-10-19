import csv
import json

import pytest

from agent_spend_guard import (
    SpendTracker,
    export_csv,
    export_json,
    summarize_by_tag,
)


def _sample_tracker() -> SpendTracker:
    tracker = SpendTracker()
    tracker.record_llm("gpt-4o-mini", 1_000_000, 0, team="support")  # 0.15
    tracker.record_llm("gpt-4o-mini", 1_000_000, 0, team="research")  # 0.15
    tracker.record_llm("gpt-4o-mini", 1_000_000, 0, team="support")  # 0.15
    tracker.record_tool("web_search", 0.05)  # untagged
    return tracker


def test_summarize_by_tag():
    result = summarize_by_tag(_sample_tracker(), "team")
    assert result == {"(untagged)": 0.05, "research": 0.15, "support": 0.30}


def test_summarize_accepts_raw_entries():
    tracker = _sample_tracker()
    assert summarize_by_tag(tracker.entries, "team")["support"] == 0.30


def test_export_json_round_trip(tmp_path):
    tracker = _sample_tracker()
    path = export_json(tracker, tmp_path / "ledger.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == 4
    assert sum(item["cost_usd"] for item in payload) == pytest.approx(0.50)
    first = payload[0]
    assert first["kind"] == "llm"
    assert first["tags"] == {"team": "support"}


def test_export_csv_round_trip(tmp_path):
    tracker = _sample_tracker()
    path = export_csv(tracker, tmp_path / "ledger.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["kind"] == "llm"
    assert rows[0]["model"] == "gpt-4o-mini"
    assert json.loads(rows[3]["tags"]) == {}
    total = sum(float(row["cost_usd"]) for row in rows)
    assert total == pytest.approx(0.50)
