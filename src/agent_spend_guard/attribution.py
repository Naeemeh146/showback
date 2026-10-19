"""Attribution (showback/chargeback) and ledger export. Fully offline."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from agent_spend_guard.tracker import LedgerEntry, SpendTracker

EntriesLike = Iterable[LedgerEntry] | SpendTracker


def _as_entries(source: EntriesLike) -> tuple[LedgerEntry, ...]:
    if isinstance(source, SpendTracker):
        return source.entries
    return tuple(source)


def summarize_by_tag(source: EntriesLike, tag: str) -> dict[str, float]:
    """Total USD per value of ``tag`` — the showback/chargeback primitive.

    Example: ``summarize_by_tag(tracker, "team")`` ->
    ``{"support": 1.25, "research": 4.10}``. Entries missing the tag are
    grouped under ``"(untagged)"``.
    """
    totals: dict[str, float] = {}
    for entry in _as_entries(source):
        key = entry.tags.get(tag, "(untagged)")
        totals[key] = round(totals.get(key, 0.0) + entry.cost_usd, 6)
    return dict(sorted(totals.items()))


def export_json(source: EntriesLike, path: str | Path) -> Path:
    """Write the ledger to a JSON file. Returns the path written."""
    target = Path(path)
    payload = [entry.to_dict() for entry in _as_entries(source)]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


_CSV_COLUMNS = [
    "id",
    "timestamp",
    "kind",
    "model",
    "name",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "tags",
    "note",
]


def export_csv(source: EntriesLike, path: str | Path) -> Path:
    """Write the ledger to a CSV file (tags serialized as compact JSON)."""
    target = Path(path)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for entry in _as_entries(source):
            row = entry.to_dict()
            row["tags"] = json.dumps(row["tags"], separators=(",", ":"))
            writer.writerow({column: row[column] for column in _CSV_COLUMNS})
    return target
