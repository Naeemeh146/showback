"""SpendTracker: the local cost ledger. Thread-safe, fully offline."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_spend_guard.pricing import PricingTable


@dataclass(frozen=True)
class LedgerEntry:
    """One recorded spend event: an LLM call or a tool call."""

    id: str
    timestamp: str  # UTC ISO-8601
    kind: str  # "llm" or "tool"
    model: str | None  # model name for LLM calls
    name: str | None  # tool name for tool calls
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tags: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "model": self.model,
            "name": self.name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "tags": dict(self.tags),
            "note": self.note,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_tags(tags: dict) -> dict[str, str]:
    return {str(key): str(value) for key, value in tags.items()}


class SpendTracker:
    """Append-only ledger of LLM and tool-call spend.

    All recording methods are thread-safe, so one tracker can be shared by
    concurrent agent workers.
    """

    def __init__(self, pricing: PricingTable | None = None) -> None:
        self.pricing = pricing if pricing is not None else PricingTable()
        self._entries: list[LedgerEntry] = []
        self._lock = threading.Lock()

    def _append(self, entry: LedgerEntry) -> LedgerEntry:
        with self._lock:
            self._entries.append(entry)
        return entry

    def record_llm(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        note: str = "",
        **tags: object,
    ) -> LedgerEntry:
        """Record one LLM call. Cost is computed from the pricing table.

        Tags are free-form strings, e.g. ``agent="researcher"``,
        ``task_id="t-42"``, ``team="support"``, ``customer="acme"``.
        """
        cost = self.pricing.cost(model, int(input_tokens), int(output_tokens))
        return self._append(
            LedgerEntry(
                id=uuid.uuid4().hex,
                timestamp=_now_iso(),
                kind="llm",
                model=model,
                name=None,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cost_usd=cost,
                tags=_coerce_tags(tags),
                note=note,
            )
        )

    def record_tool(
        self,
        name: str,
        cost_usd: float,
        *,
        note: str = "",
        **tags: object,
    ) -> LedgerEntry:
        """Record one tool/MCP-style call against the same ledger.

        Tool calls rarely have a standard token price, so the caller supplies
        the cost directly (e.g. a per-call fee, or an estimated token cost).
        """
        if cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")
        return self._append(
            LedgerEntry(
                id=uuid.uuid4().hex,
                timestamp=_now_iso(),
                kind="tool",
                model=None,
                name=name,
                input_tokens=0,
                output_tokens=0,
                cost_usd=round(float(cost_usd), 6),
                tags=_coerce_tags(tags),
                note=note,
            )
        )

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Immutable snapshot of the ledger."""
        with self._lock:
            return tuple(self._entries)

    def total_cost(self) -> float:
        """Total USD across the whole ledger."""
        return round(sum(e.cost_usd for e in self.entries), 6)

    def cost_for_tags(self, match: dict[str, str]) -> float:
        """Total USD for entries whose tags contain all of ``match``.

        This is what lets one shared tracker serve per-agent or per-task
        budgets: ``tracker.cost_for_tags({"task_id": "t-42"})``.
        """
        wanted = {str(k): str(v) for k, v in match.items()}
        total = sum(
            e.cost_usd for e in self.entries if all(e.tags.get(k) == v for k, v in wanted.items())
        )
        return round(total, 6)

    def clear(self) -> None:
        """Empty the ledger (useful in tests)."""
        with self._lock:
            self._entries.clear()
