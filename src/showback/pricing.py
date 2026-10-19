"""Per-model token pricing and cost calculation. Fully offline.

Prices are USD per 1M tokens. The built-in table is a set of editable
defaults so the library works out of the box — verify them against your
provider's current pricing and override via JSON/YAML file or the Python API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class UnknownModelError(KeyError):
    """Raised when a model has no registered price and no default price is set."""


# Editable defaults (USD per 1M tokens). Intended as starting points, not as
# a price feed — providers change pricing, so pin your own table in production.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "grok-4": {"input": 3.00, "output": 15.00},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "llama-3.3-70b": {"input": 0.35, "output": 0.40},
}

_JSON_TEMPLATE = """{
  "gpt-4o-mini": {"input": 0.15, "output": 0.60},
  "my-fine-tune": {"input": 3.00, "output": 12.00},
  "*": {"input": 1.00, "output": 4.00}
}
"""


class PricingTable:
    """Mutable table of per-model USD-per-1M-token prices with cost math."""

    def __init__(self, prices: dict[str, dict[str, float]] | None = None) -> None:
        self._prices: dict[str, dict[str, float]] = {}
        for model, rates in (prices if prices is not None else DEFAULT_PRICES).items():
            self.set_price(model, rates["input"], rates["output"])

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float]]) -> PricingTable:
        """Build a table from a plain mapping. The ``"*"`` key acts as a wildcard default."""
        table = cls({})
        for model, rates in data.items():
            table.set_price(model, rates["input"], rates["output"])
        return table

    @classmethod
    def from_json_file(cls, path: str | Path) -> PricingTable:
        """Load a table from a JSON file shaped like :data:`_JSON_TEMPLATE`."""
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> PricingTable:
        """Load a pricing table from a YAML file.

        Needs the ``yaml`` extra: ``pip install showback[yaml]``.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML pricing files need PyYAML: pip install showback[yaml]") from exc
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @staticmethod
    def json_template() -> str:
        """Return an example JSON pricing file you can copy and edit."""
        return _JSON_TEMPLATE

    def set_price(self, model: str, input_per_1m: float, output_per_1m: float) -> None:
        """Register or override a model's prices (USD per 1M tokens)."""
        if input_per_1m < 0 or output_per_1m < 0:
            raise ValueError("prices must be non-negative")
        self._prices[model.strip().lower()] = {
            "input": float(input_per_1m),
            "output": float(output_per_1m),
        }

    def get_price(self, model: str) -> dict[str, float]:
        """Return ``{"input": ..., "output": ...}`` for a model.

        Resolution order: exact name (case-insensitive), name with any
        ``provider/`` prefix stripped (e.g. ``openai/gpt-4o``), then the
        ``"*"`` wildcard entry. Raises :class:`UnknownModelError` otherwise.
        """
        key = model.strip().lower()
        if key in self._prices:
            return dict(self._prices[key])
        if "/" in key:
            short = key.split("/", 1)[1]
            if short in self._prices:
                return dict(self._prices[short])
        if "*" in self._prices:
            return dict(self._prices["*"])
        raise UnknownModelError(
            f"No price registered for model {model!r}. "
            "Call set_price() for it, or register a '*' wildcard default."
        )

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Cost in USD for a call, rounded to 6 decimals."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        rates = self.get_price(model)
        total = (input_tokens / 1_000_000) * rates["input"]
        total += (output_tokens / 1_000_000) * rates["output"]
        return round(total, 6)

    def models(self) -> list[str]:
        """Sorted list of registered model names (``"*"`` included when set)."""
        return sorted(self._prices)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {model: dict(rates) for model, rates in self._prices.items()}

    def to_json_file(self, path: str | Path) -> Path:
        """Persist the table so pricing is version-controlled with your app."""
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target
