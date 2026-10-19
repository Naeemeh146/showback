import json

import pytest

from showback import DEFAULT_PRICES, PricingTable, UnknownModelError


def test_default_table_cost_math():
    table = PricingTable()
    # gpt-4o-mini defaults: $0.15 / $0.60 per 1M
    assert table.cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert table.cost("gpt-4o-mini", 0, 0) == 0.0


def test_case_insensitive_and_provider_prefix():
    table = PricingTable()
    assert table.cost("GPT-4O-MINI", 1_000_000, 0) == pytest.approx(0.15)
    assert table.cost("openai/gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)


def test_unknown_model_raises():
    table = PricingTable()
    with pytest.raises(UnknownModelError):
        table.cost("not-a-real-model-xyz", 100, 100)


def test_wildcard_default_price():
    table = PricingTable.from_dict({"*": {"input": 1.0, "output": 2.0}})
    assert table.cost("anything-at-all", 1_000_000, 1_000_000) == pytest.approx(3.0)


def test_set_price_overrides():
    table = PricingTable()
    table.set_price("gpt-4o-mini", 0.30, 1.20)
    assert table.cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.30)


def test_negative_values_rejected():
    table = PricingTable()
    with pytest.raises(ValueError):
        table.set_price("x", -1.0, 1.0)
    with pytest.raises(ValueError):
        table.cost("gpt-4o-mini", -5, 10)


def test_json_round_trip(tmp_path):
    table = PricingTable()
    path = table.to_json_file(tmp_path / "prices.json")
    loaded = PricingTable.from_json_file(path)
    assert loaded.cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert loaded.models() == table.models()


def test_json_template_is_valid():
    data = json.loads(PricingTable.json_template())
    table = PricingTable.from_dict(data)
    assert table.cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert table.cost("brand-new-model", 1_000_000, 0) == pytest.approx(1.0)  # wildcard


def test_yaml_round_trip(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "prices.yaml"
    path.write_text("gpt-4o-mini:\n  input: 0.15\n  output: 0.60\n", encoding="utf-8")
    table = PricingTable.from_yaml_file(path)
    assert table.cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert yaml  # silence linters about the unused import


def test_default_prices_exported():
    assert DEFAULT_PRICES["gpt-4o-mini"] == {"input": 0.15, "output": 0.60}
