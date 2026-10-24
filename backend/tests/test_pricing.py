import dataclasses
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, StageConfig, parse_prices, settings
from app.ingest.loader import SourceDoc
from app.llm.client import PRICING, LLMClient, UsageLog, price_table, unpriced_models
from app.llm.generate import generate_cards
from app.llm.schemas import Card, CardSet
from app.main import create_app
from app.pipeline.worker import CostCeilingExceeded, check_cost_ceiling
from tests.conftest import FakeSDK

OPUS = "claude-opus-5-5"
UNPRICED = "claude-mystery-9"


def usage(inp=1_000_000, out=100_000):
    return SimpleNamespace(
        input_tokens=inp, output_tokens=out, cache_read_input_tokens=0, cache_creation_input_tokens=0,
        server_tool_use=SimpleNamespace(web_search_requests=0),
    )


# --- pricing -------------------------------------------------------------------------------


def test_a_known_model_is_costed_from_the_price_table():
    log = UsageLog()
    log.add("generate", OPUS, usage())  # 1M in at $4 + 0.1M out at $20
    assert log.total_cost == pytest.approx(4.0 + 2.0)
    assert log.unpriced() == []


def test_an_unknown_model_has_no_cost_and_makes_the_whole_total_unknown():
    log = UsageLog()
    log.add("understand", OPUS, usage())
    log.add("generate", UNPRICED, usage())
    assert [r.cost is None for r in log.rows] == [False, True]
    assert log.total_cost is None  # never a partial sum that looks complete
    assert log.unpriced() == [UNPRICED]


def test_extra_prices_make_a_model_priced_and_override_the_built_in_table():
    cfg = dataclasses.replace(settings, extra_prices={UNPRICED: (1.0, 5.0), OPUS: (8.0, 40.0)})
    table = price_table(cfg)
    assert table[UNPRICED] == (1.0, 5.0) and table[OPUS] == (8.0, 40.0)
    assert PRICING[OPUS] == (4.0, 20.0)  # the built-in table itself is untouched

    client = LLMClient(FakeSDK(), cfg)
    client.usage.add("generate", UNPRICED, usage())
    assert client.usage.total_cost == pytest.approx(1.0 + 0.5)


def test_unpriced_models_lists_only_configured_stage_models_without_a_price():
    assert unpriced_models(settings) == []
    cfg = dataclasses.replace(settings, generate=StageConfig(UNPRICED, "medium", 1000), verify=StageConfig("another-one", "low", 1000))
    assert unpriced_models(cfg) == ["another-one", UNPRICED]
    assert unpriced_models(dataclasses.replace(cfg, extra_prices={UNPRICED: (1, 2), "another-one": (1, 2)})) == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", {}),
        ("  ", {}),
        ('{"m": [3, 15]}', {"m": (3.0, 15.0)}),
        ('{"a": [1.5, 7], "b": [0, 0]}', {"a": (1.5, 7.0), "b": (0.0, 0.0)}),
    ],
)
def test_parse_prices_accepts_the_documented_shape(raw, expected):
    assert parse_prices(raw) == expected


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"m": 5}', '{"m": [1]}', '{"m": ["x", "y"]}', '{"m": [-1, 2]}', '{"m": null}'])
def test_parse_prices_rejects_anything_else_with_a_helpful_message(raw):
    with pytest.raises(ValueError, match="SNAPNOTES_EXTRA_PRICES"):
        parse_prices(raw)


def test_prices_can_be_supplied_through_the_environment(monkeypatch):
    monkeypatch.setenv("SNAPNOTES_EXTRA_PRICES", '{"claude-future-1": [2, 9]}')
    assert Settings().extra_prices == {"claude-future-1": (2.0, 9.0)}
    monkeypatch.setenv("SNAPNOTES_EXTRA_PRICES", "oops")
    with pytest.raises(ValueError, match="SNAPNOTES_EXTRA_PRICES"):
        Settings()


# --- the spend ceiling fails closed -------------------------------------------------------------


def test_the_ceiling_allows_spend_up_to_the_limit_and_stops_beyond_it():
    log = UsageLog()
    log.add("understand", OPUS, usage(500_000, 0))  # $2.00
    check_cost_ceiling(0.0, log, ceiling=2.0)  # exactly at the limit is fine
    check_cost_ceiling(0.0, log, ceiling=5.0)
    with pytest.raises(CostCeilingExceeded, match=r"\$1\.99 cost limit"):
        check_cost_ceiling(0.0, log, ceiling=1.99)


def test_the_ceiling_counts_what_earlier_attempts_already_cost():
    log = UsageLog()
    log.add("generate", OPUS, usage(250_000, 0))  # $1.00 this run
    check_cost_ceiling(0.5, log, ceiling=2.0)
    with pytest.raises(CostCeilingExceeded):
        check_cost_ceiling(1.5, log, ceiling=2.0)  # a retry cannot dodge the limit


def test_the_ceiling_stops_a_job_that_used_a_model_with_no_price():
    log = UsageLog()
    log.add("generate", UNPRICED, usage())
    with pytest.raises(CostCeilingExceeded, match=f"no price is known for {UNPRICED}"):
        check_cost_ceiling(0.0, log, ceiling=1_000_000.0)  # even a huge limit must not wave it through


def test_an_empty_log_costs_nothing_so_the_first_stage_always_starts():
    check_cost_ceiling(0.0, UsageLog(), ceiling=0.01)


# --- startup check -------------------------------------------------------------------------------


def test_the_server_refuses_to_start_when_a_stage_model_has_no_price(tmp_path):
    cfg = Settings(data_dir=tmp_path, api_token="t" * 32, generate=StageConfig(UNPRICED, "medium", 1000))
    with pytest.raises(RuntimeError, match=UNPRICED) as err:
        with TestClient(create_app(cfg)):
            pass
    assert "SNAPNOTES_EXTRA_PRICES" in str(err.value)  # and it says how to fix it


def test_the_server_starts_once_the_price_is_supplied(tmp_path):
    cfg = Settings(data_dir=tmp_path, api_token="t" * 32, generate=StageConfig(UNPRICED, "medium", 1000), extra_prices={UNPRICED: (3.0, 15.0)})
    with TestClient(create_app(cfg)) as client:
        assert client.get("/health").status_code == 200


# --- every stage really uses the settings it is priced with ----------------------------------------


def test_stages_use_the_clients_own_settings_not_a_global_copy(tmp_path):
    cfg = dataclasses.replace(settings, generate=StageConfig("claude-sonnet-5-5", "low", 1234), max_cards=2)
    card = Card(type="basic", front="Q", back="A", origin="notes", source_ref="p.1", source_snippet="s", tags=[], difficulty="easy")
    sdk = FakeSDK(parse_results={CardSet: CardSet(cards=[card] * 5)})

    cards = generate_cards(LLMClient(sdk, cfg), SourceDoc(name="n", blocks=[{"type": "text", "text": "notes"}]), [])

    _, _, kwargs = sdk.calls[0]
    assert (kwargs["model"], kwargs["max_tokens"], kwargs["output_config"]) == ("claude-sonnet-5-5", 1234, {"effort": "low"})
    assert len(cards) == 2  # max_cards also comes from the client's settings
