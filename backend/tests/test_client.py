import dataclasses
import json
from types import SimpleNamespace

import pytest

from app.core.config import StageConfig, settings
from app.llm.client import FALLBACK_BETA, LLMClient, RefusalError, TruncatedError
from app.llm.schemas import CardSet
from tests.conftest import FakeSDK, citation, response, search_result_block, text_block, usage

CONTENT = [{"type": "text", "text": "x"}]
OPUS = StageConfig("claude-opus-5-5", "medium", 1000)
HAIKU = StageConfig("claude-haiku-5-5", "low", 1000)


def parse_call(stage, cfg=settings):
    sdk = FakeSDK(parse_results={CardSet: CardSet(cards=[])})
    LLMClient(sdk, cfg).parse("t", stage, content=CONTENT, output_model=CardSet)
    return sdk.calls[0]


def test_default_route_is_beta_with_server_side_fallbacks_and_effort():
    tag, kind, kw = parse_call(OPUS)
    assert (tag, kind) == ("beta", "parse")
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert kw["output_config"] == {"effort": "medium"}
    assert "temperature" not in kw and "thinking" not in kw  # sampling params 400; thinking is adaptive by default


def test_haiku_never_gets_fallbacks():
    tag, _, kw = parse_call(HAIKU)
    assert tag == "plain" and "fallbacks" not in kw


def test_fallbacks_can_be_disabled():
    tag, _, kw = parse_call(OPUS, dataclasses.replace(settings, use_fallbacks=False))
    assert tag == "plain" and "betas" not in kw


def test_refusal_and_truncation_raise():
    sdk = FakeSDK(parse_results={CardSet: CardSet(cards=[])})
    sdk.parse_stop[CardSet] = "refusal"
    with pytest.raises(RefusalError):
        LLMClient(sdk).parse("t", OPUS, content=CONTENT, output_model=CardSet)
    sdk.parse_stop[CardSet] = "max_tokens"
    with pytest.raises(TruncatedError):
        LLMClient(sdk).parse("t", OPUS, content=CONTENT, output_model=CardSet)


def test_research_collects_citations_and_continues_on_pause_turn():
    first = response(
        [search_result_block(), text_block("- Mitochondria make ATP.", [citation("https://a.org/x", "A", "ATP is made in mitochondria")])],
        stop_reason="pause_turn",
    )
    second = response(
        [
            text_block("- Mitochondria make ATP.", [citation("https://a.org/x", "A", "ATP is made in mitochondria")]),  # repeat
            text_block("- DNA is double stranded.", [citation("https://b.org/y", "B", "double helix")]),
        ]
    )
    sdk = FakeSDK(create_responses=[first, second])
    result = LLMClient(sdk).research("enrich", OPUS, content=CONTENT)

    assert [e.url for e in result.evidence] == ["https://a.org/x", "https://b.org/y"]  # duplicate citation dropped
    assert result.searches_used == 1
    assert len(sdk.calls) == 2
    # the second request carries the partial assistant turn back
    assert sdk.calls[1][2]["messages"][-1]["role"] == "assistant"
    tool = sdk.calls[0][2]["tools"][0]
    assert tool["type"] == "web_search_20260209" and tool["max_uses"] == settings.web_search_max_uses


def test_research_with_no_citations_returns_no_evidence():
    sdk = FakeSDK(create_responses=[response([text_block("Nothing found.")])])
    result = LLMClient(sdk).research("enrich", OPUS, content=CONTENT)
    assert result.evidence == []


def test_searching_but_collecting_no_evidence_is_reported_with_what_came_back(caplog):
    """Seen on the first real run: 5 searches, a long answer, and no evidence. The log must say what the API returned."""
    sdk = FakeSDK(create_responses=[response([search_result_block(), text_block("Ribosomes contain rRNA.")], usage=usage(searches=5))])
    with caplog.at_level("WARNING", logger="app.llm.client"):
        result = LLMClient(sdk).research("enrich", OPUS, content=CONTENT)
    assert result.evidence == []
    message = caplog.records[-1].getMessage()
    assert "no web evidence was collected" in message
    assert "5 search(es) reported by the API" in message and "1 result block(s) recognised" in message
    assert "'web_search_tool_result': 1" in message and "'text': 1" in message
    assert "Citation types: none" in message and "SNAPNOTES_DEBUG_DUMP" in message


def test_citations_of_an_unexpected_type_are_named_in_that_report(caplog):
    odd = SimpleNamespace(type="char_location", url="https://a.org", title="A", cited_text="x")
    sdk = FakeSDK(create_responses=[response([text_block("Claim.", [odd])])])
    with caplog.at_level("WARNING", logger="app.llm.client"):
        assert LLMClient(sdk).research("enrich", OPUS, content=CONTENT).evidence == []
    assert "'char_location': 1" in caplog.records[-1].getMessage()


def test_no_warning_when_evidence_was_found(caplog):
    sdk = FakeSDK(create_responses=[response([text_block("Fact.", [citation("https://a.org", "A", "x")])])])
    with caplog.at_level("WARNING", logger="app.llm.client"):
        assert len(LLMClient(sdk).research("enrich", OPUS, content=CONTENT).evidence) == 1
    assert not caplog.records


def test_raw_responses_are_saved_only_when_a_folder_is_configured(tmp_path):
    def run(cfg):
        sdk = FakeSDK(create_responses=[response([search_result_block(), text_block("Fact.", [citation("https://a.org", "A", "x")])])])
        LLMClient(sdk, cfg).research("enrich", OPUS, content=CONTENT)

    run(settings)
    assert not list(tmp_path.iterdir())

    run(dataclasses.replace(settings, debug_dump_dir=tmp_path / "dump"))
    [saved] = list((tmp_path / "dump").iterdir())
    assert "research-enrich" in saved.name
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert [b["type"] for b in data["content"]] == ["web_search_tool_result", "text"]
    assert data["content"][1]["citations"][0]["url"] == "https://a.org"


def test_a_failing_dump_never_breaks_the_research(tmp_path, caplog):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a folder")
    sdk = FakeSDK(create_responses=[response([text_block("Fact.", [citation("https://a.org", "A", "x")])])])
    with caplog.at_level("ERROR", logger="app.llm.client"):
        result = LLMClient(sdk, dataclasses.replace(settings, debug_dump_dir=blocker / "dump")).research("enrich", OPUS, content=CONTENT)
    assert len(result.evidence) == 1
    assert any("could not write the debug dump" in r.getMessage() for r in caplog.records)


def test_the_dump_folder_comes_from_the_environment(monkeypatch, tmp_path):
    from app.core.config import Settings

    monkeypatch.delenv("SNAPNOTES_DEBUG_DUMP", raising=False)
    assert Settings().debug_dump_dir is None
    monkeypatch.setenv("SNAPNOTES_DEBUG_DUMP", f"  {tmp_path}  ")
    assert Settings().debug_dump_dir == tmp_path


def test_domain_filters_are_mutually_exclusive():
    cfg = dataclasses.replace(settings, web_allowed_domains=("wikipedia.org",), web_blocked_domains=("spam.com",))
    tool = LLMClient(FakeSDK(), cfg)._search_tool()
    assert tool["allowed_domains"] == ["wikipedia.org"] and "blocked_domains" not in tool


def test_usage_cost_uses_cache_multipliers():
    sdk = FakeSDK(parse_results={CardSet: CardSet(cards=[])})
    client = LLMClient(sdk)
    client.parse("t", OPUS, content=CONTENT, output_model=CardSet)
    row = client.usage.rows[0]
    assert row.cost == pytest.approx((100 * 4.0 + 50 * 20.0) / 1_000_000)
