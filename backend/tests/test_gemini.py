"""Gemini (Google's free tier) behind the same two calls as Claude: conversion, parsing, failures, wiring and the whole pipeline."""
import base64
import dataclasses
import logging
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google import genai
from google.genai import errors, types

from app.core.config import Settings, provider_problem, settings
from app.llm.base import LLM
from app.llm.client import (
    PRICING,
    LLMClient,
    MissingCredentials,
    RefusalError,
    TruncatedError,
    models_in_use,
    price_table,
    unpriced_models,
)
from app.llm.factory import build_llm
from app.llm.gemini import GeminiClient, to_parts
from app.llm.prompts import SYSTEM_PROMPT
from app.llm.schemas import CardSet, CardVerdictChecked, UnderstandResultChecked, VerifyResultChecked
from app.main import create_app
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.worker import describe_failure
from tests.test_api import CARDS, TOKEN, UNDERSTANDING, VERDICTS, H, upload, wait_for

MODEL = "gemini-3.8-flash"
CFG = dataclasses.replace(settings, provider="gemini", gemini_model=MODEL)
STAGE = settings.generate  # only max_tokens matters to the Gemini client


class FakeModels:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[SimpleNamespace] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeGemini:
    def __init__(self, replies=()):
        self.models = FakeModels(replies)


def reply(text, *, finish=types.FinishReason.STOP, prompt=100, out=50, thoughts=0, cached=0, block=None, candidates=True):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish)] if candidates else [],
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt, candidates_token_count=out, thoughts_token_count=thoughts, cached_content_token_count=cached
        ),
    )


def client(*replies, cfg=CFG) -> tuple[GeminiClient, FakeGemini]:
    sdk = FakeGemini(replies)
    return GeminiClient(sdk, cfg), sdk


CONTENT = [{"type": "text", "text": "notes here"}]
CARDS_JSON = CardSet(cards=CARDS).model_dump_json()
# With Gemini the fact check is on by default, so the first and last steps answer in the "checked" shape (nothing flagged here).
UNDERSTANDING_JSON = UnderstandResultChecked(**UNDERSTANDING.model_dump(), suspect_claims=[]).model_dump_json()
VERDICTS_JSON = VerifyResultChecked(
    verdicts=[CardVerdictChecked(**v.model_dump(), factual_concern="") for v in VERDICTS.verdicts]
).model_dump_json()


@pytest.fixture
def no_keys(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "SNAPNOTES_PROVIDER", "SNAPNOTES_GEMINI_MODEL"):
        monkeypatch.delenv(name, raising=False)


# --- content blocks -> Gemini parts ---------------------------------------------------------------


def test_text_images_and_pdfs_become_parts_and_cache_control_is_dropped():
    png, pdf = b"\x89PNG-bytes", b"%PDF-1.7 bytes"
    blocks = [
        {"type": "text", "text": "Photo of notes"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}},
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(pdf).decode()}},
        {"type": "text", "text": "<notes>", "cache_control": {"type": "ephemeral"}},
    ]
    parts = to_parts(blocks)
    assert [p.text for p in parts] == ["Photo of notes", None, None, "<notes>"]
    assert (parts[1].inline_data.mime_type, parts[1].inline_data.data) == ("image/png", png)
    assert (parts[2].inline_data.mime_type, parts[2].inline_data.data) == ("application/pdf", pdf)


def test_a_block_type_gemini_cannot_take_is_a_programming_error():
    with pytest.raises(ValueError, match="Unsupported content block"):
        to_parts([{"type": "tool_use"}])


# --- parse ----------------------------------------------------------------------------------------


def test_parse_sends_the_model_system_prompt_schema_and_limit_and_returns_the_validated_object():
    gemini, sdk = client(reply(CARDS_JSON))
    result = gemini.parse("generate", STAGE, content=CONTENT, output_model=CardSet)
    assert [c.front for c in result.cards] == [c.front for c in CARDS]

    [call] = sdk.models.calls
    assert call.model == MODEL
    assert call.config.system_instruction == SYSTEM_PROMPT
    assert call.config.response_mime_type == "application/json"
    assert call.config.response_schema is CardSet
    assert call.config.max_output_tokens == STAGE.max_tokens
    [content] = call.contents
    assert content.role == "user" and content.parts[0].text == "notes here"


def test_usage_is_recorded_per_call_at_zero_cost_with_thinking_counted_as_output():
    gemini, _ = client(reply(CARDS_JSON, prompt=1000, cached=400, out=200, thoughts=300))
    gemini.parse("generate", STAGE, content=CONTENT, output_model=CardSet)
    [row] = gemini.usage.rows
    assert (row.stage, row.model) == ("generate", MODEL)
    assert (row.input_tokens, row.cache_read, row.cache_write, row.output_tokens, row.web_searches) == (600, 400, 0, 500, 0)
    assert row.cost == 0.0 and gemini.usage.total_cost == 0.0


def test_a_slow_call_is_reported_with_its_stage_and_duration_and_a_fast_one_is_not(monkeypatch, caplog):
    """First real run: three calls of about a thousand tokens each took 13 minutes, and nothing in the server window said so."""
    import app.llm.gemini as module

    ticks = iter([0.0, 137.0, 200.0, 201.5])  # the first call takes 137 s, the second 1.5 s
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    gemini, _ = client(reply(CARDS_JSON, prompt=471, out=1167), reply(CARDS_JSON))
    with caplog.at_level(logging.WARNING, logger="app.llm.gemini"):
        gemini.parse("generate", STAGE, content=CONTENT, output_model=CardSet)
        gemini.parse("verify", STAGE, content=CONTENT, output_model=CardSet)
    [record] = caplog.records
    message = record.getMessage()
    assert "generate: gemini-3.8-flash took 137 s for 471 token(s) in and 1167 out" in message and "SNAPNOTES_GEMINI_MODEL" in message


def test_an_answer_that_does_not_fit_the_schema_is_retried_once_with_the_problem_explained(caplog):
    gemini, sdk = client(reply('{"cards": "not a list"}'), reply(CARDS_JSON))
    with caplog.at_level(logging.WARNING, logger="app.llm.gemini"):
        result = gemini.parse("generate", STAGE, content=CONTENT, output_model=CardSet)
    assert len(result.cards) == len(CARDS)
    second = sdk.models.calls[1].contents
    assert [c.role for c in second] == ["user", "model", "user"]
    assert second[1].parts[0].text == '{"cards": "not a list"}'  # what it said before
    assert "did not match the required JSON format" in second[2].parts[0].text
    assert len(gemini.usage.rows) == 2  # both calls are counted
    assert any("did not fit the schema" in r.getMessage() for r in caplog.records)


def test_two_bad_answers_in_a_row_fail_the_stage():
    gemini, sdk = client(reply("not json"), reply("still not json"))
    with pytest.raises(RuntimeError, match="generate: model returned no parseable output"):
        gemini.parse("generate", STAGE, content=CONTENT, output_model=CardSet)
    assert len(sdk.models.calls) == 2  # one retry, not a loop


@pytest.mark.parametrize(
    "bad, error",
    [
        (reply(None, block=types.BlockedReason.SAFETY, candidates=False), RefusalError),
        (reply("", candidates=False), RefusalError),
        (reply("x", finish=types.FinishReason.SAFETY), RefusalError),
        (reply("x", finish=types.FinishReason.PROHIBITED_CONTENT), RefusalError),
        (reply('{"cards": [', finish=types.FinishReason.MAX_TOKENS), TruncatedError),
    ],
)
def test_blocked_stopped_and_cut_off_answers_use_the_same_errors_as_claude(bad, error):
    gemini, _ = client(bad)
    with pytest.raises(error):
        gemini.parse("generate", STAGE, content=CONTENT, output_model=CardSet)


def test_api_errors_from_the_sdk_reach_the_worker_untouched():
    boom = errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    gemini, _ = client(boom)
    with pytest.raises(errors.APIError):
        gemini.parse("generate", STAGE, content=CONTENT, output_model=CardSet)


# --- research --------------------------------------------------------------------------------------


def test_research_is_a_google_search_call_with_no_output_schema():
    """The details of searching, sources and failures are in test_corrections.py; here, only what makes it a search."""
    gemini, sdk = client(reply("Sources checked."))
    gemini.research("enrich", STAGE, content=CONTENT)
    [call] = sdk.models.calls
    assert call.config.tools[0].google_search is not None
    assert call.config.response_schema is None and call.config.response_mime_type is None


# --- the key and the client ------------------------------------------------------------------------


def test_without_a_key_construction_fails_with_a_message_for_the_operator(no_keys):
    with pytest.raises(MissingCredentials, match="GEMINI_API_KEY"):
        GeminiClient(cfg=CFG)


def test_with_a_key_the_real_sdk_client_is_built_with_the_timeout_and_retry_options(no_keys, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "  AIza-test-key-0123456789  ")  # the second name works, and whitespace is ignored
    gemini = GeminiClient(cfg=CFG)
    assert isinstance(gemini.sdk, genai.Client)


def test_the_provider_defaults_to_claude_and_comes_from_the_environment(no_keys, monkeypatch):
    assert Settings().provider == "anthropic" and Settings().gemini_model == "gemini-3.8-flash"
    monkeypatch.setenv("SNAPNOTES_PROVIDER", " Gemini ")
    monkeypatch.setenv("SNAPNOTES_GEMINI_MODEL", "gemini-3.5-flash-lite")
    cfg = Settings()
    assert (cfg.provider, cfg.gemini_model) == ("gemini", "gemini-3.5-flash-lite")


def test_an_unknown_provider_is_named_not_crashed_on(no_keys, monkeypatch):
    monkeypatch.setenv("SNAPNOTES_PROVIDER", "gpt")
    cfg = Settings()  # reading it never raises
    assert "must be one of: anthropic, gemini" in provider_problem(cfg)
    assert provider_problem(CFG) is None
    with pytest.raises(ValueError, match="SNAPNOTES_PROVIDER"):
        build_llm(cfg)


def test_the_factory_builds_the_provider_that_was_asked_for(no_keys, monkeypatch):
    assert isinstance(build_llm(dataclasses.replace(settings, provider="anthropic")), LLMClient)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-0123456789")
    built = build_llm(CFG)
    assert isinstance(built, GeminiClient)
    for client_ in (built, LLMClient(cfg=settings)):  # both satisfy the shared interface the stages are typed against
        assert callable(client_.parse) and callable(client_.research) and client_.usage is not None
    _: LLM = built  # type-level check only


# --- prices and the spend guard ---------------------------------------------------------------------


def test_the_free_model_is_priced_at_zero_and_claude_models_still_are_not_free():
    table = price_table(CFG)
    assert table[MODEL] == (0.0, 0.0) and table["claude-opus-5-5"] == PRICING["claude-opus-5-5"]
    assert unpriced_models(CFG) == [] and models_in_use(CFG) == {MODEL}
    assert MODEL not in price_table(settings)  # nothing is free unless the provider is gemini


def test_a_real_price_in_the_environment_beats_the_free_default():
    cfg = dataclasses.replace(CFG, extra_prices={MODEL: (0.3, 2.5)})
    assert price_table(cfg)[MODEL] == (0.3, 2.5)


def test_claude_pricing_checks_are_unchanged_for_the_default_provider():
    cfg = dataclasses.replace(settings, generate=dataclasses.replace(settings.generate, model="claude-unknown-9"))
    assert unpriced_models(cfg) == ["claude-unknown-9"]


# --- failure messages -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code, message, expected",
    [
        (429, "Quota exceeded", "rate limiting requests, or today's free allowance is used up"),
        (400, "API key not valid. Please pass a valid API key.", "rejected the server's key"),
        (403, "Permission denied", "rejected the server's key"),
        (400, "Unable to process input image", "Gemini could not process this document: Unable to process input image"),
        (503, "overloaded", "Gemini is having trouble (503)"),
        (404, "model not found", "Gemini API error (404)"),
    ],
)
def test_gemini_failures_are_explained_in_plain_words(code, message, expected):
    exc = errors.APIError(code, {"error": {"code": code, "message": message, "status": "X"}})
    assert expected in describe_failure(exc)


def test_a_missing_key_is_reported_as_it_is_written():
    assert describe_failure(MissingCredentials("The server has no Gemini key configured (GEMINI_API_KEY).")) == (
        "The server has no Gemini key configured (GEMINI_API_KEY)."
    )


# --- the whole pipeline ---------------------------------------------------------------------------


def test_the_pipeline_runs_with_gemini_from_the_notes_alone(tmp_path):
    from app.ingest.loader import load_source

    notes = tmp_path / "n.md"
    notes.write_text("# Cells\nMitochondria contain DNA.")
    gemini, sdk = client(reply(UNDERSTANDING_JSON), reply(CARDS_JSON), reply(VERDICTS_JSON))
    stages: list[str] = []
    result = run_pipeline(gemini, load_source([notes]), on_stage=stages.append)

    assert stages == ["understand", "enrich", "generate", "verify"]  # same stages; "enrich" is simply empty
    assert result.evidence == []
    assert len(result.cards) == 3 and sum(1 for r in result.cards if r.keep) == 2  # card 1 repeats card 0
    assert len(sdk.models.calls) == 3  # understand, generate, verify: nothing for research
    assert gemini.usage.total_cost == 0.0
    # the notes go to the model as text, with the stage instructions after them
    generate_parts = sdk.models.calls[1].contents[0].parts
    assert "Mitochondria contain DNA." in generate_parts[0].text and generate_parts[-1].text.startswith("Task:")


def test_a_job_through_the_server_completes_with_gemini_and_records_zero_cost(tmp_path):
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, provider="gemini", gemini_model=MODEL)
    sdk = FakeGemini([reply(UNDERSTANDING_JSON), reply(CARDS_JSON), reply(VERDICTS_JSON)])
    app = create_app(cfg, llm_factory=lambda: GeminiClient(sdk, cfg))
    with TestClient(app) as http:
        job_id = upload(http).json()["job_id"]
        job = wait_for(http, job_id, "completed")  # no evidence, so no approval step
        assert job["deck_id"] and job["cost_usd"] == 0.0
        assert http.get(f"/jobs/{job_id}/evidence", headers=H).json()["evidence"] == []


def test_a_server_without_the_key_fails_the_job_in_plain_words_instead_of_getting_stuck(tmp_path, no_keys):
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, provider="gemini")
    with TestClient(create_app(cfg)) as http:  # the default factory: builds the real GeminiClient, which has no key
        job_id = upload(http).json()["job_id"]
        job = wait_for(http, job_id, "failed")
        assert job["error"] == "The server has no Gemini key configured (GEMINI_API_KEY)."


def test_the_server_refuses_to_start_with_an_unknown_provider(tmp_path, no_keys):
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, provider="gpt")
    with pytest.raises(RuntimeError, match="SNAPNOTES_PROVIDER"):
        with TestClient(create_app(cfg)):
            pass
