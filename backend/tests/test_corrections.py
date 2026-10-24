"""Corrected cards: research aimed at the doubtful statements, Gemini's Google Search, and cards made from the right fact."""
import dataclasses
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types

from app.admin import BAD, OK, preflight
from app.core.config import Settings, settings
from app.export.anki import card_tags, source_html
from app.ingest.loader import load_source
from app.llm.client import LLMClient
from app.llm.enrich import enrich
from app.llm.gemini import GeminiClient, MAX_EVIDENCE, evidence_from_grounding
from app.llm.generate import generate_cards
from app.llm.prompts import SYSTEM_PROMPT, enrich_task, generate_task
from app.llm.schemas import Card, CardSet, CardVerdictChecked, UnderstandResultChecked, VerifyResultChecked
from app.llm.verify import AI_NOTE, FLAG_PREFIX, verify_cards
from app.main import create_app
from tests.conftest import FakeSDK, response, text_block
from tests.test_api import TOKEN, UNDERSTANDING, H, upload, wait_for
from tests.test_gemini import CFG, CONTENT, STAGE, FakeGemini, client, reply
from tests.test_pipeline import make_card

CLAIM = "The human heart has three chambers. — It has four."
CLAIM2 = "The chemical symbol for sodium is So. — It is Na."


def grounded(text="Answer.", *, supports=(), chunks=(), queries=("heart chambers",), finish=types.FinishReason.STOP):
    """A Gemini reply that carries Google Search grounding, built from the SDK's own types."""
    meta = types.GroundingMetadata(
        web_search_queries=list(queries),
        grounding_chunks=[types.GroundingChunk(web=types.GroundingChunkWeb(uri=uri, title=title, domain=title)) for uri, title in chunks],
        grounding_supports=[types.GroundingSupport(segment=types.Segment(text=s), grounding_chunk_indices=list(idx)) for s, idx in supports],
    )
    base = reply(text, finish=finish)
    base.candidates = [SimpleNamespace(finish_reason=finish, grounding_metadata=meta)]
    return base


REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"


def src(n: int) -> tuple[str, str]:
    return f"{REDIRECT}{n}", f"site{n}.org"


def understanding(*claims: str) -> UnderstandResultChecked:
    return UnderstandResultChecked(**UNDERSTANDING.model_dump(), suspect_claims=list(claims))


def notes(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("# Biology\nThe human heart has three chambers.")
    return load_source([f])


# --- prompts -----------------------------------------------------------------------------------


def test_the_research_prompt_is_unchanged_without_doubtful_statements_and_names_them_with():
    base = enrich_task("Cells", ["gap"], ["query"])
    assert enrich_task("Cells", ["gap"], ["query"], ()) == base and "look factually wrong" not in base
    with_claims = enrich_task("Cells", ["gap"], ["query"], [CLAIM, CLAIM2])
    assert "These statements from the notes look factually wrong" in with_claims and f"- {CLAIM}\n- {CLAIM2}" in with_claims
    assert with_claims.index(CLAIM) < with_claims.index("Use the web_search tool")
    block = with_claims[with_claims.index("These statements") : with_claims.index("Use the web_search tool")]
    assert with_claims.replace(block, "") == base  # purely additive


def test_card_prompt_offers_ai_corrections_only_when_asked_and_only_with_doubtful_statements():
    plain = generate_task("E", 40)
    assert generate_task("E", 40, (), ai_corrections=True) == plain  # no statements: nothing changes
    sourced_only = generate_task("E", 40, [CLAIM], ai_corrections=False)
    assert 'origin "ai"' not in sourced_only and "Otherwise make no card from it." in sourced_only
    both = generate_task("E", 40, [CLAIM], ai_corrections=True)
    assert 'origin "ai"' in both and 'source_ref "Corrects: "' in both and "make no card from it." in both
    assert "from your own knowledge" in both


# --- enrich ------------------------------------------------------------------------------------


def test_gemini_makes_no_search_unless_something_looks_wrong():
    gemini, sdk = client()
    result = enrich(gemini, UNDERSTANDING)  # no suspect_claims at all (the base shape)
    assert (result.text, result.evidence, result.searches_used) == ("", [], 0) and sdk.models.calls == []
    assert enrich(gemini, understanding()).evidence == [] and sdk.models.calls == []  # an empty list is the same


def test_gemini_searches_when_there_are_doubtful_statements_and_sends_them():
    gemini, sdk = client(grounded(supports=[("The heart has four chambers.", [0])], chunks=[src(1)]))
    result = enrich(gemini, understanding(CLAIM))
    [call] = sdk.models.calls
    assert CLAIM in call.contents[0].parts[0].text and "Use the web_search tool" in call.contents[0].parts[0].text
    assert [e.claim for e in result.evidence] == ["The heart has four chambers."]


def test_claude_still_always_researches_and_the_statements_ride_along_when_the_check_is_on():
    sdk = FakeSDK(create_responses=[response([text_block("Nothing.")]), response([text_block("Nothing.")])])
    claude = LLMClient(sdk, dataclasses.replace(settings, fact_check="1"))
    enrich(claude, UNDERSTANDING)  # no statements: it researches anyway, as before
    enrich(claude, understanding(CLAIM))
    plain, with_claims = (c[2]["messages"][0]["content"][0]["text"] for c in sdk.calls)
    assert "look factually wrong" not in plain and CLAIM in with_claims


# --- Gemini Google Search -----------------------------------------------------------------------


def test_each_supported_sentence_becomes_evidence_once_per_source_at_most_two():
    resp = grounded(
        supports=[
            ("The heart has four chambers.", [0, 1, 2]),  # three sources: two are kept
            ("Sodium's symbol is Na.", [2]),
            ("The heart has four chambers.", [0]),  # the same sentence and source again
        ],
        chunks=[src(1), src(2), src(3)],
    )
    items = evidence_from_grounding(resp)
    assert [(e.claim, e.url, e.title) for e in items] == [
        ("The heart has four chambers.", src(1)[0], "site1.org"),
        ("The heart has four chambers.", src(2)[0], "site2.org"),
        ("Sodium's symbol is Na.", src(3)[0], "site3.org"),
    ]
    assert all(e.snippet == e.claim and e.approved for e in items)


def test_unusable_results_are_dropped():
    resp = grounded(
        supports=[
            ("Not a web link.", [0]),
            ("Out of range index.", [7]),
            ("   ", [1]),  # no text
            ("A fine sentence.", [1]),
        ],
        chunks=[("javascript:alert(1)", "bad"), src(1)],
    )
    assert [e.claim for e in evidence_from_grounding(resp)] == ["A fine sentence."]
    assert evidence_from_grounding(reply("No grounding at all.")) == []
    assert evidence_from_grounding(SimpleNamespace(candidates=[])) == []


def test_a_title_falls_back_to_the_domain_then_the_link():
    chunk = types.GroundingChunk(web=types.GroundingChunkWeb(uri="https://x.org/a", title="", domain="x.org"))
    bare = types.GroundingChunk(web=types.GroundingChunkWeb(uri="https://y.org/b"))
    meta = types.GroundingMetadata(
        grounding_chunks=[chunk, bare],
        grounding_supports=[types.GroundingSupport(segment=types.Segment(text="S."), grounding_chunk_indices=[0, 1])],
    )
    resp = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=meta)])
    assert [e.title for e in evidence_from_grounding(resp)] == ["x.org", "https://y.org/b"]


def test_evidence_is_capped():
    many = [(f"Sentence {i}.", [0]) for i in range(MAX_EVIDENCE + 8)]
    assert len(evidence_from_grounding(grounded(supports=many, chunks=[src(1)]))) == MAX_EVIDENCE


def test_the_search_call_is_configured_as_a_search_and_its_queries_are_recorded():
    gemini, sdk = client(grounded(queries=["heart chambers", "sodium symbol"], supports=[("S.", [0])], chunks=[src(1)]))
    result = gemini.research("enrich", STAGE, content=CONTENT)
    [call] = sdk.models.calls
    assert call.config.system_instruction == SYSTEM_PROMPT and call.config.response_schema is None
    assert call.config.tools[0].google_search.exclude_domains is None
    [row] = gemini.usage.rows
    assert (row.stage, row.web_searches, row.cost) == ("enrich", 2, 0.0) and result.searches_used == 2


def test_blocked_sites_are_left_out_and_allowed_sites_are_reported_as_unsupported(caplog):
    cfg = dataclasses.replace(CFG, web_blocked_domains=("spam.example",), web_allowed_domains=("only.example",))
    gemini, sdk = client(grounded(), cfg=cfg)
    with caplog.at_level(logging.WARNING, logger="app.llm.gemini"):
        gemini.research("enrich", STAGE, content=CONTENT)
    assert sdk.models.calls[0].config.tools[0].google_search.exclude_domains == ["spam.example"]
    assert any("WEB_ALLOWED_DOMAINS is ignored with Gemini" in r.getMessage() for r in caplog.records)


def test_a_search_that_finds_nothing_says_so(caplog):
    gemini, _ = client(grounded(supports=[], chunks=[], queries=["q"]))
    with caplog.at_level(logging.WARNING, logger="app.llm.gemini"):
        result = gemini.research("enrich", STAGE, content=CONTENT)
    assert result.evidence == [] and any("returned no usable sources" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "failure",
    [
        errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
        errors.APIError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}}),
        ConnectionError("no network"),
        TimeoutError("slow"),
    ],
)
def test_a_failed_search_never_fails_the_job(failure, caplog):
    gemini, _ = client(failure)
    with caplog.at_level(logging.WARNING, logger="app.llm.gemini"):
        result = gemini.research("enrich", STAGE, content=CONTENT)
    assert (result.text, result.evidence, result.searches_used) == ("", [], 0)
    assert any("Google Search did not work" in r.getMessage() and "AI's own knowledge" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "failure, expected",
    [
        (errors.APIError(429, {"error": {"code": 429, "message": "Quota exceeded for metric", "status": "RESOURCE_EXHAUSTED"}}), "free search allowance is used up"),
        (errors.APIError(400, {"error": {"code": 400, "message": "Search grounding is not supported for this model", "status": "INVALID_ARGUMENT"}}), "refused the search for this key or model"),
        (errors.APIError(403, {"error": {"code": 403, "message": "API key not valid", "status": "PERMISSION_DENIED"}}), "rejected the key"),
        (errors.APIError(503, {"error": {"code": 503, "message": "Overloaded", "status": "UNAVAILABLE"}}), "Google answered 503: Overloaded"),
        (ConnectionError("no network"), "ConnectionError: no network"),
    ],
)
def test_a_failed_search_leaves_a_note_in_plain_words_for_the_app(failure, expected):
    gemini, _ = client(failure)
    result = gemini.research("enrich", STAGE, content=CONTENT)
    assert result.evidence == [] and result.note.startswith("The web search did not work (") and expected in result.note
    assert result.note.endswith("so the corrections come from the AI's own knowledge.")


def test_a_search_with_no_sources_says_so_and_a_good_one_leaves_no_note():
    empty, _ = client(grounded(supports=[], chunks=[]))
    assert empty.research("enrich", STAGE, content=CONTENT).note == "The web search found no usable sources, so the corrections come from the AI's own knowledge."
    good, _ = client(grounded(supports=[("S.", [0])], chunks=[src(1)]))
    assert good.research("enrich", STAGE, content=CONTENT).note == ""


def test_the_note_reaches_the_app_with_the_job(tmp_path):
    sdk = FakeGemini(
        [
            reply(understanding(CLAIM2).model_dump_json()),
            errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
            reply(CardSet(cards=[ai_card()]).model_dump_json()),
            reply(verdicts(OK_VERDICT).model_dump_json()),
        ]
    )
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, provider="gemini", gemini_model=CFG.gemini_model)
    with TestClient(create_app(cfg, llm_factory=lambda: GeminiClient(sdk, cfg))) as http:
        job_id = upload(http).json()["job_id"]
        job = wait_for(http, job_id, "completed")
    assert job["understanding"]["suspect_claims"] == [CLAIM2]
    assert job["understanding"]["research_note"].startswith("The web search did not work (Google's free search allowance is used up")


def test_a_job_without_a_search_problem_has_no_note(tmp_path):
    sdk = FakeGemini([reply(understanding().model_dump_json()), reply(CardSet(cards=[make_card("Q")]).model_dump_json()), reply(verdicts(OK_VERDICT).model_dump_json())])
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, provider="gemini", gemini_model=CFG.gemini_model)
    with TestClient(create_app(cfg, llm_factory=lambda: GeminiClient(sdk, cfg))) as http:
        job_id = upload(http).json()["job_id"]
        assert wait_for(http, job_id, "completed")["understanding"]["research_note"] == ""


def test_a_blocked_search_prompt_degrades_the_same_way():
    gemini, _ = client(reply(None, finish=types.FinishReason.SAFETY))
    assert gemini.research("enrich", STAGE, content=CONTENT).evidence == []


def test_search_responses_are_saved_when_a_dump_folder_is_set(tmp_path):
    cfg = dataclasses.replace(CFG, debug_dump_dir=tmp_path / "dump")
    gemini, _ = client(grounded(supports=[("S.", [0])], chunks=[src(1)]), cfg=cfg)
    gemini.research("enrich", STAGE, content=CONTENT)
    [saved] = (tmp_path / "dump").iterdir()
    assert "research-enrich" in saved.name
    assert json.loads(saved.read_text(encoding="utf-8"))["candidates"][0]["grounding_metadata"]["web_search_queries"] == ["heart chambers"]


# --- the cards ---------------------------------------------------------------------------------


def test_the_card_step_passes_the_ai_corrections_setting_into_the_prompt(tmp_path):
    cards = CardSet(cards=[make_card("Q")])
    for flag, expected in ((True, True), (False, False)):
        sdk = FakeSDK(parse_results={CardSet: cards})
        generate_cards(LLMClient(sdk, dataclasses.replace(settings, ai_corrections=flag)), notes(tmp_path), [], [CLAIM])
        assert ('origin "ai"' in sdk.calls[0][2]["messages"][0]["content"][-1]["text"]) is expected


def test_the_setting_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("SNAPNOTES_AI_CORRECTIONS", raising=False)
    assert Settings().ai_corrections is True
    for value, expected in (("0", False), ("off", False), (" No ", False), ("1", True), ("yes", True)):
        monkeypatch.setenv("SNAPNOTES_AI_CORRECTIONS", value)
        assert Settings().ai_corrections is expected


def ai_card(front="What is the chemical symbol for sodium?") -> Card:
    return Card(type="basic", front=front, back="Na. (The notes said So.)", origin="ai", source_ref="Corrects: The chemical symbol for sodium is So",
                source_snippet="", tags=["chemistry"], difficulty="easy")


def verdicts(*entries):
    return VerifyResultChecked(verdicts=[CardVerdictChecked(index=i, **e) for i, e in enumerate(entries)])


OK_VERDICT = dict(atomic=True, grounded=True, issue="", duplicate_of=-1, factual_concern="")


def test_an_ai_correction_is_not_set_aside_as_ungrounded_and_carries_its_label(tmp_path):
    cards = [make_card("Notes card"), ai_card()]
    verdict = verdicts(OK_VERDICT, {**OK_VERDICT, "grounded": False, "issue": "Not in the notes."})  # the verifier says: not supported
    reviewed = verify_cards(LLMClient(FakeSDK(parse_results={VerifyResultChecked: verdict}), dataclasses.replace(settings, fact_check="1")), notes(tmp_path), [], cards)
    notes_card, ai = reviewed
    assert notes_card.keep and notes_card.issue == ""  # unchanged
    assert ai.grounded and ai.keep and ai.issue == f"{AI_NOTE} Not in the notes."


def test_a_second_opinion_that_disagrees_is_added_to_the_label(tmp_path):
    verdict = verdicts({**OK_VERDICT, "factual_concern": "Sodium's symbol is Na, but check the second part."})
    [ai] = verify_cards(LLMClient(FakeSDK(parse_results={VerifyResultChecked: verdict}), dataclasses.replace(settings, fact_check="1")), notes(tmp_path), [], [ai_card()])
    assert ai.issue == f"{AI_NOTE} {FLAG_PREFIX}Sodium's symbol is Na, but check the second part." and ai.keep


def test_an_ai_card_with_no_verdict_is_still_a_labelled_draft_and_a_duplicate_is_still_set_aside(tmp_path):
    client_ = LLMClient(FakeSDK(parse_results={VerifyResultChecked: verdicts(OK_VERDICT)}), dataclasses.replace(settings, fact_check="1"))
    first, missing = verify_cards(client_, notes(tmp_path), [], [make_card("Notes card"), ai_card()])  # the second has no verdict
    assert missing.keep and missing.issue.startswith(AI_NOTE) and "not verified" in missing.issue

    duplicate = verdicts(OK_VERDICT, {**OK_VERDICT, "duplicate_of": 0})
    client_ = LLMClient(FakeSDK(parse_results={VerifyResultChecked: duplicate}), dataclasses.replace(settings, fact_check="1"))
    _, dup = verify_cards(client_, notes(tmp_path), [], [make_card("What is the symbol for sodium?"), ai_card()])
    assert not dup.keep and dup.duplicate_of == 0


# --- through the server ------------------------------------------------------------------------


def test_a_job_with_doubtful_statements_searches_asks_for_approval_then_makes_sourced_and_labelled_cards(tmp_path):
    notes_card = make_card("What do mitochondria make?")
    web_card = Card(type="basic", front="How many chambers does the human heart have?", back="Four.", origin="web", source_ref=src(1)[0],
                    source_snippet="The heart has four chambers.", tags=["heart"], difficulty="easy")
    cards = CardSet(cards=[notes_card, web_card, ai_card()])
    checked = verdicts(OK_VERDICT, OK_VERDICT, {**OK_VERDICT, "grounded": False, "issue": "Not in the notes."})
    sdk = FakeGemini(
        [
            reply(understanding(CLAIM, CLAIM2).model_dump_json()),
            grounded(supports=[("The human heart has four chambers.", [0])], chunks=[src(1)]),
            reply(cards.model_dump_json()),
            reply(checked.model_dump_json()),
        ]
    )
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, provider="gemini", gemini_model=CFG.gemini_model)
    with TestClient(create_app(cfg, llm_factory=lambda: GeminiClient(sdk, cfg))) as http:
        job_id = upload(http).json()["job_id"]
        wait_for(http, job_id, "awaiting_approval")  # a source was found, so you are asked, as on the Claude path
        listed = http.get(f"/jobs/{job_id}/evidence", headers=H).json()
        assert listed["understanding"]["suspect_claims"] == [CLAIM, CLAIM2]
        assert [(e["title"], e["url"]) for e in listed["evidence"]] == [("site1.org", src(1)[0])]

        assert http.post(f"/jobs/{job_id}/approve", headers=H, json={"rejected_ids": []}).status_code == 200
        job = wait_for(http, job_id, "completed")
        by_front = {c["front"]: c for c in http.get(f"/decks/{job['deck_id']}/cards", headers=H).json()}

    web = by_front["How many chambers does the human heart have?"]
    ai = by_front["What is the chemical symbol for sodium?"]
    assert (web["origin"], web["status"], web["source_ref"]) == ("web", "draft", src(1)[0])
    assert (ai["origin"], ai["status"], ai["grounded"]) == ("ai", "draft", True)
    assert ai["issue"].startswith("Not sourced: made from the AI's own knowledge") and ai["source_ref"].startswith("Corrects: ")
    generate_prompt = sdk.models.calls[2].contents[0].parts[-1].text
    assert CLAIM in generate_prompt and 'origin "ai"' in generate_prompt and "The human heart has four chambers." in generate_prompt


def test_when_the_search_fails_the_job_still_finishes_with_labelled_corrections(tmp_path):
    cards = CardSet(cards=[ai_card()])
    sdk = FakeGemini(
        [
            reply(understanding(CLAIM2).model_dump_json()),
            errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
            reply(cards.model_dump_json()),
            reply(verdicts(OK_VERDICT).model_dump_json()),
        ]
    )
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, provider="gemini", gemini_model=CFG.gemini_model)
    with TestClient(create_app(cfg, llm_factory=lambda: GeminiClient(sdk, cfg))) as http:
        job_id = upload(http).json()["job_id"]
        job = wait_for(http, job_id, "completed")  # no sources, so no approval step
        [card] = http.get(f"/decks/{job['deck_id']}/cards", headers=H).json()
    assert card["origin"] == "ai" and card["status"] == "draft"


def test_a_job_without_doubtful_statements_makes_no_search_call(tmp_path):
    sdk = FakeGemini([reply(understanding().model_dump_json()), reply(CardSet(cards=[make_card("Q")]).model_dump_json()), reply(verdicts(OK_VERDICT).model_dump_json())])
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, provider="gemini", gemini_model=CFG.gemini_model)
    with TestClient(create_app(cfg, llm_factory=lambda: GeminiClient(sdk, cfg))) as http:
        job_id = upload(http).json()["job_id"]
        wait_for(http, job_id, "completed")
    assert len(sdk.models.calls) == 3 and all(c.config.tools is None for c in sdk.models.calls)  # understand, generate, verify


def test_demo_mode_shows_a_labelled_ai_correction_only_when_the_check_is_on(tmp_path):
    from tests.test_factcheck import run_demo

    job, cards = run_demo(tmp_path / "on", fact_check="1")
    assert len(job["understanding"]["suspect_claims"]) == 2
    [ai] = [c for c in cards if c["origin"] == "ai"]
    assert ai["status"] == "draft" and ai["issue"].startswith(AI_NOTE) and ai["source_ref"].startswith("Corrects: ")

    job, cards = run_demo(tmp_path / "off")
    assert not [c for c in cards if c["origin"] == "ai"] and len(cards) == 8


# --- exports and the check command -----------------------------------------------------------------


def test_exports_label_an_ai_correction_honestly():
    card = {"origin": "ai", "source_ref": "Corrects: Sodium is So", "tags": ["chemistry"]}
    assert source_html(card) == "From AI knowledge (not sourced): Corrects: Sodium is So"
    assert source_html({**card, "source_ref": ""}) == "From AI knowledge (not sourced)"
    assert "source-ai" in card_tags(card)
    hostile = {**card, "source_ref": "<img src=x onerror=alert(1)>"}
    assert "<img" not in source_html(hostile)


def search_line(factory, **changes):
    cfg = dataclasses.replace(settings, demo_mode=False, api_token="T" * 43, provider="gemini", **changes)
    return preflight(cfg, search=True, gemini_factory=factory)[-1]


def test_check_search_runs_one_real_style_search_and_names_what_came_back():
    working = FakeGemini([grounded(queries=["boiling point of water"], supports=[("Water boils at 100 C.", [0, 1])], chunks=[src(1), src(2)])])
    level, text = search_line(lambda: working)
    assert level == OK and text.startswith("Google Search works: 1 search(es), 2 source(s) from site1.org, site2.org.")
    [call] = working.models.calls
    assert call.config.tools[0].google_search is not None and "boil" in call.contents.lower()  # a harmless question, and the search tool


def test_check_search_shows_the_reason_instead_of_hiding_it():
    refused = errors.APIError(400, {"error": {"code": 400, "message": "Search tool is not supported for this model", "status": "INVALID_ARGUMENT"}})
    level, text = search_line(lambda: FakeGemini([refused]))
    assert level == BAD and "Google Search through gemini-3.8-flash failed: Google refused the search for this key or model" in text

    level, text = search_line(lambda: FakeGemini([grounded(supports=[], chunks=[], queries=["q"])]))
    assert level != OK and "returned no usable sources (1 search(es))" in text


def test_check_search_only_applies_to_gemini():
    cfg = dataclasses.replace(settings, demo_mode=False, api_token="T" * 43, provider="anthropic")
    level, text = preflight(cfg, search=True, client_factory=lambda: SimpleNamespace(api_key="x", auth_token=None, credentials=None, custom_auth=None))[-1]
    assert text.startswith("--search tests Gemini's Google Search only")


def test_the_check_command_reports_the_ai_corrections_setting():
    def lines(**changes):
        cfg = dataclasses.replace(settings, demo_mode=False, api_token="T" * 43, provider="gemini", **changes)
        return [text for _, text in preflight(cfg, client_factory=lambda: None) if text.startswith("AI corrections")]

    [on] = lines(ai_corrections=True)
    assert on.startswith("AI corrections: on.") and "not sourced" in on
    assert lines(ai_corrections=False) == ["AI corrections: off. Corrected cards are made only when a web source backs them."]
    assert lines(fact_check="0") == []  # nothing to say when the fact check is off
