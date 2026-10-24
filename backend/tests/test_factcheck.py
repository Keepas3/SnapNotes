"""The fact check: statements in the notes that look factually wrong are listed, get no card, and cards the check doubts are flagged."""
import dataclasses
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.admin import BAD, OK, preflight
from app.core.config import Settings, fact_check_enabled, fact_check_problem, settings
from app.ingest.loader import load_source
from app.llm.client import LLMClient
from app.llm.prompts import generate_task, understand_task, verify_task
from app.llm.schemas import (
    Card,
    CardSet,
    CardVerdict,
    CardVerdictChecked,
    UnderstandResult,
    UnderstandResultChecked,
    VerifyResult,
    VerifyResultChecked,
)
from app.llm.understand import understand
from app.llm.verify import FLAG_PREFIX, verify_cards
from app.main import create_app
from app.pipeline.orchestrator import run_generation
from tests.conftest import FakeSDK, response, text_block
from tests.test_api import CARDS, TOKEN, UNDERSTANDING, VERDICTS, H, make_client, make_sdk, upload, wait_for
from tests.test_pipeline import make_card, verdict

CLAIM = "Ribosomes are made of protein only — ribosomes also contain ribosomal RNA."
ON = dataclasses.replace(settings, fact_check="1")
OFF = dataclasses.replace(settings, fact_check="0")


def checked_understanding(*claims: str) -> UnderstandResultChecked:
    return UnderstandResultChecked(**UNDERSTANDING.model_dump(), suspect_claims=list(claims))


def checked_verdicts(concerns: dict[int, str], n: int = 3) -> VerifyResultChecked:
    return VerifyResultChecked(
        verdicts=[
            CardVerdictChecked(index=i, atomic=True, grounded=True, issue="", duplicate_of=-1, factual_concern=concerns.get(i, ""))
            for i in range(n)
        ]
    )


def notes(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("# Cells\nRibosomes are made of protein only.")
    return load_source([f])


def prompt_of(call) -> str:
    return call[2]["messages"][0]["content"][-1]["text"]  # the stage instructions are the last block


# --- the setting ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider, value, expected",
    [
        ("gemini", "auto", True),
        ("anthropic", "auto", False),
        ("anthropic", "1", True),
        ("anthropic", "ON", True),
        ("gemini", "0", False),
        ("gemini", "off", False),
    ],
)
def test_the_check_is_on_for_the_free_provider_and_off_for_claude_unless_told_otherwise(provider, value, expected):
    cfg = dataclasses.replace(settings, provider=provider, fact_check=value.lower())
    assert fact_check_enabled(cfg) is expected and fact_check_problem(cfg) is None


def test_the_setting_comes_from_the_environment_and_a_typo_is_named(monkeypatch):
    monkeypatch.delenv("SNAPNOTES_FACT_CHECK", raising=False)
    assert Settings().fact_check == "auto"
    monkeypatch.setenv("SNAPNOTES_FACT_CHECK", " Maybe ")
    problem = fact_check_problem(Settings())
    assert problem == "SNAPNOTES_FACT_CHECK is 'maybe'; use auto, 1 or 0."


def test_the_server_refuses_to_start_with_a_mistyped_setting(tmp_path):
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, fact_check="perhaps")
    with pytest.raises(RuntimeError, match="SNAPNOTES_FACT_CHECK"):
        with TestClient(create_app(cfg)):
            pass


# --- Claude's requests stay exactly as they were when the check is off -------------------------------


def test_prompts_are_unchanged_when_the_check_is_off():
    assert "suspect_claims" not in understand_task() and understand_task() == understand_task(fact_check=False)
    assert "factual_concern" not in verify_task("E", "[]") and verify_task("E", "[]") == verify_task("E", "[]", fact_check=False)
    assert "doubtful" not in generate_task("E", 40) and generate_task("E", 40) == generate_task("E", 40, ())


def test_the_added_prompt_text_is_purely_additive():
    """Strip what the check adds and the original prompt text is left, character for character."""
    base_understand, base_verify, base_generate = understand_task(), verify_task("E", "[]"), generate_task("E", 40)
    assert understand_task(fact_check=True).startswith(base_understand) and len(understand_task(fact_check=True)) > len(base_understand)
    assert verify_task("E", "[]", fact_check=True).startswith(base_verify)
    with_claims = generate_task("E", 40, [CLAIM])
    block = with_claims[with_claims.index("<doubtful_statements>") : with_claims.index("Card rules:")]
    assert with_claims.replace(block, "") == base_generate


def test_with_the_check_off_the_stages_ask_for_the_original_output_models(tmp_path):
    sdk = FakeSDK(parse_results={UnderstandResult: UNDERSTANDING, VerifyResult: VERDICTS})
    client = LLMClient(sdk, OFF)
    understand(client, notes(tmp_path))
    verify_cards(client, notes(tmp_path), [], [make_card("Q1"), make_card("Q2"), make_card("Q3")])
    assert [kw["output_format"] for _, _, kw in sdk.calls] == [UnderstandResult, VerifyResult]
    assert "suspect_claims" not in prompt_of(sdk.calls[0]) and "factual_concern" not in prompt_of(sdk.calls[1])


# --- with the check on -----------------------------------------------------------------------------


def test_with_the_check_on_the_first_step_asks_for_suspect_claims_in_the_checked_shape(tmp_path):
    sdk = FakeSDK(parse_results={UnderstandResultChecked: checked_understanding(CLAIM)})
    result = understand(LLMClient(sdk, ON), notes(tmp_path))
    assert sdk.calls[0][2]["output_format"] is UnderstandResultChecked
    assert "suspect_claims" in prompt_of(sdk.calls[0]) and "clearly factually wrong" in prompt_of(sdk.calls[0])
    assert result.suspect_claims == [CLAIM]


def test_doubtful_statements_reach_the_card_writing_prompt_only_when_there_are_some():
    with_claims, without = generate_task("E", 40, [CLAIM, "Water boils at 50 C — at sea level it boils at 100 C."]), generate_task("E", 40, [])
    assert "<doubtful_statements>\n- " + CLAIM + "\n- Water boils" in with_claims
    assert "Never state one of them as true" in with_claims and 'make no card from it' in with_claims
    assert "doubtful" not in without


def test_a_concern_is_added_to_the_cards_issue_and_the_card_stays_a_draft(tmp_path):
    cards = [make_card("Q1"), make_card("Q2"), make_card("Q3")]
    verdicts = VerifyResultChecked(
        verdicts=[
            CardVerdictChecked(index=0, atomic=True, grounded=True, issue="", duplicate_of=-1, factual_concern="Ribosomes also contain rRNA."),
            CardVerdictChecked(index=1, atomic=True, grounded=True, issue="Minor wording.", duplicate_of=-1, factual_concern="Water boils at 100 C."),
            CardVerdictChecked(index=2, atomic=True, grounded=True, issue="", duplicate_of=-1, factual_concern="  "),  # blank: nothing to flag
        ]
    )
    reviewed = verify_cards(LLMClient(FakeSDK(parse_results={VerifyResultChecked: verdicts}), ON), notes(tmp_path), [], cards)
    assert [r.issue for r in reviewed] == [
        FLAG_PREFIX + "Ribosomes also contain rRNA.",
        "Minor wording. " + FLAG_PREFIX + "Water boils at 100 C.",
        "",
    ]
    assert all(r.keep for r in reviewed)  # flagged cards still reach the review queue; the student decides


def test_a_missing_verdict_is_still_marked_unverified_with_the_check_on(tmp_path):
    verdicts = VerifyResultChecked(verdicts=[CardVerdictChecked(index=0, atomic=True, grounded=True, issue="", duplicate_of=-1, factual_concern="")])
    reviewed = verify_cards(LLMClient(FakeSDK(parse_results={VerifyResultChecked: verdicts}), ON), notes(tmp_path), [], [make_card("Q1"), make_card("Q2")])
    assert reviewed[1].issue == "not verified" and not reviewed[1].grounded


def test_the_generation_phase_passes_the_claims_to_the_card_prompt(tmp_path):
    sdk = FakeSDK(parse_results={CardSet: CardSet(cards=CARDS), VerifyResultChecked: checked_verdicts({})})
    run_generation(LLMClient(sdk, ON), notes(tmp_path), [], suspect_claims=[CLAIM])
    generate_call = next(c for c in sdk.calls if c[2]["output_format"] is CardSet)
    assert "<doubtful_statements>" in prompt_of(generate_call) and CLAIM in prompt_of(generate_call)


# --- through the server ---------------------------------------------------------------------------


def test_a_job_lists_the_claims_skips_them_and_flags_doubtful_cards(tmp_path):
    sdk = FakeSDK(
        parse_results={
            UnderstandResultChecked: checked_understanding(CLAIM),
            CardSet: CardSet(cards=CARDS),
            VerifyResultChecked: checked_verdicts({2: "Osmosis is described the wrong way round."}),
        },
        create_responses=[response([text_block("Nothing relevant found.")])],  # web research finds nothing
    )
    with make_client(tmp_path, sdk=sdk, fact_check="1") as client:
        job_id = upload(client).json()["job_id"]
        job = wait_for(client, job_id, "completed")  # no evidence, so no approval step
        assert job["understanding"]["suspect_claims"] == [CLAIM]  # the app can show them

        generate_call = next(c for c in sdk.calls if c[2].get("output_format") is CardSet)
        assert CLAIM in prompt_of(generate_call)  # the second phase read the claims back from the database

        cards = client.get(f"/decks/{job['deck_id']}/cards", headers=H).json()
        flagged = [c for c in cards if "May be wrong" in c["issue"]]
        assert [c["front"] for c in flagged] == ["Define osmosis"] and flagged[0]["status"] == "draft"


def test_jobs_without_the_check_report_an_empty_list(tmp_path):
    with make_client(tmp_path, sdk=make_sdk(), fact_check="0") as client:
        job_id = upload(client, auto_approve="true").json()["job_id"]
        assert wait_for(client, job_id, "completed")["understanding"]["suspect_claims"] == []


def test_claims_are_listed_while_the_job_waits_for_source_approval(tmp_path):
    with make_client(tmp_path, sdk=_sdk_with_one_source(), fact_check="1") as client:
        job_id = upload(client).json()["job_id"]
        wait_for(client, job_id, "awaiting_approval")
        assert client.get(f"/jobs/{job_id}/evidence", headers=H).json()["understanding"]["suspect_claims"] == [CLAIM]


def _sdk_with_one_source() -> FakeSDK:
    from tests.conftest import citation

    return FakeSDK(
        parse_results={UnderstandResultChecked: checked_understanding(CLAIM), CardSet: CardSet(cards=CARDS), VerifyResultChecked: checked_verdicts({})},
        create_responses=[response([text_block("- Ribosomes contain rRNA.", [citation("https://example.org/r", "R", "rRNA")])])],
    )


def test_the_store_returns_no_claims_for_a_job_without_any(tmp_path):
    from app.core.db import Database
    from app.store.jobs import JobStore

    db = Database(tmp_path / "x.db")
    db.init()
    store = JobStore(db)
    job_id = store.create(doc_id="d1", filename="n.md", kind="text", stored_names=["n.md"], auto_approve=False)
    assert store.suspect_claims(job_id) == [] and store.suspect_claims("no-such-job") == []  # no understanding saved yet


# --- demo mode (the emulator can show the feature without any key) ---------------------------------


def run_demo(tmp_path, **overrides):
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, demo_mode=True, demo_delay=0, **overrides)
    with TestClient(create_app(cfg)) as client:
        job_id = upload(client, auto_approve="true").json()["job_id"]
        job = wait_for(client, job_id, "completed")
        return job, client.get(f"/decks/{job['deck_id']}/cards", headers=H).json()


def test_demo_mode_shows_the_feature_when_asked_and_not_otherwise(tmp_path):
    job, cards = run_demo(tmp_path / "on", fact_check="1")
    assert job["understanding"]["suspect_claims"] and "Ribosomes" in job["understanding"]["suspect_claims"][0]
    assert [c["issue"] for c in cards if "May be wrong" in c["issue"]]

    job, cards = run_demo(tmp_path / "off")
    assert job["understanding"]["suspect_claims"] == [] and not [c for c in cards if "May be wrong" in c["issue"]]


# --- admin check -----------------------------------------------------------------------------------


def _check_lines(**changes):
    cfg = dataclasses.replace(settings, demo_mode=False, api_token="T" * 43, **changes)
    return [(level, text) for level, text in preflight(cfg, client_factory=lambda: SimpleNamespace(api_key="x", auth_token=None, credentials=None, custom_auth=None)) if "Fact check" in text or "SNAPNOTES_FACT_CHECK" in text]


def test_the_check_command_says_whether_the_fact_check_is_on_and_why():
    [(level, text)] = _check_lines(provider="gemini")
    assert level == OK and "Fact check: on (on by default for the free Gemini provider)" in text and "can miss errors" in text
    [(level, text)] = _check_lines(provider="anthropic")
    assert level == OK and text.startswith("Fact check: off")
    [(level, text)] = _check_lines(provider="anthropic", fact_check="1")
    assert level == OK and "switched on with SNAPNOTES_FACT_CHECK" in text
    [(level, text)] = _check_lines(fact_check="sometimes")
    assert level == BAD and "use auto, 1 or 0" in text
