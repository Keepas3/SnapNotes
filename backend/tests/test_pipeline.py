from app.ingest.loader import load_source
from app.llm.client import LLMClient
from app.llm.schemas import (
    Card,
    CardSet,
    CardVerdict,
    EvidenceItem,
    UnderstandResult,
    VerifyResult,
)
from app.llm.verify import find_duplicates, verify_cards
from app.pipeline.orchestrator import run_pipeline
from tests.conftest import FakeSDK, citation, response, text_block


def make_card(front: str, origin="notes") -> Card:
    return Card(
        type="basic", front=front, back="answer", origin=origin, source_ref="p.1",
        source_snippet="snippet", tags=["bio"], difficulty="easy",
    )


def verdict(index: int, *, grounded=True, duplicate_of=-1, issue="") -> CardVerdict:
    return CardVerdict(index=index, atomic=True, grounded=grounded, issue=issue, duplicate_of=duplicate_of)


UNDERSTANDING = UnderstandResult(
    main_topic="Cell biology", subject_area="biology", subtopics=["Mitochondria"],
    key_terms=["ATP"], gaps=["Notes say mitochondria make DNA only"], search_queries=["mitochondria function"],
)


def test_fuzzy_backstop_only_catches_near_identical_fronts():
    cards = [
        make_card("What organelle produces ATP?"),
        make_card("what organelle produces ATP"),  # same up to case and punctuation
        make_card("Which organelle produces ATP?"),  # a paraphrase is the model's call, not string matching
    ]
    assert find_duplicates(cards) == {1: 0}


def test_fuzzy_backstop_does_not_merge_distinct_concepts():
    cards = [
        make_card("Osmosis"),
        make_card("Water moves by osmosis from low to high solute concentration"),  # token_set_ratio would say 100
        make_card("Define mitosis"),
        make_card("Define meiosis"),
    ]
    assert find_duplicates(cards) == {}


def test_verify_uses_model_duplicates_and_ignores_bad_pointers(tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("notes")
    source = load_source([f])
    cards = [make_card("What does ATP stand for?"), make_card("Expand the acronym ATP"), make_card("Define osmosis"), make_card("Define diffusion")]
    verdicts = VerifyResult(verdicts=[
        verdict(0),
        verdict(1, duplicate_of=0),  # real paraphrase duplicate
        verdict(2, duplicate_of=3),  # points forward: ignored
        verdict(3, duplicate_of=3),  # points at itself: ignored
    ])
    reviewed = verify_cards(LLMClient(FakeSDK(parse_results={VerifyResult: verdicts})), source, [], cards)
    assert [r.duplicate_of for r in reviewed] == [None, 0, None, None]
    assert [r.keep for r in reviewed] == [True, False, True, True]


def test_verify_marks_missing_verdicts_unverified(tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("notes")
    source = load_source([f])
    cards = [make_card("What organelle produces ATP?"), make_card("Define osmosis")]
    sdk = FakeSDK(parse_results={VerifyResult: VerifyResult(verdicts=[verdict(0)])})  # index 1 missing on purpose
    reviewed = verify_cards(LLMClient(sdk), source, [], cards)
    assert [r.keep for r in reviewed] == [True, False]
    assert reviewed[1].issue == "not verified"


def test_full_pipeline_with_fake_sdk(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("# Cells\nMitochondria contain DNA.")
    source = load_source([f])

    web_card = make_card("What do mitochondria produce?", origin="web")
    sdk = FakeSDK(
        parse_results={
            UnderstandResult: UNDERSTANDING,
            CardSet: CardSet(cards=[make_card("What do mitochondria contain?"), web_card]),
            VerifyResult: VerifyResult(verdicts=[verdict(0), verdict(1)]),
        },
        create_responses=[response([text_block("- Mitochondria produce ATP.", [citation("https://a.org", "A", "produce ATP")])])],
    )
    stages: list[str] = []

    def approve(items: list[EvidenceItem]) -> list[EvidenceItem]:
        items[0].approved = True
        return items

    result = run_pipeline(LLMClient(sdk), source, approve_evidence=approve, on_stage=stages.append)

    assert stages == ["understand", "enrich", "generate", "verify"]
    assert [e.url for e in result.evidence] == ["https://a.org"]
    assert len(result.cards) == 2 and all(r.keep for r in result.cards)
    # web research never receives the notes themselves
    enrich_content = next(kw for _, kind, kw in sdk.calls if kind == "create")["messages"][0]["content"]
    assert all("Mitochondria contain DNA" not in b.get("text", "") for b in enrich_content)
    # the generate prompt includes the approved evidence
    gen_kw = next(kw for _, kind, kw in sdk.calls if kind == "parse" and kw["output_format"] is CardSet)
    assert "Mitochondria produce ATP" in gen_kw["messages"][0]["content"][-1]["text"]
    assert len(result.usage.rows) == 4
