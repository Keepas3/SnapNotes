import json
import re

from rapidfuzz import fuzz

from app.core.config import fact_check_enabled
from app.ingest.loader import SourceDoc
from app.llm.base import LLM
from app.llm.evidence import format_evidence
from app.llm.prompts import verify_task
from app.llm.schemas import Card, EvidenceItem, ReviewedCard, VerifyResult, VerifyResultChecked

FLAG_PREFIX = "May be wrong: "  # starts a fact-check concern inside a card's `issue` text
AI_NOTE = "Not sourced: made from the AI's own knowledge. Check it before approving."  # starts the `issue` of an origin "ai" card

# Semantic duplicates are judged by the model (it sees every card). This string match is only a backstop for
# near-identical fronts, so it must be strict: token_set_ratio would call a one-word front like "Osmosis" a
# duplicate of any card mentioning osmosis, and a looser token_sort_ratio would merge "Define mitosis" with
# "Define meiosis".
FUZZY_DUPLICATE_THRESHOLD = 95  # rapidfuzz token_sort_ratio, 0-100


def _norm(text: str) -> str:
    text = re.sub(r"\{\{c\d+::(.*?)(::.*?)?\}\}", r"\1", text)  # cloze markers -> plain answer
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def find_duplicates(cards: list[Card], threshold: int = FUZZY_DUPLICATE_THRESHOLD) -> dict[int, int]:
    """Map duplicate index -> index of the earlier card it repeats."""
    dupes: dict[int, int] = {}
    fronts = [_norm(c.front) for c in cards]
    for j in range(len(cards)):
        for i in range(j):
            if i in dupes:
                continue
            if fuzz.token_sort_ratio(fronts[i], fronts[j]) >= threshold:
                dupes[j] = i
                break
    return dupes


def verify_cards(
    client: LLM, source: SourceDoc, evidence: list[EvidenceItem], cards: list[Card]
) -> list[ReviewedCard]:
    if not cards:
        return []
    checked = fact_check_enabled(client.cfg)  # also asks whether any card contradicts well-established knowledge
    cards_json = json.dumps([{"index": i, **c.model_dump()} for i, c in enumerate(cards)], ensure_ascii=False)
    task = verify_task(format_evidence(evidence), cards_json, fact_check=checked)
    result = client.parse(
        "verify", client.cfg.verify, content=source.content(task), output_model=VerifyResultChecked if checked else VerifyResult
    )
    verdicts = {v.index: v for v in result.verdicts}
    fuzzy = find_duplicates(cards)

    reviewed: list[ReviewedCard] = []
    for i, card in enumerate(cards):
        v = verdicts.get(i)
        # Only trust a model-reported duplicate that points at an earlier card.
        llm_dup = v.duplicate_of if v is not None and 0 <= v.duplicate_of < i else None
        issue = v.issue if v else "not verified"
        concern = (getattr(v, "factual_concern", "") or "").strip()
        if concern:  # shown as a warning; the card stays a draft, so the student decides
            issue = f"{issue} {FLAG_PREFIX}{concern}".strip()
        grounded = v.grounded if v else False  # a missing verdict is treated as unverified
        if card.origin == "ai":
            # A correction made from the model's own knowledge is unsourced by definition, so "not supported by the notes or the
            # evidence" says nothing about it. It stays a draft with a plain label; the concern above is the second opinion.
            grounded = True
            issue = f"{AI_NOTE} {issue}".strip()
        reviewed.append(
            ReviewedCard(
                card=card,
                atomic=v.atomic if v else False,
                grounded=grounded,
                issue=issue,
                duplicate_of=llm_dup if llm_dup is not None else fuzzy.get(i),
            )
        )
    return reviewed
