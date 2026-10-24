from typing import Sequence

from app.ingest.loader import SourceDoc
from app.llm.base import LLM
from app.llm.evidence import format_evidence
from app.llm.prompts import generate_task
from app.llm.schemas import Card, CardSet, EvidenceItem


def generate_cards(client: LLM, source: SourceDoc, evidence: list[EvidenceItem], suspect_claims: Sequence[str] = ()) -> list[Card]:
    cfg = client.cfg
    task = generate_task(format_evidence(evidence), cfg.max_cards, suspect_claims, ai_corrections=cfg.ai_corrections)
    result = client.parse("generate", cfg.generate, content=source.content(task), output_model=CardSet)
    return result.cards[: cfg.max_cards]
