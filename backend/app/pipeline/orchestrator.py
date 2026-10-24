"""understand -> enrich -> (user approves sources) -> generate -> verify.

The work is split in two phases so the job worker can pause for source approval between them:
`run_research` (understand + enrich) and `run_generation` (generate + verify). `run_pipeline` composes both
for the spike CLI and for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from app.ingest.loader import SourceDoc
from app.llm.base import LLM
from app.llm.client import UsageLog
from app.llm.enrich import enrich
from app.llm.generate import generate_cards
from app.llm.schemas import EvidenceItem, ReviewedCard, UnderstandResult
from app.llm.understand import understand
from app.llm.verify import verify_cards

ApproveFn = Callable[[list[EvidenceItem]], list[EvidenceItem]]
ProgressFn = Callable[[str], None]


def _noop(_stage: str) -> None:
    pass


@dataclass
class ResearchOutcome:
    understanding: UnderstandResult
    research_text: str
    evidence: list[EvidenceItem]
    note: str = ""  # set when a wanted web search failed or found nothing


@dataclass
class PipelineResult:
    understanding: UnderstandResult
    research_text: str
    evidence: list[EvidenceItem]
    cards: list[ReviewedCard]
    usage: UsageLog


def run_research(client: LLM, source: SourceDoc, *, on_stage: ProgressFn | None = None) -> ResearchOutcome:
    progress = on_stage or _noop
    progress("understand")
    understanding = understand(client, source)
    progress("enrich")
    research = enrich(client, understanding)
    return ResearchOutcome(understanding, research.text, research.evidence, research.note)


def run_generation(
    client: LLM,
    source: SourceDoc,
    evidence: list[EvidenceItem],
    *,
    suspect_claims: Sequence[str] = (),
    on_stage: ProgressFn | None = None,
) -> list[ReviewedCard]:
    """`suspect_claims`: statements in the notes that look wrong (from the fact check); no card is made from them as written."""
    progress = on_stage or _noop
    progress("generate")
    cards = generate_cards(client, source, evidence, suspect_claims)
    progress("verify")
    return verify_cards(client, source, evidence, cards)


def run_pipeline(
    client: LLM,
    source: SourceDoc,
    *,
    approve_evidence: ApproveFn | None = None,
    on_stage: ProgressFn | None = None,
) -> PipelineResult:
    research = run_research(client, source, on_stage=on_stage)
    # In the app this is the "approve sources" screen; here the caller decides.
    evidence = approve_evidence(research.evidence) if approve_evidence else research.evidence
    suspect = getattr(research.understanding, "suspect_claims", ())
    reviewed = run_generation(client, source, evidence, suspect_claims=suspect, on_stage=on_stage)
    return PipelineResult(
        understanding=research.understanding,
        research_text=research.research_text,
        evidence=evidence,
        cards=reviewed,
        usage=client.usage,
    )
