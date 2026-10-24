"""Pydantic models for every structured Claude output and for the data that flows between stages.

Models used as `output_format` are kept deliberately plain (no numeric/length constraints, no defaults)
because the API's structured-output JSON Schema support is a subset of full JSON Schema.
"""
from typing import Literal

from pydantic import BaseModel

CardType = Literal["basic", "cloze", "definition"]
Origin = Literal["notes", "web", "ai"]  # "ai": a correction made from the model's own knowledge, with no source
Difficulty = Literal["easy", "medium", "hard"]


class UnderstandResult(BaseModel):
    main_topic: str
    subject_area: str
    subtopics: list[str]
    key_terms: list[str]
    gaps: list[str]  # things in the notes that look incomplete, ambiguous, or worth fact-checking
    search_queries: list[str]


class Card(BaseModel):
    type: CardType
    front: str  # question; for cloze cards the sentence containing {{c1::answer}} markers
    back: str  # answer; for cloze cards a short explanation
    origin: Origin
    source_ref: str  # notes: page/slide/heading. web: the URL
    source_snippet: str  # short supporting phrase (<= 200 chars)
    tags: list[str]
    difficulty: Difficulty


class CardSet(BaseModel):
    cards: list[Card]


class CardVerdict(BaseModel):
    index: int
    atomic: bool
    grounded: bool
    issue: str  # empty string when there is no problem
    duplicate_of: int  # index of an EARLIER card that tests the same fact, or -1


class VerifyResult(BaseModel):
    verdicts: list[CardVerdict]


# --- Fact-check variants -----------------------------------------------------------------
# Used only when the fact check is on (see fact_check_enabled). They are separate classes, not extra fields on the ones
# above, so that when the check is off the requests sent to the model are exactly what they always were.


class UnderstandResultChecked(UnderstandResult):
    # Statements in the notes that a textbook would contradict. Each: the claim as written, " — ", the correction.
    suspect_claims: list[str]


class CardVerdictChecked(CardVerdict):
    factual_concern: str  # empty when the card agrees with well-established knowledge, else one short sentence on why it may be wrong


class VerifyResultChecked(BaseModel):
    verdicts: list[CardVerdictChecked]


# --- Not produced directly by the model -------------------------------------------------


class EvidenceItem(BaseModel):
    """One web-sourced statement plus the passage that supports it (built from API citations)."""

    claim: str
    snippet: str
    url: str
    title: str
    approved: bool = True


class ReviewedCard(BaseModel):
    card: Card
    atomic: bool
    grounded: bool
    issue: str = ""
    duplicate_of: int | None = None

    @property
    def keep(self) -> bool:
        """Cards that are ungrounded or duplicates never reach the review queue by default."""
        return self.grounded and self.duplicate_of is None
