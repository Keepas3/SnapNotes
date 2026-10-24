"""Request/response bodies for the HTTP API."""
from typing import Literal

from pydantic import BaseModel, Field

from app.llm.schemas import UnderstandResult

CardStatus = Literal["draft", "approved", "rejected"]


class UnderstandingOut(UnderstandResult):
    """What the app sees of the first step. `suspect_claims` is empty when the fact check was off, and for jobs from before it existed."""

    suspect_claims: list[str] = Field(default_factory=list)
    research_note: str = ""  # why a web search that was wanted did not work, when it did not


class UploadOut(BaseModel):
    document_id: str
    job_id: str


class JobOut(BaseModel):
    id: str
    document_id: str
    filename: str
    kind: str
    status: Literal["queued", "researching", "awaiting_approval", "approved", "generating", "completed", "failed"]
    stage: str | None
    error: str | None
    auto_approve: bool
    cost_usd: float
    deck_id: str | None
    target_deck_id: str | None = None  # the deck this upload was told to add its cards to, when it was not meant to make a new one
    understanding: UnderstandingOut | None
    created_at: str
    updated_at: str


class EvidenceOut(BaseModel):
    id: int
    claim: str
    snippet: str
    url: str
    title: str
    approved: bool


class EvidenceResponse(BaseModel):
    understanding: UnderstandingOut | None
    evidence: list[EvidenceOut]


class ApproveIn(BaseModel):
    rejected_ids: list[int] = Field(default_factory=list, description="Evidence ids to drop; everything else is approved")


class ApproveOut(BaseModel):
    approved: int
    rejected: int


class CardOut(BaseModel):
    id: str
    deck_id: str
    job_id: str | None
    type: str
    front: str
    back: str
    origin: str
    source_ref: str
    source_snippet: str
    tags: list[str]
    difficulty: str
    status: CardStatus
    atomic: bool
    grounded: bool
    issue: str
    duplicate_of: str | None
    created_at: str
    updated_at: str


class CardPatch(BaseModel):
    front: str | None = Field(default=None, min_length=1)
    back: str | None = None  # may be empty for a fill-in-the-blank card (its explanation is optional); the store insists on text for any other card
    tags: list[str] | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    status: CardStatus | None = None


class ReviewIn(BaseModel):
    approve_ids: list[str] = Field(default_factory=list)
    reject_ids: list[str] = Field(default_factory=list)


class ReviewOut(BaseModel):
    changed: int


class DeckCounts(BaseModel):
    draft: int
    approved: int
    rejected: int


class DeckOut(BaseModel):
    id: str
    name: str
    topic: str
    counts: DeckCounts
    created_at: str
    updated_at: str


class DeckIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    topic: str = Field(default="", max_length=120)


class DeckPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    topic: str | None = Field(default=None, max_length=120)


class MergeIn(BaseModel):
    from_deck_id: str = Field(min_length=1, description="The deck whose cards move into the deck named in the URL; it is removed afterwards")


class MergeOut(BaseModel):
    deck: DeckOut
    moved: int


class CardIn(BaseModel):
    """A card typed in by hand. Which sides are required depends on the type, and the server says so in plain words."""

    type: Literal["basic", "cloze"] = "basic"
    front: str = Field(max_length=2000)
    back: str = Field(default="", max_length=2000)
