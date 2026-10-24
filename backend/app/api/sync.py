"""One round trip that both uploads a device's reviews and downloads what changed on the server."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_sync_store
from app.store.library import now
from app.store.sync import ReviewLogData, SyncStore

router = APIRouter(prefix="/sync", tags=["sync"])

MAX_LOGS_PER_REQUEST = 500


class FsrsState(BaseModel):
    """A card's FSRS state as computed by the device (ts-fsrs shape). Unknown extra fields are kept, so a newer app
    can add fields without the server dropping them."""

    model_config = ConfigDict(extra="allow")

    due: str
    stability: float
    difficulty: float
    reps: int
    lapses: int
    state: int = Field(ge=0, le=3)
    last_review: str | None = None


class ReviewLogIn(BaseModel):
    id: str = Field(min_length=8, max_length=64, description="Client-generated, so retrying a push never duplicates a review")
    card_id: str = Field(min_length=1, max_length=64)
    rating: int = Field(description="1 Again, 2 Hard, 3 Good, 4 Easy; other values are rejected per log")
    reviewed_at: str
    elapsed_ms: int = 0
    state_before: int = 0
    state_after: FsrsState

    # Cosmetic fields are clamped instead of validated, so one odd value can never block a whole batch forever.
    @field_validator("elapsed_ms", mode="after")
    @classmethod
    def _clamp_elapsed(cls, v: int) -> int:
        return max(0, min(v, 24 * 3600 * 1000))

    @field_validator("state_before", mode="after")
    @classmethod
    def _clamp_state_before(cls, v: int) -> int:
        return max(0, min(v, 3))


class SyncRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    cursor: int = Field(default=0, ge=0, description="The `cursor` from the previous response; 0 for a first sync")
    epoch: str | None = Field(default=None, max_length=64, description="The `epoch` from the previous response; omit on a first sync")
    logs: list[ReviewLogIn] = Field(default_factory=list, max_length=MAX_LOGS_PER_REQUEST)
    limit: int = Field(default=500, ge=1, le=1000, description="Max rows per kind in the response")


class RejectedLog(BaseModel):
    id: str
    reason: Literal["bad_rating", "bad_time", "bad_state", "unknown_card"]


class SyncDeck(BaseModel):
    id: str
    name: str
    topic: str
    updated_at: str


class SyncCard(BaseModel):
    id: str
    deck_id: str
    type: str
    front: str
    back: str
    origin: str
    source_ref: str
    source_snippet: str
    status: Literal["draft", "approved", "rejected"]
    created_at: str
    updated_at: str


class SyncCardState(BaseModel):
    card_id: str
    last_review: str
    state: dict[str, Any]


class Deleted(BaseModel):
    kind: Literal["deck", "card"]
    id: str


class SyncResponse(BaseModel):
    epoch: str = Field(description="Identity of this database; store it and send it back with every request")
    reset: bool = Field(
        description="True when the device's cursor or epoch did not belong to this database. The response is then a full "
        "download from zero: the device must discard its cached library and re-upload its whole review history."
    )
    cursor: int
    has_more: bool
    server_time: str
    accepted: int
    duplicates: int
    rejected: list[RejectedLog]
    decks: list[SyncDeck]
    cards: list[SyncCard]
    states: list[SyncCardState]
    deleted: list[Deleted]


@router.post("", response_model=SyncResponse)
async def sync(body: SyncRequest, store: SyncStore = Depends(get_sync_store)) -> SyncResponse:
    """Upload review logs (idempotent), then return changes after `cursor`.

    Keep calling with the returned cursor while `has_more` is true. Logs that cannot be stored (unknown card, bad
    rating...) are listed in `rejected` and should not be resent; the rest of the batch is still applied."""
    logs = [
        ReviewLogData(
            id=log.id, card_id=log.card_id, rating=log.rating, reviewed_at=log.reviewed_at,
            elapsed_ms=log.elapsed_ms, state_before=log.state_before, state_after=log.state_after.model_dump(),
        )
        for log in body.logs
    ]

    def work():
        epoch = store.epoch()
        reset = store.needs_reset(body.epoch, body.cursor)  # decided before this request's own reviews move the counter
        pushed = store.push_logs(body.device_id, logs)  # first, so the pull below already includes these reviews
        return epoch, reset, pushed, store.pull(0 if reset else body.cursor, body.limit)

    epoch, reset, pushed, pulled = await run_in_threadpool(work)
    return SyncResponse(
        epoch=epoch,
        reset=reset,
        cursor=pulled.cursor,
        has_more=pulled.has_more,
        server_time=now(),
        accepted=pushed.accepted,
        duplicates=pushed.duplicates,
        rejected=[RejectedLog(id=i, reason=r) for i, r in pushed.rejected],  # type: ignore[arg-type]
        decks=pulled.decks,  # type: ignore[arg-type]
        cards=pulled.cards,  # type: ignore[arg-type]
        states=pulled.states,  # type: ignore[arg-type]
        deleted=pulled.deleted,  # type: ignore[arg-type]
    )
