from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response

import logging

from starlette.concurrency import run_in_threadpool

from app.api.deps import get_cfg, get_choice_limiter, get_library, get_llm_factory
from app.api.models import (
    CardIn,
    CardOut,
    CardStatus,
    ChoiceSkipped,
    ChoiceSuggestion,
    DeckIn,
    DeckOut,
    DeckPatch,
    MergeIn,
    MergeOut,
    SuggestChoicesIn,
    SuggestChoicesOut,
)
from app.core.config import Settings
from app.llm.choices import suggest_choices
from app.pipeline.worker import CostCeilingExceeded, check_cost_ceiling, describe_failure
from app.export import content_disposition, safe_filename_stem
from app.export.anki import build_apkg
from app.export.csv_export import build_csv
from app.store.library import FolderNotFound, Library

router = APIRouter(prefix="/decks", tags=["decks"])
log = logging.getLogger(__name__)

SUGGEST_AT_MOST = 25  # cards per request: one AI call, so one bounded cost


@router.get("", response_model=list[DeckOut])
def list_decks(library: Library = Depends(get_library)):
    return library.list_decks()


@router.post("", response_model=DeckOut, status_code=201)
def create_deck(body: DeckIn, library: Library = Depends(get_library)):
    return library.create_deck(body.name, body.topic)


@router.get("/{deck_id}", response_model=DeckOut)
def get_deck(deck_id: str, library: Library = Depends(get_library)):
    deck = library.get_deck(deck_id)
    if deck is None:
        raise HTTPException(404, "Deck not found")
    return deck


@router.patch("/{deck_id}", response_model=DeckOut)
def update_deck(deck_id: str, body: DeckPatch, library: Library = Depends(get_library)):
    deck = library.update_deck(deck_id, name=body.name, topic=body.topic)
    if deck is None:
        raise HTTPException(404, "Deck not found")
    if "folder_id" in body.model_fields_set:  # sent, possibly as null: file it, or take it out
        try:
            deck = library.set_deck_folder(deck_id, body.folder_id)
        except FolderNotFound as exc:
            raise HTTPException(404, "Folder not found") from exc
        if deck is None:
            raise HTTPException(404, "Deck not found")
    return deck


@router.delete("/{deck_id}", status_code=204)
def delete_deck(deck_id: str, library: Library = Depends(get_library)) -> None:
    if not library.delete_deck(deck_id):
        raise HTTPException(404, "Deck not found")


@router.post("/{deck_id}/merge", response_model=MergeOut)
def merge_deck(deck_id: str, body: MergeIn, library: Library = Depends(get_library)):
    """Move every card of another deck into this one, then remove the other deck.

    Cards keep their ids, so study history and schedules carry over; drafts and set-aside cards move too, with their status."""
    if library.get_deck(deck_id) is None:
        raise HTTPException(404, "Deck not found")
    if library.get_deck(body.from_deck_id) is None:
        raise HTTPException(404, "The deck to merge in was not found")
    try:
        result = library.merge_decks(deck_id, body.from_deck_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if result is None:  # deleted between the checks above and the merge
        raise HTTPException(404, "Deck not found")
    deck, moved = result
    return {"deck": deck, "moved": moved}


@router.post("/{deck_id}/cards", response_model=CardOut, status_code=201)
def create_card(deck_id: str, body: CardIn, library: Library = Depends(get_library)):
    """Add a card you typed yourself. It is approved at once and reaches the phone on the next sync."""
    try:
        card = library.create_card(deck_id, type=body.type, front=body.front, back=body.back, choices=body.choices)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if card is None:
        raise HTTPException(404, "Deck not found")
    return card


@router.post("/{deck_id}/suggest-choices", response_model=SuggestChoicesOut)
async def suggest_deck_choices(
    deck_id: str,
    body: SuggestChoicesIn,
    library: Library = Depends(get_library),
    cfg: Settings = Depends(get_cfg),
    llm_factory=Depends(get_llm_factory),
    limiter=Depends(get_choice_limiter),
):
    """Ask the AI for wrong answers for multiple choice. **Nothing is saved**: these are suggestions for the user to check, edit or drop,
    then save with `POST /cards/choices`. Only approved basic and definition cards qualify (a fill-in-the-blank card has no natural options).

    With no `card_ids`, the deck's qualifying cards that have fewer wrong answers than `wrong_per_card` are used, up to 25."""
    if library.get_deck(deck_id) is None:
        raise HTTPException(404, "Deck not found")
    qualifying = {c["id"]: c for c in library.list_cards(deck_id=deck_id, status="approved") if c["type"] in ("basic", "definition") and c["back"].strip()}
    skipped: list[ChoiceSkipped] = []
    if body.card_ids is not None:
        chosen = [qualifying[i] for i in dict.fromkeys(body.card_ids) if i in qualifying]
        skipped = [ChoiceSkipped(card_id=i, reason="Not an approved question or definition card in this deck.") for i in dict.fromkeys(body.card_ids) if i not in qualifying]
    else:
        chosen = [c for c in qualifying.values() if len(c["choices"]) < body.wrong_per_card][:SUGGEST_AT_MOST]
    if not chosen:
        return SuggestChoicesOut(suggestions=[], skipped=skipped)
    if not limiter.allow("choices"):
        raise HTTPException(429, "You have asked for a lot of wrong answers in the last hour. Try again a little later.", headers={"Retry-After": "600"})

    def work():
        client = llm_factory()
        found, not_made = suggest_choices(client, chosen, body.wrong_per_card)
        check_cost_ceiling(0.0, client.usage, cfg.max_job_cost_usd)  # also refuses a model whose price is unknown
        return found, not_made

    try:
        found, not_made = await run_in_threadpool(work)
    except CostCeilingExceeded as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:  # the provider's own failures, in words the user can act on
        log.exception("Suggesting wrong answers failed")
        raise HTTPException(502, describe_failure(exc)) from exc

    by_id = {c["id"]: c for c in chosen}
    return SuggestChoicesOut(
        suggestions=[ChoiceSuggestion(card_id=i, front=by_id[i]["front"], answer=by_id[i]["back"], wrong_answers=w) for i, w in found.items()],
        skipped=skipped + [ChoiceSkipped(card_id=i, reason=r) for i, r in not_made.items()],
    )


@router.get("/{deck_id}/export")
def export_deck(deck_id: str, format: Literal["apkg", "csv"] = "apkg", library: Library = Depends(get_library)) -> Response:
    """Download the deck's approved cards as an Anki package (`apkg`) or a CSV file.

    Anki imports the package with cloze cards, tags and source links intact. Cards are exported as new cards: study
    history stays in SnapNotes. Importing the same deck again updates the existing notes rather than duplicating them."""
    deck = library.get_deck(deck_id)
    if deck is None:
        raise HTTPException(404, "Deck not found")
    cards = library.list_cards(deck_id=deck_id, status="approved")
    if not cards:
        raise HTTPException(409, "This deck has no approved cards to export yet")

    stem = safe_filename_stem(deck["name"])
    if format == "apkg":
        body, media_type = build_apkg(deck, cards), "application/octet-stream"
    else:
        body, media_type = build_csv(deck, cards), "text/csv; charset=utf-8"
    return Response(content=body, media_type=media_type, headers={"Content-Disposition": content_disposition(stem, format)})


@router.get("/{deck_id}/cards", response_model=list[CardOut])
def deck_cards(deck_id: str, status: CardStatus | None = None, library: Library = Depends(get_library)):
    if library.get_deck(deck_id) is None:
        raise HTTPException(404, "Deck not found")
    return library.list_cards(deck_id=deck_id, status=status)
