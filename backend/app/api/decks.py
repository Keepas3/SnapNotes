from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response

from app.api.deps import get_library
from app.api.models import CardIn, CardOut, CardStatus, DeckIn, DeckOut, DeckPatch, MergeIn, MergeOut
from app.export import content_disposition, safe_filename_stem
from app.export.anki import build_apkg
from app.export.csv_export import build_csv
from app.store.library import Library

router = APIRouter(prefix="/decks", tags=["decks"])


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
        card = library.create_card(deck_id, type=body.type, front=body.front, back=body.back)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if card is None:
        raise HTTPException(404, "Deck not found")
    return card


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
