from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response

from app.api.deps import get_library
from app.api.models import CardOut, CardStatus, DeckIn, DeckOut, DeckPatch
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
