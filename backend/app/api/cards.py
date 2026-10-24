from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_library
from app.api.models import CardOut, CardPatch, ReviewIn, ReviewOut
from app.store.library import Library

router = APIRouter(prefix="/cards", tags=["cards"])


# Declared before "/{card_id}" so "review" is never read as a card id.
@router.post("/review", response_model=ReviewOut)
def review_cards(body: ReviewIn, library: Library = Depends(get_library)):
    """Bulk approve/reject after swiping through a draft deck."""
    return {"changed": library.review(body.approve_ids, body.reject_ids)}


@router.get("/{card_id}", response_model=CardOut)
def get_card(card_id: str, library: Library = Depends(get_library)):
    card = library.get_card(card_id)
    if card is None:
        raise HTTPException(404, "Card not found")
    return card


@router.patch("/{card_id}", response_model=CardOut)
def update_card(card_id: str, body: CardPatch, library: Library = Depends(get_library)):
    card = library.update_card(card_id, body.model_dump(exclude_unset=True))
    if card is None:
        raise HTTPException(404, "Card not found")
    return card


@router.delete("/{card_id}", status_code=204)
def delete_card(card_id: str, library: Library = Depends(get_library)) -> None:
    if not library.delete_card(card_id):
        raise HTTPException(404, "Card not found")
