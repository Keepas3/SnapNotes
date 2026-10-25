"""Folders that group decks (Biology > Bio I, Bio II). A deck is in at most one folder; a folder never owns its decks."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_library
from app.api.models import FolderIn, FolderOut
from app.store.library import Library

router = APIRouter(prefix="/folders", tags=["folders"])


@router.get("", response_model=list[FolderOut])
def list_folders(library: Library = Depends(get_library)):
    return library.list_folders()


@router.post("", response_model=FolderOut, status_code=201)
def create_folder(body: FolderIn, library: Library = Depends(get_library)):
    try:
        return library.create_folder(body.name)
    except ValueError as exc:  # empty, too long, or the name is taken
        raise HTTPException(409 if "already" in str(exc) else 422, str(exc)) from exc


@router.patch("/{folder_id}", response_model=FolderOut)
def rename_folder(folder_id: str, body: FolderIn, library: Library = Depends(get_library)):
    try:
        folder = library.rename_folder(folder_id, body.name)
    except ValueError as exc:
        raise HTTPException(409 if "already" in str(exc) else 422, str(exc)) from exc
    if folder is None:
        raise HTTPException(404, "Folder not found")
    return folder


@router.delete("/{folder_id}", status_code=204)
def delete_folder(folder_id: str, library: Library = Depends(get_library)) -> None:
    """Removes the folder only. Its decks are kept and become unfiled."""
    if not library.delete_folder(folder_id):
        raise HTTPException(404, "Folder not found")
