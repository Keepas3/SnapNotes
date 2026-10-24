"""Upload notes (one or more files form a single document) and start a job."""
from __future__ import annotations

import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_cfg, get_library, get_store, get_worker
from app.api.models import UploadOut
from app.core.config import Settings
from app.ingest.loader import ALLOWED_SUFFIXES, IMAGE_TYPES, UnsupportedFileError, validate_file
from app.pipeline.worker import JobWorker
from app.store.jobs import JobStore
from app.store.library import Library, new_id

router = APIRouter(tags=["documents"])

CHUNK = 1024 * 1024


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:60] or "notes"


def _kind(suffixes: set[str]) -> str:
    if suffixes <= {".pdf"}:
        return "pdf"
    if suffixes <= set(IMAGE_TYPES):
        return "images"
    if len(suffixes) == 1:
        return suffixes.copy().pop().lstrip(".")
    return "mixed"


@router.post("/documents", response_model=UploadOut, status_code=202)
async def upload_document(
    files: list[UploadFile],
    auto_approve: bool = Form(False),
    deck_id: str | None = Form(None, description="Add the new cards to this existing deck instead of making a new deck"),
    cfg: Settings = Depends(get_cfg),
    store: JobStore = Depends(get_store),
    library: Library = Depends(get_library),
    worker: JobWorker = Depends(get_worker),
) -> UploadOut:
    if not files:
        raise HTTPException(400, "Attach at least one file")
    if len(files) > cfg.max_files_per_upload:
        raise HTTPException(400, f"At most {cfg.max_files_per_upload} files per upload")
    deck_id = (deck_id or "").strip() or None  # an empty form field means "no deck chosen"
    if deck_id is not None and await run_in_threadpool(library.get_deck, deck_id) is None:
        raise HTTPException(404, "Deck not found")

    # Spend guard: bound how many jobs can be started per hour.
    since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    if await run_in_threadpool(store.count_since, since) >= cfg.max_jobs_per_hour:
        raise HTTPException(429, "Hourly job limit reached; try again later", headers={"Retry-After": "600"})

    doc_id = new_id()
    dest = cfg.upload_dir / doc_id
    dest.mkdir(parents=True, exist_ok=True)
    stored: list[str] = []
    total = 0
    try:
        for idx, f in enumerate(files):
            suffix = Path(f.filename or "").suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(415, f"Unsupported file type: {f.filename or '(unnamed)'}")
            # Never reuse the client's path: generated prefix + sanitised stem (also what the model sees).
            name = f"{idx:02d}_{_safe_stem(f.filename or '')}{suffix}"
            with (dest / name).open("wb") as out:
                while chunk := await f.read(CHUNK):
                    total += len(chunk)
                    if total > cfg.max_upload_bytes:
                        raise HTTPException(413, f"Upload exceeds {cfg.max_upload_bytes // (1024 * 1024)} MB")
                    out.write(chunk)
            stored.append(name)
            if (dest / name).stat().st_size == 0:
                raise HTTPException(422, f"{f.filename or name} is empty")
            try:
                await run_in_threadpool(validate_file, dest / name)
            except UnsupportedFileError as exc:
                raise HTTPException(415, str(exc)) from exc
            except Exception as exc:  # corrupt DOCX/PPTX etc.
                raise HTTPException(422, f"Could not read {f.filename or name}") from exc

        shown = files[0].filename or stored[0]
        filename = shown if len(files) == 1 else f"{shown} (+{len(files) - 1} more)"
        kind = _kind({Path(n).suffix.lower() for n in stored})
        job_id = await run_in_threadpool(
            lambda: store.create(
                doc_id=doc_id, filename=filename, kind=kind, stored_names=stored, auto_approve=auto_approve, target_deck_id=deck_id
            )
        )
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    worker.wake()
    return UploadOut(document_id=doc_id, job_id=job_id)


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, cfg: Settings = Depends(get_cfg), store: JobStore = Depends(get_store)) -> None:
    """Removes the original files, the job and its evidence, and any card that was never approved."""
    if not store.delete_document(document_id):
        raise HTTPException(404, "Document not found")
    shutil.rmtree(cfg.upload_dir / document_id, ignore_errors=True)
