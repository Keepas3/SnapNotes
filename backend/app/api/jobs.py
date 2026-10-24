from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_library, get_store, get_worker
from app.api.models import ApproveIn, ApproveOut, CardOut, CardStatus, EvidenceResponse, JobOut
from app.pipeline.worker import JobWorker
from app.store.jobs import InvalidState, JobStore, NotFound
from app.store.library import Library

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _require_job(store: JobStore, job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.get("", response_model=list[JobOut])
def list_jobs(limit: int = Query(50, ge=1, le=200), store: JobStore = Depends(get_store)):
    return store.list(limit)


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, store: JobStore = Depends(get_store)):
    return _require_job(store, job_id)


@router.get("/{job_id}/evidence", response_model=EvidenceResponse)
def get_evidence(job_id: str, store: JobStore = Depends(get_store)):
    job = _require_job(store, job_id)
    return {"understanding": job["understanding"], "evidence": store.evidence(job_id)}


@router.post("/{job_id}/approve", response_model=ApproveOut)
def approve_sources(job_id: str, body: ApproveIn, store: JobStore = Depends(get_store), worker: JobWorker = Depends(get_worker)):
    """Approve every web source except `rejected_ids`, then queue card generation."""
    try:
        result = store.apply_approval(job_id, body.rejected_ids)
    except NotFound:
        raise HTTPException(404, "Job not found")
    except InvalidState as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    worker.wake()
    return result


@router.post("/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: str, store: JobStore = Depends(get_store), worker: JobWorker = Depends(get_worker)):
    try:
        store.retry(job_id)
    except NotFound:
        raise HTTPException(404, "Job not found")
    except InvalidState as exc:
        raise HTTPException(409, str(exc))
    worker.wake()
    return store.get(job_id)


@router.get("/{job_id}/cards", response_model=list[CardOut])
def job_cards(
    job_id: str,
    status: CardStatus | None = None,
    store: JobStore = Depends(get_store),
    library: Library = Depends(get_library),
):
    _require_job(store, job_id)
    return library.list_cards(job_id=job_id, status=status)
