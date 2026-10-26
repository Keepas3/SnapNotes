"""Delete uploaded originals nobody is going to use again."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from app.store.jobs import JobStore

log = logging.getLogger(__name__)


def sweep_old_uploads(upload_dir: Path, store: JobStore, ttl_days: int, now: float | None = None) -> int:
    """Remove upload folders older than `ttl_days` that no queued or running job needs; returns how many.

    Originals are normally removed when a job finishes. A job that failed, or one that waits for you to approve its sources, keeps them
    (so it can be retried or continued); without this they would stay for ever and could fill the disk. After the limit, such a job
    has to be started again from the notes. A folder that is still being written is new, so it is never touched."""
    if ttl_days <= 0 or not upload_dir.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - ttl_days * 86_400
    busy = store.active_document_ids()
    removed = 0
    for folder in upload_dir.iterdir():
        try:
            if folder.is_dir() and folder.name not in busy and folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        log.info("Removed %d old upload folder(s)", removed)
    return removed
