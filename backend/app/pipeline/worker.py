"""In-process background worker: claims queued jobs from SQLite and runs the Claude pipeline in threads.

At personal scale this replaces Celery/Redis. Because every state change is persisted, a restart just
re-queues whatever was mid-flight (JobStore.recover_interrupted).
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Callable

import anthropic

from app.core.config import Settings
from app.ingest.loader import UnsupportedFileError, load_source
from app.llm.base import LLM
from app.llm.client import MissingCredentials, RefusalError, TruncatedError, UsageLog
from app.pipeline.orchestrator import run_generation, run_research
from app.store.jobs import JobStore

log = logging.getLogger(__name__)


class CostCeilingExceeded(RuntimeError):
    pass


def check_cost_ceiling(spent_before: float, usage: UsageLog, ceiling: float) -> None:
    """Stop a job between stages when its estimated spend passes the limit, or cannot be estimated at all.

    A model with no known price makes the estimate impossible; carrying on would mean spending with no limit,
    so that fails closed too (the server already refuses to start with such a model configured)."""
    this_run = usage.total_cost
    if this_run is None:
        raise CostCeilingExceeded(f"Stopped: no price is known for {', '.join(usage.unpriced())}, so spending cannot be limited.")
    if spent_before + this_run > ceiling:
        raise CostCeilingExceeded(f"Stopped: this document exceeded the ${ceiling:.2f} cost limit.")


def _describe_gemini_failure(exc: BaseException) -> str | None:
    """Reason for a failure that came from Google's SDK, or None when it is not one. The SDK is only loaded if it already is
    (the exception being Google's proves that), so a Claude-only server never imports it."""
    if not type(exc).__module__.startswith("google."):
        return None
    from google.genai import errors

    if not isinstance(exc, errors.APIError):
        return None
    code = exc.code
    message = str(exc.message or "")
    if code in (401, 403) or "API key" in message:
        return "Gemini rejected the server's key. Check GEMINI_API_KEY."
    if code == 429:
        return "The free AI is rate limiting requests, or today's free allowance is used up. Wait a minute and retry, or try again tomorrow."
    if code == 400:
        return f"Gemini could not process this document: {message[:300]}"
    if code and code >= 500:
        return f"Gemini is having trouble ({code}). Retry later."
    return f"Gemini API error ({code}). Retry later."


def describe_failure(exc: BaseException) -> str:
    """User-facing reason for a failed job. Never leaks stack traces or credentials."""
    if isinstance(exc, CostCeilingExceeded | UnsupportedFileError | MissingCredentials):
        return str(exc)
    gemini = _describe_gemini_failure(exc)
    if gemini is not None:
        return gemini
    if isinstance(exc, RefusalError):
        return "The AI declined to process this content."
    if isinstance(exc, TruncatedError):
        return "The AI's response was too long to complete. Try a shorter document."
    if isinstance(exc, TypeError) and "authentication method" in str(exc):
        return "The server has no Claude credentials configured."
    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return "Claude rejected the server's credentials."
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude is rate limiting requests. Retry in a minute."
    if isinstance(exc, anthropic.BadRequestError):
        return f"Claude could not process this document: {str(exc.message)[:300]}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach Claude. Retry shortly."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Claude API error ({exc.status_code}). Retry later."
    return "Unexpected error while processing this document. Check the server logs."


class JobWorker:
    def __init__(self, cfg: Settings, store: JobStore, llm_factory: Callable[[], LLM], poll_seconds: float = 2.0):
        self.cfg = cfg
        self.store = store
        self.llm_factory = llm_factory
        self.poll_seconds = poll_seconds
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._running: set[asyncio.Task] = set()
        self._stopping = False

    # -- lifecycle -----------------------------------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._stopping = False
        recovered = await asyncio.to_thread(self.store.recover_interrupted)
        if recovered:
            log.info("Re-queued %d job(s) interrupted by the last shutdown", recovered)
        self._task = asyncio.create_task(self._run(), name="job-worker")

    async def stop(self, grace_seconds: float = 30.0) -> None:
        self._stopping = True
        self.wake()
        if self._task:
            await self._task
        if self._running:
            # Threads cannot be cancelled; anything still running is re-queued on next start.
            await asyncio.wait(self._running, timeout=grace_seconds)

    def wake(self) -> None:
        """Safe to call from request handlers running in the threadpool."""
        if self._loop is not None and self._wake is not None:
            try:
                self._loop.call_soon_threadsafe(self._wake.set)
            except RuntimeError:  # loop already closed during shutdown
                pass

    # -- loop ----------------------------------------------------------------------------

    async def _run(self) -> None:
        assert self._wake is not None
        while not self._stopping:
            try:
                self._wake.clear()
                while len(self._running) < self.cfg.job_concurrency:
                    job = await asyncio.to_thread(self.store.claim_next)
                    if job is None:
                        break
                    task = asyncio.create_task(self._run_job(job))
                    self._running.add(task)
                    task.add_done_callback(self._on_done)
            except Exception:
                log.exception("Job worker iteration failed")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass

    def _on_done(self, task: asyncio.Task) -> None:
        self._running.discard(task)
        self.wake()  # a slot is free; look for more work

    async def _run_job(self, job: dict) -> None:
        await asyncio.to_thread(self._process, job)

    # -- one job (runs in a thread) ------------------------------------------------------

    def _process(self, job: dict) -> None:
        job_id = job["id"]
        client: LLM | None = None  # built inside the try below: a provider with no key fails here, and that must fail the job, not the worker
        prior_cost = job["cost_usd"]
        flushed = 0

        def flush_usage() -> None:
            """Persist spend before the job's final status becomes visible, so a client that sees
            "completed"/"failed" never reads a stale cost."""
            nonlocal flushed
            if client is None:
                return
            rows = client.usage.rows[flushed:]
            flushed = len(client.usage.rows)
            self.store.add_usage(job_id, rows)

        def on_stage(stage: str) -> None:
            # Spend ceiling is enforced between stages; a single stage can still overshoot it.
            check_cost_ceiling(prior_cost, client.usage, self.cfg.max_job_cost_usd)
            self.store.set_stage(job_id, stage)

        try:
            client = self.llm_factory()
            doc = self.store.document(job["document_id"])
            if doc is None:  # deleted while queued
                return
            source = load_source([self.cfg.upload_dir / doc["id"] / name for name in doc["stored_names"]])

            if job["phase"] == "research":
                outcome = run_research(client, source, on_stage=on_stage)
                flush_usage()
                status = self.store.save_research(job_id, outcome.understanding, outcome.evidence, outcome.note)
                log.info("Job %s research done: %d evidence item(s) -> %s", job_id, len(outcome.evidence), status)
            else:
                evidence = self.store.approved_evidence(job_id)
                suspect = self.store.suspect_claims(job_id)
                reviewed = run_generation(client, source, evidence, suspect_claims=suspect, on_stage=on_stage)
                flush_usage()
                self.store.save_generation(job_id, reviewed)
                log.info("Job %s completed with %d card(s)", job_id, len(reviewed))
                if self.cfg.delete_originals:
                    shutil.rmtree(self.cfg.upload_dir / doc["id"], ignore_errors=True)
        except Exception as exc:
            log.exception("Job %s failed in phase %s", job_id, job["phase"])
            flush_usage()
            self.store.fail(job_id, describe_failure(exc))
        finally:
            flush_usage()  # no-op unless the doc was deleted mid-run and we returned early
