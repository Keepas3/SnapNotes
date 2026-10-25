"""SnapNotes API.

    SNAPNOTES_TOKEN=<long random string> uvicorn app.main:app

For local development without a token: SNAPNOTES_ALLOW_NO_AUTH=1.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import cards, decks, folders, jobs, sync, uploads
from app.core.auth import require_auth, token_problem
from app.core.bodylimit import BodyLimitMiddleware
from app.core.config import Settings, fact_check_problem, provider_problem
from app.core.db import Database
from app.core.ratelimit import SlidingWindowLimiter
from app.llm.base import LLM
from app.llm.client import LLMClient, unpriced_models
from app.llm.demo import DemoSDK
from app.llm.factory import build_llm
from app.pipeline.worker import JobWorker
from app.store.jobs import JobStore
from app.store.library import Library
from app.store.sync import SyncStore

log = logging.getLogger(__name__)


def create_app(cfg: Settings | None = None, llm_factory: Callable[[], LLM] | None = None) -> FastAPI:
    cfg = cfg or Settings()
    if llm_factory is None:
        if cfg.demo_mode:
            log.warning("DEMO MODE: Claude is not called; responses are canned")
            demo_sdk = DemoSDK(cfg.demo_delay)
            llm_factory = lambda: LLMClient(demo_sdk, cfg)  # noqa: E731
        else:
            if cfg.provider != "anthropic":
                log.warning("AI provider: %s (web research is not available with it; cards come from the notes alone)", cfg.provider)
            llm_factory = lambda: build_llm(cfg)  # noqa: E731
    db = Database(cfg.db_path)
    store = JobStore(db)
    library = Library(db)
    worker = JobWorker(cfg, store, llm_factory)
    limiter = SlidingWindowLimiter(cfg.requests_per_minute)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        problem = token_problem(cfg) or provider_problem(cfg) or fact_check_problem(cfg)
        if problem:
            raise RuntimeError(problem)
        missing = unpriced_models(cfg)
        if missing:
            raise RuntimeError(
                f"No price is known for {', '.join(missing)}, so the per-document spend limit could not work. Add it with "
                f'SNAPNOTES_EXTRA_PRICES=\'{{"{missing[0]}": [input_usd_per_million_tokens, output_usd_per_million_tokens]}}\'.'
            )
        db.init()
        cfg.upload_dir.mkdir(parents=True, exist_ok=True)
        await worker.start()
        try:
            yield
        finally:
            await worker.stop()

    app = FastAPI(
        title="SnapNotes",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if cfg.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if cfg.docs_enabled else None,
    )
    app.state.cfg = cfg
    app.state.store = store
    app.state.library = library
    app.state.sync_store = SyncStore(db)
    app.state.worker = worker

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if request.url.path != "/health":
            # Behind Fly.io the proxy puts the real client address in a header it controls (configurable; empty = ignore).
            forwarded = request.headers.get(cfg.client_ip_header) if cfg.client_ip_header else None
            ip = (forwarded or (request.client.host if request.client else "unknown"))[:64]
            if not limiter.allow(ip):
                return JSONResponse({"detail": "Too many requests"}, status_code=429, headers={"Retry-After": "60"})
        return await call_next(request)

    # In front of everything that reads a body: an oversized request is refused before any of it is read.
    # Uploads get the configured file limit plus a little room for the multipart framing around the files.
    app.add_middleware(
        BodyLimitMiddleware,
        upload_path="/documents",
        upload_limit=cfg.max_upload_bytes + 1024 * 1024,
        default_limit=cfg.max_body_kb * 1024,
    )

    # Added last so it is the outermost layer: rejected (401/413/429) responses still carry CORS headers.
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["authorization", "content-type"],
            expose_headers=["content-disposition"],  # so the web preview can name downloaded exports
        )

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    api = APIRouter(dependencies=[Depends(require_auth)])
    for module in (uploads, jobs, decks, folders, cards, sync):
        api.include_router(module.router)
    app.include_router(api)
    return app


app = create_app()
