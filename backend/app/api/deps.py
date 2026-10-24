from fastapi import Request

from app.core.config import Settings
from app.pipeline.worker import JobWorker
from app.store.jobs import JobStore
from app.store.library import Library
from app.store.sync import SyncStore


def get_cfg(request: Request) -> Settings:
    return request.app.state.cfg


def get_store(request: Request) -> JobStore:
    return request.app.state.store


def get_library(request: Request) -> Library:
    return request.app.state.library


def get_worker(request: Request) -> JobWorker:
    return request.app.state.worker


def get_sync_store(request: Request) -> SyncStore:
    return request.app.state.sync_store
