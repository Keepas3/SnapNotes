"""Chooses the AI behind the pipeline from the settings (SNAPNOTES_PROVIDER)."""
from __future__ import annotations

from app.core.config import Settings, provider_problem
from app.llm.base import LLM
from app.llm.client import LLMClient


def build_llm(cfg: Settings) -> LLM:
    problem = provider_problem(cfg)
    if problem:
        raise ValueError(problem)
    if cfg.provider == "gemini":
        from app.llm.gemini import GeminiClient  # imported here so a Claude-only run never loads Google's SDK

        return GeminiClient(cfg=cfg)
    return LLMClient(cfg=cfg)
