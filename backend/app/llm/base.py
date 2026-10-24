"""What the pipeline needs from an AI provider: two calls and a usage log.

Claude (`LLMClient`) and Gemini (`GeminiClient`) both provide this, so the stages, the orchestrator and the job worker do
not care which one is behind them.
"""
from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.core.config import Settings, StageConfig
from app.llm.client import ResearchResult, UsageLog

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    cfg: Settings
    usage: UsageLog

    def parse(self, stage_name: str, stage: StageConfig, *, content: list[dict], output_model: type[T]) -> T:
        """Read the notes in `content` (Claude-format blocks) and return the answer as `output_model`."""
        ...

    def research(self, stage_name: str, stage: StageConfig, *, content: list[dict]) -> ResearchResult:
        """Look things up online and return the evidence found (a provider without web access returns none)."""
        ...
