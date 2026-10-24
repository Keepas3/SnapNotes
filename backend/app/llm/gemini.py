"""Gemini (Google's free tier) as an alternative to Claude: the same two calls.

Differences from `LLMClient` worth knowing:
- Web research uses Gemini's built-in Google Search, and the pipeline only asks for it when the fact check found a statement
  in the notes that looks wrong (see `enrich`). Otherwise there is no search and every card comes from the student's own notes.
  Google returns the model's own supported sentences with links, not the passage it read, and the links are Google redirect
  links. A search that fails never fails the job: it is reported in the server log and the job carries on without sources.
- Free-tier content may be used by Google to improve its products (its pricing page says so), so only non-private notes
  belong here. A search sends the topic and the doubtful statements to Google, not the notes themselves.
- One model serves every stage (`SNAPNOTES_GEMINI_MODEL`); the per-stage Claude model names and efforts do not apply.
"""
from __future__ import annotations

import base64
import itertools
import logging
import time
from types import SimpleNamespace
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.config import Settings, StageConfig, gemini_api_key, settings
from app.llm.client import MissingCredentials, RefusalError, ResearchResult, TruncatedError, UsageLog, price_table
from app.llm.dump import write_dump
from app.llm.prompts import SYSTEM_PROMPT
from app.llm.schemas import EvidenceItem

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

REQUEST_TIMEOUT_MS = 180_000  # the SDK counts milliseconds
RETRY_ATTEMPTS = 3  # the SDK retries rate limits (429) and server errors with a growing pause; attempts include the first try
MAX_FORMAT_RETRIES = 1  # one more try when the answer is not valid JSON for the schema
SLOW_CALL_SECONDS = 30  # a call slower than this is reported in the server log
# Finish reasons that mean the model declined or was stopped, as opposed to simply finishing.
STOPPED_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "LANGUAGE", "OTHER", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT"}


def _name(value: Any) -> str:
    """An SDK enum (or a plain string) as its name; empty when absent."""
    return getattr(value, "name", None) or str(value or "")


def to_parts(blocks: list[dict]) -> list[types.Part]:
    """Claude-format content blocks (what the loader builds) as Gemini parts. `cache_control` has no meaning here."""
    parts: list[types.Part] = []
    for block in blocks:
        kind = block.get("type")
        source = block.get("source") or {}
        if kind == "text":
            parts.append(types.Part.from_text(text=block["text"]))
        elif kind in ("image", "document") and source.get("type") == "base64":
            parts.append(types.Part.from_bytes(data=base64.b64decode(source["data"]), mime_type=source["media_type"]))
        else:
            raise ValueError(f"Unsupported content block for Gemini: {kind!r}")
    return parts


def describe_search_failure(exc: BaseException) -> str:
    """A short reason a student can read, without anything secret. Google's own message is included, trimmed."""
    code = getattr(exc, "code", None)
    message = " ".join(str(getattr(exc, "message", "") or exc).split())[:160]
    if code == 429:
        return f"Google's free search allowance is used up or the limit was hit: {message}"
    if code in (400, 403) and ("search" in message.lower() or "tool" in message.lower() or "ground" in message.lower()):
        return f"Google refused the search for this key or model: {message}"
    if code in (401, 403):
        return f"Google rejected the key: {message}"
    if code:
        return f"Google answered {code}: {message}"
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


MAX_EVIDENCE = 12  # sources kept from one search; more is noise on the approval screen
SOURCES_PER_STATEMENT = 2


def _grounding(resp: Any) -> Any:
    candidates = getattr(resp, "candidates", None) or []
    return getattr(candidates[0], "grounding_metadata", None) if candidates else None


def _search_count(resp: Any) -> int:
    """How many Google searches the answer says it ran."""
    return len(getattr(_grounding(resp), "web_search_queries", None) or [])


def evidence_from_grounding(resp: Any) -> list[EvidenceItem]:
    """Each sentence of the answer that Google backed with a source, once per source (at most two), as evidence.

    The claim and the supporting passage are both that sentence: Google does not return the passage it read."""
    meta = _grounding(resp)
    if meta is None:
        return []
    chunks = list(getattr(meta, "grounding_chunks", None) or [])
    items: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for support in getattr(meta, "grounding_supports", None) or []:
        text = (getattr(getattr(support, "segment", None), "text", "") or "").strip()
        if not text:
            continue
        taken = 0
        for index in getattr(support, "grounding_chunk_indices", None) or []:
            web = getattr(chunks[index], "web", None) if 0 <= index < len(chunks) else None
            uri = (getattr(web, "uri", "") or "").strip()
            if not uri.lower().startswith(("http://", "https://")) or (uri, text[:80]) in seen:
                continue
            seen.add((uri, text[:80]))
            title = (getattr(web, "title", "") or getattr(web, "domain", "") or uri).strip()
            items.append(EvidenceItem(claim=text, snippet=text, url=uri, title=title))
            taken += 1
            if taken >= SOURCES_PER_STATEMENT or len(items) >= MAX_EVIDENCE:
                break
        if len(items) >= MAX_EVIDENCE:
            break
    return items


def _usage(resp: Any) -> SimpleNamespace:
    """Gemini's token counts in the shape `UsageLog.add` reads. Thinking tokens are billed as output."""
    meta = getattr(resp, "usage_metadata", None)
    prompt = getattr(meta, "prompt_token_count", 0) or 0
    cached = getattr(meta, "cached_content_token_count", 0) or 0
    output = (getattr(meta, "candidates_token_count", 0) or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)
    return SimpleNamespace(
        input_tokens=max(prompt - cached, 0),
        output_tokens=output,
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=0,
        server_tool_use=SimpleNamespace(web_search_requests=_search_count(resp)),
    )


class GeminiClient:
    def __init__(self, sdk: Any | None = None, cfg: Settings = settings):
        self.cfg = cfg
        self.model = cfg.gemini_model
        self.usage = UsageLog(prices=price_table(cfg))
        if sdk is None:
            key = gemini_api_key()
            if not key:
                raise MissingCredentials("The server has no Gemini key configured (GEMINI_API_KEY).")
            sdk = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=REQUEST_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS, initial_delay=2.0, max_delay=30.0),
                ),
            )
        self.sdk = sdk
        self._dumps = itertools.count(1)

    def _dump(self, name: str, resp: Any) -> None:
        write_dump(self.cfg.debug_dump_dir, name, next(self._dumps), resp, log)

    def _call(self, stage_name: str, contents: list[types.Content], config: types.GenerateContentConfig) -> Any:
        """One request, with its usage recorded and a warning when it was slow."""
        started = time.monotonic()
        resp = self.sdk.models.generate_content(model=self.model, contents=contents, config=config)
        took = time.monotonic() - started
        self.usage.add(stage_name, self.model, _usage(resp))
        if took > SLOW_CALL_SECONDS:  # shown in the server window: a slow stage is otherwise indistinguishable from a stuck one
            row = self.usage.rows[-1]
            log.warning(
                "%s: %s took %.0f s for %d token(s) in and %d out. Google's free tier is slow or busy right now; "
                "a lighter model (SNAPNOTES_GEMINI_MODEL) may be faster.",
                stage_name, self.model, took, row.input_tokens, row.output_tokens,
            )
        return resp

    @staticmethod
    def _check(resp: Any) -> None:
        reason = _name(getattr(getattr(resp, "prompt_feedback", None), "block_reason", None))
        if reason and reason != "BLOCKED_REASON_UNSPECIFIED":
            raise RefusalError(f"Request refused (category: {reason})")
        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            raise RefusalError("Request refused (no answer was returned)")
        finish = _name(getattr(candidates[0], "finish_reason", None))
        if finish == "MAX_TOKENS":
            raise TruncatedError("Response hit the output limit; raise the stage limit or use a shorter document")
        if finish in STOPPED_REASONS:
            raise RefusalError(f"Request refused (category: {finish})")

    def parse(self, stage_name: str, stage: StageConfig, *, content: list[dict], output_model: type[T]) -> T:
        contents = [types.Content(role="user", parts=to_parts(content))]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=output_model,
            max_output_tokens=stage.max_tokens,
        )
        for attempt in range(MAX_FORMAT_RETRIES + 1):
            resp = self._call(stage_name, contents, config)
            self._check(resp)
            text = resp.text or ""
            try:
                return output_model.model_validate_json(text)
            except ValueError as exc:  # pydantic's ValidationError is a ValueError, and so is bad JSON
                problem = str(exc)[:600]
                log.warning("%s: Gemini's answer did not fit the schema (attempt %d): %s", stage_name, attempt + 1, problem)
                contents = contents + [
                    types.Content(role="model", parts=[types.Part.from_text(text=text)]),
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=f"Your reply did not match the required JSON format ({problem}). Reply again with only the corrected JSON.")],
                    ),
                ]
        raise RuntimeError(f"{stage_name}: model returned no parseable output")

    def research(self, stage_name: str, stage: StageConfig, *, content: list[dict]) -> ResearchResult:
        """Google Search through Gemini. Any failure is reported and means "no sources", never a failed job."""
        if self.cfg.web_allowed_domains:
            log.warning("SNAPNOTES_WEB_ALLOWED_DOMAINS is ignored with Gemini: its Google Search cannot be limited to chosen sites. Use SNAPNOTES_WEB_BLOCKED_DOMAINS to leave sites out.")
        blocked = list(self.cfg.web_blocked_domains)
        search = types.GoogleSearch(exclude_domains=blocked) if blocked else types.GoogleSearch()
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(google_search=search)],
            max_output_tokens=stage.max_tokens,
        )
        contents = [types.Content(role="user", parts=to_parts(content))]
        try:
            resp = self._call(stage_name, contents, config)
            self._dump(f"research-{stage_name}", resp)
            self._check(resp)
        except Exception as exc:
            reason = describe_search_failure(exc)
            log.warning(
                "%s: Google Search did not work (%s: %s); carrying on without web sources, so any corrections come from the AI's own knowledge.",
                stage_name, type(exc).__name__, str(exc)[:200],
            )
            return ResearchResult(text="", evidence=[], searches_used=0, note=f"The web search did not work ({reason}), so the corrections come from the AI's own knowledge.")

        evidence = evidence_from_grounding(resp)
        searches = _search_count(resp)
        note = ""
        if evidence:
            log.info("%s: Google Search ran %d quer(y/ies) and found %d source(s)", stage_name, searches, len(evidence))
        else:
            log.warning("%s: Google Search ran %d quer(y/ies) but returned no usable sources; corrections will come from the AI's own knowledge.", stage_name, searches)
            note = "The web search found no usable sources, so the corrections come from the AI's own knowledge."
        return ResearchResult(text=resp.text or "", evidence=evidence, searches_used=searches, note=note)
