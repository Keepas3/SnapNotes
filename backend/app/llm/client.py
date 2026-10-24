"""Single entry point for every Claude call: model routing, refusal fallbacks, usage/cost logging."""
from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from app.core.config import Settings, StageConfig, settings
from app.llm.dump import write_dump
from app.llm.prompts import SYSTEM_PROMPT
from app.llm.schemas import EvidenceItem

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_SEARCH_TOOL = "web_search_20260209"
MAX_CONTINUATIONS = 5  # cap on pause_turn restarts for the web search loop

# $ per 1M tokens (input, output). Cache reads ~0.1x input, cache writes ~1.25x input.
# Web search request fees are not included.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class RefusalError(RuntimeError):
    pass


class TruncatedError(RuntimeError):
    pass


class MissingCredentials(RuntimeError):
    """The chosen provider has no key configured. The message is written for the person who runs the server."""


def price_table(cfg: Settings) -> dict[str, tuple[float, float]]:
    """Built-in prices plus any the operator added with SNAPNOTES_EXTRA_PRICES (which win on a clash).

    With the gemini provider the configured model is priced at zero: it is the free tier. Anyone on a paid Gemini plan
    can put the real price in SNAPNOTES_EXTRA_PRICES."""
    table = dict(PRICING)
    if cfg.provider == "gemini":
        table[cfg.gemini_model] = (0.0, 0.0)
    return {**table, **cfg.extra_prices}


def models_in_use(cfg: Settings) -> set[str]:
    if cfg.provider == "gemini":
        return {cfg.gemini_model}
    return {cfg.understand.model, cfg.enrich.model, cfg.generate.model, cfg.verify.model}


def unpriced_models(cfg: Settings) -> list[str]:
    """Configured models with no known price. The spend limit cannot work for them, so the server refuses to start."""
    return sorted(models_in_use(cfg) - price_table(cfg).keys())


@dataclass
class UsageRow:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    web_searches: int
    cost: float | None = None  # estimated $; None means this model has no known price


@dataclass
class UsageLog:
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(PRICING))
    rows: list[UsageRow] = field(default_factory=list)

    def add(self, stage: str, model: str, usage: Any) -> None:
        server = getattr(usage, "server_tool_use", None)
        row = UsageRow(
            stage=stage,
            model=model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            web_searches=getattr(server, "web_search_requests", 0) or 0,
        )
        price = self.prices.get(model)
        if price is not None:
            inp, out = price
            row.cost = (row.input_tokens * inp + row.cache_read * inp * 0.1 + row.cache_write * inp * 1.25 + row.output_tokens * out) / 1_000_000
        self.rows.append(row)

    @property
    def total_cost(self) -> float | None:
        """Sum of the estimated costs, or None if any call used a model with no known price."""
        costs = [r.cost for r in self.rows]
        return None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]

    def unpriced(self) -> list[str]:
        return sorted({r.model for r in self.rows if r.cost is None})


@dataclass
class ResearchResult:
    text: str
    evidence: list[EvidenceItem]
    searches_used: int
    note: str = ""  # why there is nothing to show when a search that was wanted did not work (shown to the user)


class LLMClient:
    def __init__(self, sdk: Any | None = None, cfg: Settings = settings):
        self.sdk = sdk or anthropic.Anthropic()
        self.cfg = cfg
        self.usage = UsageLog(prices=price_table(cfg))
        self._dumps = itertools.count(1)

    def _dump(self, name: str, resp: Any) -> None:
        """Save a raw response when SNAPNOTES_DEBUG_DUMP names a folder (off by default). It must never break a job."""
        write_dump(self.cfg.debug_dump_dir, name, next(self._dumps), resp, log)

    # -- routing -------------------------------------------------------------------------

    def _route(self, model: str) -> tuple[Any, dict[str, Any]]:
        """Beta namespace + fallback params when enabled. Haiku 5.5 has no server-side fallback."""
        if self.cfg.use_fallbacks and not model.startswith("claude-haiku-5"):
            return self.sdk.beta.messages, {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return self.sdk.messages, {}

    @staticmethod
    def _check_stop(resp: Any) -> None:
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None)
            raise RefusalError(f"Request refused (category: {category})")
        if resp.stop_reason == "max_tokens":
            raise TruncatedError("Response hit max_tokens; raise the stage limit or stream the call")

    # -- structured output ---------------------------------------------------------------

    def parse(self, stage_name: str, stage: StageConfig, *, content: list[dict], output_model: type[T]) -> T:
        ns, extra = self._route(stage.model)
        resp = ns.parse(
            model=stage.model,
            max_tokens=stage.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_format=output_model,
            output_config={"effort": stage.effort},
            **extra,
        )
        self.usage.add(stage_name, stage.model, resp.usage)
        self._check_stop(resp)
        if resp.parsed_output is None:
            raise RuntimeError(f"{stage_name}: model returned no parseable output")
        return resp.parsed_output

    # -- web research --------------------------------------------------------------------

    def _search_tool(self) -> dict[str, Any]:
        tool: dict[str, Any] = {"type": WEB_SEARCH_TOOL, "name": "web_search", "max_uses": self.cfg.web_search_max_uses}
        # Only one of the two domain lists may be sent.
        if self.cfg.web_allowed_domains:
            tool["allowed_domains"] = list(self.cfg.web_allowed_domains)
        elif self.cfg.web_blocked_domains:
            tool["blocked_domains"] = list(self.cfg.web_blocked_domains)
        return tool

    def research(self, stage_name: str, stage: StageConfig, *, content: list[dict]) -> ResearchResult:
        """Run the server-side web search tool and turn its citations into evidence items.

        Search results come back as content blocks (errors included, as HTTP 200), so nothing here raises
        on a failed search; the caller just sees fewer evidence items.
        """
        ns, extra = self._route(stage.model)
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        texts: list[str] = []
        evidence: list[EvidenceItem] = []
        seen: set[tuple[str, str]] = set()
        searches = 0
        reported = 0  # searches the API says it ran (usage), as opposed to result blocks this code recognised
        block_types: Counter[str] = Counter()
        citation_types: Counter[str] = Counter()

        for _ in range(MAX_CONTINUATIONS + 1):
            resp = ns.create(
                model=stage.model,
                max_tokens=stage.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[self._search_tool()],
                messages=messages,
                output_config={"effort": stage.effort},
                **extra,
            )
            self.usage.add(stage_name, stage.model, resp.usage)
            reported += self.usage.rows[-1].web_searches
            self._dump(f"research-{stage_name}", resp)
            self._check_stop(resp)

            for block in resp.content:
                block_types[block.type] += 1
                if block.type == "web_search_tool_result" and isinstance(block.content, list):
                    searches += 1
                elif block.type == "text":
                    texts.append(block.text)
                    for cit in getattr(block, "citations", None) or []:
                        citation_types[getattr(cit, "type", "?")] += 1
                        if getattr(cit, "type", "") != "web_search_result_location":
                            continue
                        key = (cit.url, (cit.cited_text or "")[:80])
                        if key in seen:
                            continue
                        seen.add(key)
                        evidence.append(
                            EvidenceItem(claim=block.text.strip(), snippet=cit.cited_text or "", url=cit.url, title=cit.title or cit.url)
                        )

            if resp.stop_reason != "pause_turn":
                break
            # Server-side tool loop hit its iteration cap: send the partial turn back to continue.
            messages.append({"role": "assistant", "content": resp.content})
        else:
            log.warning("%s: stopped after %d pause_turn continuations", stage_name, MAX_CONTINUATIONS)

        if not evidence:
            log.warning(
                "%s: no web evidence was collected (%d search(es) reported by the API, %d result block(s) recognised). "
                "Content blocks: %s. Citation types: %s. Set SNAPNOTES_DEBUG_DUMP to a folder to save the raw responses.",
                stage_name, reported, searches, dict(block_types) or "none", dict(citation_types) or "none",
            )
        return ResearchResult(text="".join(texts), evidence=evidence, searches_used=searches)
