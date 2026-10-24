"""Central settings. Everything model-related is configurable so stages can be re-tuned after the spike."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StageConfig:
    model: str
    effort: str  # low | medium | high | xhigh | max
    max_tokens: int


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_prices(raw: str) -> dict[str, tuple[float, float]]:
    """`{"model-id": [input_usd_per_million_tokens, output_usd_per_million_tokens]}` -> dict. Empty string -> {}."""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        prices = {str(model): (float(pair[0]), float(pair[1])) for model, pair in data.items()}
        if any(i < 0 or o < 0 for i, o in prices.values()):
            raise ValueError("prices cannot be negative")
        return prices
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise ValueError(
            f'SNAPNOTES_EXTRA_PRICES must look like {{"claude-model-id": [input_usd_per_1m_tokens, output_usd_per_1m_tokens]}} ({exc})'
        ) from exc


PROVIDERS = ("anthropic", "gemini")


def gemini_api_key() -> str:
    """The Gemini key from the environment (GEMINI_API_KEY, or GOOGLE_API_KEY which Google's own tools use)."""
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def provider_problem(cfg: "Settings") -> str | None:
    """Why this AI provider setting cannot be used, or None. Checked at startup and by `admin check`, not while
    reading settings, so that a typo is reported in plain words instead of crashing the import."""
    if cfg.provider not in PROVIDERS:
        return f"SNAPNOTES_PROVIDER is {cfg.provider!r}; it must be one of: {', '.join(PROVIDERS)}."
    return None


FACT_CHECK_ON = ("1", "on", "true", "yes")
FACT_CHECK_OFF = ("0", "off", "false", "no")


def fact_check_enabled(cfg: "Settings") -> bool:
    """Whether the notes and cards are also checked against the model's own general knowledge. `auto` (the default) means
    on for the free provider, which has no web research to catch a mistake in the notes, and off for Claude."""
    if cfg.fact_check in FACT_CHECK_ON:
        return True
    if cfg.fact_check in FACT_CHECK_OFF:
        return False
    return cfg.provider == "gemini"


def fact_check_problem(cfg: "Settings") -> str | None:
    if cfg.fact_check == "auto" or cfg.fact_check in FACT_CHECK_ON + FACT_CHECK_OFF:
        return None
    return f"SNAPNOTES_FACT_CHECK is {cfg.fact_check!r}; use auto, 1 or 0."


@dataclass(frozen=True)
class Settings:
    # Flag statements in the notes that look factually wrong: auto (on for gemini, off for anthropic), 1 or 0.
    fact_check: str = field(default_factory=lambda: _env("SNAPNOTES_FACT_CHECK", "auto").strip().lower() or "auto")

    # When no web source settles a statement the fact check found wrong, still make one card with the right fact from the model's
    # own knowledge (labelled "From AI knowledge, not sourced" and held as a draft). 0 means sourced corrections only.
    ai_corrections: bool = field(default_factory=lambda: _env("SNAPNOTES_AI_CORRECTIONS", "1").strip().lower() not in FACT_CHECK_OFF)

    # Which AI does the work: "anthropic" (Claude, the default, with web research) or "gemini" (Google's free tier:
    # notes-only cards, no web research). One choice for the whole server.
    provider: str = field(default_factory=lambda: _env("SNAPNOTES_PROVIDER", "anthropic").strip().lower() or "anthropic")
    # Model used for every stage when the provider is gemini. Which models are free changes; `admin check --online` lists
    # what a key can use.
    gemini_model: str = field(default_factory=lambda: _env("SNAPNOTES_GEMINI_MODEL", "gemini-3.8-flash").strip())

    # Per-stage model/effort. Defaults to Opus 5.5 everywhere; swap any stage with an env var
    # (e.g. SNAPNOTES_VERIFY_MODEL=claude-sonnet-5-5) once the spike has measured cost vs quality.
    understand: StageConfig = field(default_factory=lambda: StageConfig(
        _env("SNAPNOTES_UNDERSTAND_MODEL", "claude-opus-5-5"), "medium", 8000))
    enrich: StageConfig = field(default_factory=lambda: StageConfig(
        _env("SNAPNOTES_ENRICH_MODEL", "claude-opus-5-5"), "medium", 16000))
    generate: StageConfig = field(default_factory=lambda: StageConfig(
        _env("SNAPNOTES_GENERATE_MODEL", "claude-opus-5-5"), "medium", 16000))
    verify: StageConfig = field(default_factory=lambda: StageConfig(
        _env("SNAPNOTES_VERIFY_MODEL", "claude-opus-5-5"), "low", 16000))

    # Prices ($ per 1M tokens) for models that are not in the built-in table, so the spend limit keeps working when a
    # stage is pointed at a newer model. The server refuses to start if a configured model has no price.
    extra_prices: dict[str, tuple[float, float]] = field(default_factory=lambda: parse_prices(_env("SNAPNOTES_EXTRA_PRICES", "")))

    # Opt into server-side refusal fallbacks (beta). Not supported for Haiku 5.5.
    use_fallbacks: bool = field(default_factory=lambda: _env("SNAPNOTES_FALLBACKS", "1") == "1")

    # Debugging aid, off by default: a folder to save the raw web-research responses in (they contain page text from the web).
    debug_dump_dir: Path | None = field(default_factory=lambda: Path(v) if (v := _env("SNAPNOTES_DEBUG_DUMP", "").strip()) else None)

    # Web search guard rails
    web_search_max_uses: int = field(default_factory=lambda: int(_env("SNAPNOTES_WEB_MAX_USES", "6")))
    web_allowed_domains: tuple[str, ...] = field(default_factory=lambda: tuple(
        d for d in _env("SNAPNOTES_WEB_ALLOWED_DOMAINS", "").split(",") if d))
    web_blocked_domains: tuple[str, ...] = field(default_factory=lambda: tuple(
        d for d in _env("SNAPNOTES_WEB_BLOCKED_DOMAINS", "").split(",") if d))

    # Upload limits (native PDF block: 32 MB request, 600 pages). Preparing a file for Claude needs roughly five times
    # its size in memory (raw bytes, base64 text twice, the request body), so lower this on a small machine.
    max_upload_bytes: int = field(default_factory=lambda: int(float(_env("SNAPNOTES_MAX_UPLOAD_MB", "30")) * 1024 * 1024))
    max_files_per_upload: int = 20
    max_cards: int = field(default_factory=lambda: int(_env("SNAPNOTES_MAX_CARDS", "40")))

    # --- API server -------------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("SNAPNOTES_DATA_DIR", "data")))
    # Single personal bearer token. The server refuses to start without one unless allow_no_auth is set (local dev only),
    # and refuses a short one (see auth.token_problem) unless demo mode or allow_weak_token is on. Whitespace is stripped
    # because a trailing newline pasted into a secret is a classic reason for "the right token is rejected".
    api_token: str = field(default_factory=lambda: _env("SNAPNOTES_TOKEN", "").strip())
    allow_no_auth: bool = field(default_factory=lambda: _env("SNAPNOTES_ALLOW_NO_AUTH", "0") == "1")
    allow_weak_token: bool = field(default_factory=lambda: _env("SNAPNOTES_ALLOW_WEAK_TOKEN", "0") == "1")
    # Request header naming the real client address when behind a proxy that sets it (Fly.io: Fly-Client-IP). Used to
    # rate-limit per client. Set to an empty string anywhere that header is not set by your own proxy, because a client
    # could otherwise choose its own address.
    client_ip_header: str = field(default_factory=lambda: _env("SNAPNOTES_CLIENT_IP_HEADER", "fly-client-ip").strip().lower())
    # Largest request body for every endpoint except the upload one (which is max_upload_bytes plus framing).
    max_body_kb: int = field(default_factory=lambda: int(_env("SNAPNOTES_MAX_BODY_KB", "2048")))
    docs_enabled: bool = field(default_factory=lambda: _env("SNAPNOTES_DOCS", "1") == "1")
    # Browser origins allowed to call the API (only needed for the Expo web preview during development).
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in _env("SNAPNOTES_CORS_ORIGINS", "").split(",") if o.strip()))
    # Canned responses instead of calling Claude: develop the app with no key and no spend.
    demo_mode: bool = field(default_factory=lambda: _env("SNAPNOTES_DEMO", "0") == "1")
    demo_delay: float = field(default_factory=lambda: float(_env("SNAPNOTES_DEMO_DELAY", "1.2")))

    # Spend and memory protection. One job at a time by default: a maximum-size PDF takes ~150 MB while it is prepared,
    # so two at once on a small machine could run it out of memory. Documents simply queue behind each other.
    job_concurrency: int = field(default_factory=lambda: int(_env("SNAPNOTES_JOB_CONCURRENCY", "1")))
    max_jobs_per_hour: int = field(default_factory=lambda: int(_env("SNAPNOTES_MAX_JOBS_PER_HOUR", "10")))
    max_job_cost_usd: float = field(default_factory=lambda: float(_env("SNAPNOTES_MAX_JOB_COST_USD", "2.0")))
    requests_per_minute: int = field(default_factory=lambda: int(_env("SNAPNOTES_REQUESTS_PER_MINUTE", "240")))

    # Privacy: remove the uploaded originals once cards have been generated.
    delete_originals: bool = field(default_factory=lambda: _env("SNAPNOTES_DELETE_ORIGINALS", "1") == "1")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "snapnotes.db"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()
