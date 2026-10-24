"""Operator commands, run on the machine that holds the database.

    python -m app.admin check [--online] [--search]
    python -m app.admin show-epoch
    python -m app.admin rotate-epoch

`check` reads the same environment variables the server would and says whether they are usable, without printing any secret.

On Fly:  fly ssh console -C "python -m app.admin rotate-epoch"
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable

import anthropic

from app.core.auth import MIN_TOKEN_LENGTH, token_problem
from app.core.config import Settings, fact_check_enabled, fact_check_problem, gemini_api_key, provider_problem
from app.core.db import Database
from app.llm.client import models_in_use, unpriced_models
from app.store.sync import SyncStore

OK, WARN, BAD = "[ok]", "[!!]", "[XX]"  # plain ASCII so every Windows console shows them


def pasted_wrongly(value: str) -> str | None:
    """Mistakes made when typing a secret into a shell: leaving the <placeholder> brackets or quotes in, or a space."""
    if any(ch in value for ch in "<>\"' ") or any(not 33 <= ord(ch) <= 126 for ch in value):
        return "it contains a space, quote, bracket or unusual character. Did the <placeholder> brackets or quotes stay in? Paste only the value itself."
    return None


def preflight(
    cfg: Settings,
    *,
    online: bool = False,
    client_factory: Callable[[], anthropic.Anthropic] = anthropic.Anthropic,
    gemini_factory: Callable[[], Any] | None = None,
    search: bool = False,
) -> list[tuple[str, str]]:
    """What is right and wrong with this configuration, as (level, message) lines. Never includes a secret."""
    lines: list[tuple[str, str]] = []

    # access token
    problem = token_problem(cfg)
    mistake = pasted_wrongly(cfg.api_token) if cfg.api_token else None
    if mistake:  # said first: left-over placeholder text is also usually too short, and that is the more useful thing to hear
        lines.append((BAD, f"SNAPNOTES_TOKEN: {mistake}"))
    elif problem:
        lines.append((BAD, problem))
    elif not cfg.api_token:
        lines.append((WARN, "Authentication is switched off (SNAPNOTES_ALLOW_NO_AUTH=1). Only use this on your own computer."))
    else:
        note = " (short, allowed because demo mode or SNAPNOTES_ALLOW_WEAK_TOKEN is on)" if len(cfg.api_token) < MIN_TOKEN_LENGTH else ""
        lines.append((OK, f"SNAPNOTES_TOKEN is set ({len(cfg.api_token)} characters){note}."))

    # the AI provider: demo mode needs none; Gemini is the free tier; Claude is the default
    bad_provider = provider_problem(cfg)
    if bad_provider:
        lines.append((BAD, bad_provider))
    elif cfg.demo_mode:
        lines.append((WARN, "Demo mode is on (SNAPNOTES_DEMO=1): no AI is called, so no key is needed and every result is canned."))
        lines.extend(_spend_lines(cfg))
    elif cfg.provider == "gemini":
        lines.extend(_gemini_lines(cfg))
    else:
        lines.extend(_claude_credential_lines(client_factory))
        lines.extend(_spend_lines(cfg))

    # the fact check (flags statements in the notes that look factually wrong)
    bad_fact_check = fact_check_problem(cfg)
    if bad_fact_check:
        lines.append((BAD, bad_fact_check))
    elif fact_check_enabled(cfg):
        why = "on by default for the free Gemini provider" if cfg.fact_check == "auto" else "switched on with SNAPNOTES_FACT_CHECK"
        lines.append((OK, f"Fact check: on ({why}). Statements in the notes that look factually wrong are listed and get no card; it uses the model's own knowledge, so it can miss errors or doubt a fair simplification."))
    else:
        lines.append((OK, "Fact check: off (SNAPNOTES_FACT_CHECK=1 turns it on)."))

    # where the data goes
    where = "exists" if cfg.data_dir.exists() else "will be created on first start"
    lines.append((OK, f"Data folder: {cfg.data_dir.resolve()} ({where})."))

    if online and not cfg.demo_mode and not bad_provider:
        lines.append(_ask_gemini(cfg, gemini_factory) if cfg.provider == "gemini" else _ask_claude(client_factory))
    if search and not cfg.demo_mode and not bad_provider:
        if cfg.provider == "gemini":
            lines.append(_try_search(cfg, gemini_factory))
        else:
            lines.append((WARN, "--search tests Gemini's Google Search only; this server is set to use Claude."))
    return lines


def _claude_credential_lines(client_factory: Callable[[], anthropic.Anthropic]) -> list[tuple[str, str]]:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        mistake = pasted_wrongly(key)
        if mistake:
            return [(BAD, f"ANTHROPIC_API_KEY: {mistake}")]
        if not key.startswith("sk-ant-"):
            return [(WARN, f"ANTHROPIC_API_KEY is set ({len(key)} characters) but does not start with sk-ant-, which API keys normally do.")]
        return [(OK, f"ANTHROPIC_API_KEY is set (starts with sk-ant-, {len(key)} characters).")]
    if has_credentials(client_factory):
        return [(OK, "Claude credentials were found (not through ANTHROPIC_API_KEY, for example a login made with `ant auth login`).")]
    return [(BAD, "No Claude credentials found. Set ANTHROPIC_API_KEY in this same window before starting the server.")]


def _spend_lines(cfg: Settings) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    if cfg.max_job_cost_usd <= 0:
        lines.append((BAD, f"SNAPNOTES_MAX_JOB_COST_USD is {cfg.max_job_cost_usd}: every document would be stopped straight away."))
    else:
        lines.append((OK, f"Each document will stop once it has spent about ${cfg.max_job_cost_usd:.2f} (SNAPNOTES_MAX_JOB_COST_USD); web search fees are not counted."))
    missing = unpriced_models(cfg)
    if missing:
        lines.append((BAD, f"No price is known for {', '.join(missing)}, so the spend limit cannot work. Add it with SNAPNOTES_EXTRA_PRICES (see the README)."))
    else:
        lines.append((OK, f"Models in use: {', '.join(sorted(models_in_use(cfg)))} (all have a known price)."))
    return lines


def _gemini_lines(cfg: Settings) -> list[tuple[str, str]]:
    key = gemini_api_key()
    lines: list[tuple[str, str]] = []
    mistake = pasted_wrongly(key) if key else None
    if not key:
        lines.append((BAD, "GEMINI_API_KEY is not set (GOOGLE_API_KEY also works). Make a free key in Google AI Studio and set it in this same window before starting the server."))
    elif mistake:
        lines.append((BAD, f"GEMINI_API_KEY: {mistake}"))
    else:
        lines.append((OK, f"GEMINI_API_KEY is set ({len(key)} characters)."))
    lines.append((OK, f"AI provider: gemini, model {cfg.gemini_model} (SNAPNOTES_GEMINI_MODEL). Free tier: there is no cost, so no per-document spend limit applies; Google's own per-minute and daily limits do."))
    lines.append((WARN, "Notes sent to the free Gemini tier may be used by Google to improve its products. Use notes that are not private."))
    if fact_check_enabled(cfg):
        lines.append((WARN, "Web research with Gemini (Google Search) runs only when a statement in your notes looks wrong, and sends those statements and the topic, not your notes, to Google. Google lists 5,000 free searches a month; check your own limit in Google AI Studio."))
        if cfg.ai_corrections:
            lines.append((OK, "AI corrections: on. When no source settles a wrong statement, a card with the right fact is made from the AI's own knowledge, labelled 'not sourced' (SNAPNOTES_AI_CORRECTIONS=0 turns this off)."))
        else:
            lines.append((OK, "AI corrections: off. Corrected cards are made only when a web source backs them."))
    else:
        lines.append((WARN, "No web research with Gemini while the fact check is off: every card comes from the notes alone."))
    missing = unpriced_models(cfg)
    if missing:  # cannot happen unless the price table is edited; kept so the spend guard stays honest
        lines.append((BAD, f"No price is known for {', '.join(missing)}."))
    return lines


def has_credentials(client_factory: Callable[[], anthropic.Anthropic]) -> bool:
    """The SDK builds a client happily with nothing configured and only complains when the first request is sent,
    so look at what it found instead of whether it raised."""
    try:
        client = client_factory()
    except (TypeError, anthropic.AnthropicError):
        return False
    return any(getattr(client, name, None) for name in ("api_key", "auth_token", "credentials", "custom_auth"))


def _ask_claude(client_factory: Callable[[], anthropic.Anthropic]) -> tuple[str, str]:
    """Listing models is free and proves the key is accepted; it uses no tokens."""
    try:
        client_factory().with_options(max_retries=0, timeout=15.0).models.list(limit=1)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        return BAD, "Claude rejected the credentials. Check that the whole key was copied and that it has not been revoked."
    except anthropic.APIConnectionError:
        return WARN, "Could not reach Claude (check your internet connection), so the key has not been proven either way."
    except anthropic.APIStatusError as exc:
        return BAD, f"Claude answered with an error ({exc.status_code}): {str(exc.message)[:200]}"
    except (TypeError, anthropic.AnthropicError) as exc:
        return BAD, f"Could not ask Claude: {str(exc)[:200]}"
    return OK, "Claude accepted the credentials."


def _ask_gemini(cfg: Settings, gemini_factory: Callable[[], Any] | None) -> tuple[str, str]:
    """Listing models is free; it proves the key is accepted and shows whether the chosen model is one this key can use."""
    from google.genai import errors

    try:
        if gemini_factory is None:
            from google import genai

            client = genai.Client(api_key=gemini_api_key())
        else:
            client = gemini_factory()
        available = {str(getattr(m, "name", "")).split("/")[-1] for m in client.models.list()}
    except errors.APIError as exc:
        if exc.code in (400, 401, 403):
            return BAD, "Gemini rejected the key. Check that the whole key was copied and that it has not been deleted in Google AI Studio."
        return BAD, f"Gemini answered with an error ({exc.code}): {str(exc.message)[:200]}"
    except Exception as exc:  # no network, a proxy, a timeout: the key is simply not proven either way
        return WARN, f"Could not reach Gemini ({type(exc).__name__}), so the key has not been proven either way."
    wanted = cfg.gemini_model
    if wanted in available:
        return OK, f"Gemini accepted the key and {wanted} is available to it."
    flash = sorted(n for n in available if "flash" in n)[:8]
    hint = f" Fast models it lists: {', '.join(flash)}." if flash else ""
    return WARN, f"Gemini accepted the key, but {wanted} is not in its model list.{hint} Set SNAPNOTES_GEMINI_MODEL to one of them."


def _try_search(cfg: Settings, gemini_factory: Callable[[], Any] | None) -> tuple[str, str]:
    """Run ONE real Google Search through Gemini with a harmless question and say exactly what came back (it uses one of the
    free searches). This is the same kind of call the corrections make, but without hiding an error behind "no sources"."""
    from google import genai
    from google.genai import types

    from app.llm.gemini import _search_count, describe_search_failure, evidence_from_grounding

    try:
        client = gemini_factory() if gemini_factory is not None else genai.Client(api_key=gemini_api_key())
        resp = client.models.generate_content(
            model=cfg.gemini_model,
            contents="At what temperature does water boil at sea level, in degrees Celsius? Answer in one sentence.",
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], max_output_tokens=400),
        )
    except Exception as exc:
        return BAD, f"Google Search through {cfg.gemini_model} failed: {describe_search_failure(exc)}"
    sources = evidence_from_grounding(resp)
    searches = _search_count(resp)
    if sources:
        return OK, f"Google Search works: {searches} search(es), {len(sources)} source(s) from {', '.join(sorted({s.title for s in sources})[:3])}."
    return WARN, f"Google Search answered but returned no usable sources ({searches} search(es)). Try another SNAPNOTES_GEMINI_MODEL."


def run_check(online: bool, search: bool = False) -> int:
    try:
        cfg = Settings()
    except ValueError as exc:  # a number or JSON setting that cannot be read
        print(f"{BAD} A setting could not be read: {exc}")
        return 1
    lines = preflight(cfg, online=online, search=search)
    for level, text in lines:
        print(f"{level} {text}")
    problems = sum(level == BAD for level, _ in lines)
    warnings = sum(level == WARN for level, _ in lines)
    if problems:
        print(f"\n{problems} problem(s) marked {BAD}: fix them in this window, then run the check again. The server would not start (or not work) as is.")
        return 1
    print(f"\nLooks good{f' ({warnings} warning(s) marked {WARN})' if warnings else ''}. This window is ready to start the server.")
    if not online and not cfg.demo_mode:
        ai = "Gemini" if cfg.provider == "gemini" else "Claude"
        print(f"To also test that {ai} accepts your key (free, no tokens used): python -m app.admin check --online")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.admin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check the environment variables before starting the server (prints no secrets)")
    check.add_argument("--online", action="store_true", help="also ask the AI provider (Claude or Gemini) whether it accepts the key (free, uses no tokens)")
    check.add_argument("--search", action="store_true", help="also run one real Google Search through Gemini and show exactly what happens (uses one free search)")
    commands.add_parser("show-epoch", help="print the database's sync identity")
    commands.add_parser(
        "rotate-epoch",
        help="give the database a new sync identity. Run after restoring a backup: every phone then discards its cached library, "
        "downloads everything again and re-uploads its review history.",
    )
    args = parser.parse_args(argv)

    if args.command == "check":
        return run_check(args.online, args.search)  # reads settings only; never touches the database

    cfg = Settings()
    db = Database(cfg.db_path)
    db.init()  # also brings an older database up to date
    store = SyncStore(db)

    if args.command == "show-epoch":
        print(store.epoch())
    else:
        print(f"old epoch: {store.epoch()}")
        print(f"new epoch: {store.rotate_epoch()}")
        print("Phones will resynchronise from scratch the next time they connect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
