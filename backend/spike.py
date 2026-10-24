"""Milestone 0 spike: run the whole notes -> flashcards pipeline from the command line.

    python spike.py notes.pdf
    python spike.py page1.jpg page2.jpg --interactive      # choose which web sources to reject
    python spike.py slides.pptx --out spike_out

Needs Claude credentials (ANTHROPIC_API_KEY, or `ant auth login`), or with SNAPNOTES_PROVIDER=gemini a Gemini key
(GEMINI_API_KEY, free tier, notes-only cards). Prints the cards, and what the run cost,
so model/effort choices per stage (see app/core/config.py) can be tuned from real numbers.

Stops between stages once the estimated spend passes SNAPNOTES_MAX_JOB_COST_USD (default 2.00), like the server does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic

from app.core.config import settings
from app.ingest.loader import UnsupportedFileError, load_source
from app.llm.client import MissingCredentials, RefusalError, TruncatedError, UsageLog
from app.llm.factory import build_llm
from app.llm.schemas import EvidenceItem
from app.pipeline.orchestrator import PipelineResult, run_pipeline
from app.pipeline.worker import CostCeilingExceeded, check_cost_ceiling, describe_failure


def interactive_approval(items: list[EvidenceItem]) -> list[EvidenceItem]:
    if not items:
        print("\nNo web evidence was found.")
        return items
    print("\nWeb evidence found:")
    for i, e in enumerate(items, 1):
        print(f"  {i:>2}. {e.claim[:140]}\n      {e.title} - {e.url}")
    raw = input("\nNumbers to REJECT (comma separated, blank to keep all): ").strip()
    rejected = {int(x) for x in raw.replace(" ", "").split(",") if x.isdigit()}
    for i, e in enumerate(items, 1):
        e.approved = i not in rejected
    return items


def print_report(result: PipelineResult) -> None:
    u = result.understanding
    print(f"\n=== {u.main_topic} ({u.subject_area}) ===")
    print("Subtopics:", "; ".join(u.subtopics))
    print("Gaps:", "; ".join(u.gaps) or "none")

    print(f"\n--- Web evidence ({len(result.evidence)}) ---")
    for e in result.evidence:
        mark = "x" if e.approved else " "
        print(f"[{mark}] {e.claim[:120]}  <{e.url}>")

    print(f"\n--- Cards ({len(result.cards)}) ---")
    for i, rc in enumerate(result.cards):
        c = rc.card
        flags = []
        if not rc.grounded:
            flags.append("UNGROUNDED")
        if not rc.atomic:
            flags.append("NOT ATOMIC")
        if rc.duplicate_of is not None:
            flags.append(f"DUPLICATE OF #{rc.duplicate_of}")
        flag_text = f"  !! {', '.join(flags)}" if flags else ""
        print(f"#{i} [{c.type}/{c.origin}/{c.difficulty}] {c.front}\n     -> {c.back}\n     src: {c.source_ref}{flag_text}")
        if rc.issue:
            print(f"     note: {rc.issue}")

    kept = sum(1 for rc in result.cards if rc.keep)
    print(f"\nKept {kept}/{len(result.cards)} cards after verification and de-duplication.")

    print_usage(result.usage)


def print_usage(usage: UsageLog) -> None:
    print("\n--- Usage ---")
    for r in usage.rows:
        cost = f"${r.cost:.4f}" if r.cost is not None else "n/a"
        print(
            f"{r.stage:<11} {r.model:<18} in={r.input_tokens:>7} cache_read={r.cache_read:>7} "
            f"cache_write={r.cache_write:>7} out={r.output_tokens:>6} searches={r.web_searches} {cost}"
        )
    total = usage.total_cost
    print(f"Estimated total (excludes web search fees): {'$%.4f' % total if total is not None else 'n/a'}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="PDF, image(s), DOCX, PPTX, MD or TXT")
    parser.add_argument("--interactive", action="store_true", help="choose which web sources to reject")
    parser.add_argument("--out", default="spike_out", help="directory for the JSON result")
    args = parser.parse_args()

    try:
        source = load_source(args.files)
    except (UnsupportedFileError, OSError) as exc:
        print(f"Cannot read input: {exc}", file=sys.stderr)
        return 2

    try:
        client = build_llm(settings)  # Claude by default; SNAPNOTES_PROVIDER=gemini for the free tier
    except MissingCredentials as exc:
        print(exc, file=sys.stderr)
        return 1
    except ValueError as exc:  # an unknown SNAPNOTES_PROVIDER
        print(exc, file=sys.stderr)
        return 2
    ceiling = client.cfg.max_job_cost_usd
    if client.cfg.provider != "anthropic":
        print(f"AI provider: {client.cfg.provider} (no web research; cards come from the notes alone).")

    def on_stage(stage: str) -> None:
        print(f"-> {stage}...", flush=True)
        check_cost_ceiling(0.0, client.usage, ceiling)  # the same between-stages limit the server applies to each document

    print(f"Cost limit ${ceiling:.2f} (SNAPNOTES_MAX_JOB_COST_USD), checked before each stage; web search fees are not counted.")
    try:
        result = run_pipeline(
            client,
            source,
            approve_evidence=interactive_approval if args.interactive else None,
            on_stage=on_stage,
        )
    except CostCeilingExceeded as exc:
        print(f"\n{exc}", file=sys.stderr)
        print_usage(client.usage)
        return 1
    except anthropic.AuthenticationError:
        print("Authentication failed. Set ANTHROPIC_API_KEY or run `ant auth login`.", file=sys.stderr)
        return 1
    except TypeError as exc:
        # The SDK raises a bare TypeError (not an AnthropicError) when no credentials are configured.
        if "authentication method" in str(exc):
            print("No Claude credentials found. Set ANTHROPIC_API_KEY or run `ant auth login`.", file=sys.stderr)
            return 1
        raise
    except anthropic.AnthropicError as exc:  # includes "no credentials found"
        print(f"Claude API error: {exc}", file=sys.stderr)
        return 1
    except (RefusalError, TruncatedError) as exc:
        print(f"Pipeline stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if type(exc).__module__.startswith("google."):  # an error from Gemini's SDK
            print(f"Gemini error: {describe_failure(exc)}", file=sys.stderr)
            return 1
        raise

    print_report(result)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.files[0]).stem}.json"
    out_path.write_text(
        json.dumps(
            {
                "understanding": result.understanding.model_dump(),
                "evidence": [e.model_dump() for e in result.evidence],
                "cards": [rc.model_dump() | {"keep": rc.keep} for rc in result.cards],
                "usage": [r.__dict__ | {"cost": r.cost} for r in result.usage.rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
