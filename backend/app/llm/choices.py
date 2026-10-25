"""Wrong answers for multiple choice, suggested by the AI.

These are only ever suggestions: the caller shows them to the user, who edits or drops them before anything is saved. The model can
write wrong answers that are sloppy, too obviously wrong, or accidentally right, so everything it returns is checked here first and a
card whose suggestions do not hold up is reported as skipped rather than given weak options.
"""
from __future__ import annotations

import json
import re
from typing import Any

from rapidfuzz import fuzz

from app.llm.base import LLM
from app.llm.prompts import choices_task
from app.llm.schemas import ChoiceSet
from app.store.library import MAX_CHOICE_LENGTH

# Closer than this to the right answer (rapidfuzz token_sort_ratio, 0-100) and a "wrong" answer is really the right one reworded.
TOO_SIMILAR = 85
# A wrong answer needs at least this many companions left after checking, or the card is skipped (the right answer plus two wrong ones is the smallest sensible question).
MIN_USABLE = 2


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def usable_wrong_answers(raw: list[str], correct: str, wanted: int) -> list[str]:
    """The suggestions that are fit to show: trimmed, distinct, not too long, and clearly not the right answer."""
    right = _norm(correct)
    kept: list[str] = []
    for item in raw:
        text = " ".join(str(item).split())
        norm = _norm(text)
        if not norm or len(text) > MAX_CHOICE_LENGTH or norm == right:
            continue
        if fuzz.token_sort_ratio(norm, right) >= TOO_SIMILAR:
            continue  # the right answer reworded
        if len(right) >= 4 and right in norm:
            continue  # contains the right answer ("Not the mitochondria", "The mitochondria and the nucleus")
        if any(norm == _norm(other) or fuzz.token_sort_ratio(norm, _norm(other)) >= 95 for other in kept):
            continue
        kept.append(text)
        if len(kept) == wanted:
            break
    return kept


def suggest_choices(client: LLM, cards: list[dict[str, Any]], wrong_per_card: int) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Ask the AI for wrong answers for `cards` (dicts with `id`, `front`, `back`) in one call.

    Returns (suggestions by card id, the reason for each card that got none)."""
    payload = json.dumps([{"index": i, "front": c["front"], "back": c["back"]} for i, c in enumerate(cards)], ensure_ascii=False)
    result = client.parse(
        "choices",
        client.cfg.verify,  # the cheap, low-effort stage: this is a small text-only task
        content=[{"type": "text", "text": choices_task(payload, wrong_per_card)}],
        output_model=ChoiceSet,
    )
    answered: dict[int, list[str]] = {}
    for item in result.items:
        answered.setdefault(item.index, item.wrong_answers)  # the first answer for a card wins

    suggestions: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    for i, card in enumerate(cards):
        raw = answered.get(i)
        if raw is None:
            skipped[card["id"]] = "The AI gave no answer for this card."
            continue
        good = usable_wrong_answers(raw, card["back"], wrong_per_card)
        if len(good) >= MIN_USABLE:
            suggestions[card["id"]] = good
        elif not raw:
            skipped[card["id"]] = "The AI found no clear wrong answers for this question."
        else:
            skipped[card["id"]] = "The AI's suggestions were too close to the right answer to use."
    return suggestions, skipped
