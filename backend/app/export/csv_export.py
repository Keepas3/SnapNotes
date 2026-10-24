"""CSV export for spreadsheets and other flashcard apps."""
from __future__ import annotations

import csv
import io
from typing import Any

from app.export import source_label

HEADER = ["deck", "type", "front", "back", "source", "origin", "tags", "difficulty"]

# A cell that starts with one of these is read as a formula by Excel and Google Sheets, and note text can come from
# documents and web pages the user did not write. Prefixing a quote makes spreadsheets treat it as plain text.
_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value: str) -> str:
    return "'" + value if value.startswith(_FORMULA_STARTS) else value


def build_csv(deck: dict[str, Any], cards: list[dict[str, Any]]) -> bytes:
    """UTF-8 with a byte-order mark, which is what makes Excel read non-ASCII text correctly."""
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(HEADER)
    for card in cards:
        writer.writerow(
            [
                safe_cell(value)
                for value in (
                    deck["name"],
                    card["type"],
                    card["front"],
                    card["back"],
                    source_label(card),
                    card["origin"],
                    " ".join(card.get("tags", [])),
                    card["difficulty"],
                )
            ]
        )
    return out.getvalue().encode("utf-8-sig")
