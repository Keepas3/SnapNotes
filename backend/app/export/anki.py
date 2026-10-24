"""Anki package (.apkg) export, built with genanki.

Re-exporting a deck and importing it again updates the notes already in Anki instead of duplicating them: every note
gets a GUID derived from the card id, and the deck and note types keep fixed ids.
"""
from __future__ import annotations

import io
import itertools
import re
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import genanki

from app.export import domain_of, html_attr, html_text, safe_http_url

# Anki identifies note types by id: these must never change, or re-imports would create duplicate note types.
BASIC_MODEL_ID = 1830716181
CLOZE_MODEL_ID = 1830716182

_CSS = """
.card { font-family: Arial, Helvetica, sans-serif; font-size: 20px; line-height: 1.4; text-align: center; color: #111; background-color: #fff; }
.nightMode .card, .night_mode .card { color: #eee; background-color: #1c1c1e; }
.cloze { font-weight: bold; color: #4f46e5; }
.source { margin-top: 1.2em; font-size: 12px; color: #888; }
.source a { color: #888; }
"""

BASIC_MODEL = genanki.Model(
    BASIC_MODEL_ID,
    "SnapNotes Basic",
    fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Source"}],
    templates=[
        {
            "name": "Card 1",
            "qfmt": "{{Front}}",
            "afmt": '{{FrontSide}}<hr id="answer">{{Back}}<div class="source">{{Source}}</div>',
        }
    ],
    css=_CSS,
)

CLOZE_MODEL = genanki.Model(
    CLOZE_MODEL_ID,
    "SnapNotes Cloze",
    fields=[{"name": "Text"}, {"name": "Extra"}, {"name": "Source"}],
    templates=[
        {
            "name": "Cloze",
            "qfmt": "{{cloze:Text}}",
            "afmt": '{{cloze:Text}}<br>{{Extra}}<div class="source">{{Source}}</div>',
        }
    ],
    css=_CSS,
    model_type=genanki.Model.CLOZE,
)

_CLOZE_MARKER = re.compile(r"\{\{c\d+::.+?\}\}", re.DOTALL)


def anki_tag(tag: str) -> str:
    """Anki tags cannot contain spaces."""
    return re.sub(r"\s+", "_", tag.strip())[:60]


def card_tags(card: dict[str, Any]) -> list[str]:
    tags = [anki_tag(t) for t in card.get("tags", [])]
    tags += ["snapnotes", f"source-{card['origin']}"]
    return [t for t in dict.fromkeys(tags) if t]  # no blanks, no duplicates, order kept


def source_html(card: dict[str, Any]) -> str:
    ref = card["source_ref"]
    if card["origin"] == "web":
        url = safe_http_url(ref)
        if url:
            return f'From the web: <a href="{html_attr(url)}">{html_text(domain_of(url))}</a>'
        return f"From the web: {html_text(ref)}" if ref else "From the web"
    if card["origin"] == "ai":
        return f"From AI knowledge (not sourced): {html_text(ref)}" if ref else "From AI knowledge (not sourced)"
    if card["origin"] == "manual":
        return "Added by you"
    return f"From your notes: {html_text(ref)}" if ref else "From your notes"


def deck_id_for(deck_uuid: str) -> int:
    """Stable Anki deck id (Anki expects one between 2**30 and 2**31) derived from our deck id."""
    return (1 << 30) + int(deck_uuid[:8], 16) % (1 << 30)


def build_note(card: dict[str, Any]) -> genanki.Note:
    guid = genanki.guid_for("snapnotes-card", card["id"])
    tags = card_tags(card)
    source = source_html(card)

    if card["type"] == "cloze" and _CLOZE_MARKER.search(card["front"]):
        # {{c1::answer}} is Anki's own cloze syntax, so the text goes in unchanged (apart from HTML escaping).
        return genanki.Note(model=CLOZE_MODEL, fields=[html_text(card["front"]), html_text(card["back"]), source], tags=tags, guid=guid)

    # A "cloze" card that lost its blanks while being edited would produce a note with no cards in Anki;
    # it becomes a plain front/back card instead.
    return genanki.Note(model=BASIC_MODEL, fields=[html_text(card["front"]), html_text(card["back"]), source], tags=tags, guid=guid)


def build_apkg(deck: dict[str, Any], cards: list[dict[str, Any]], timestamp: float | None = None) -> bytes:
    """An .apkg file for `deck` containing `cards`. `timestamp` makes the output reproducible (used by tests)."""
    anki_deck = genanki.Deck(deck_id_for(deck["id"]), deck["name"].replace("::", " - "), description=deck.get("topic", ""))
    for card in cards:
        anki_deck.add_note(build_note(card))

    package = genanki.Package(anki_deck)
    ts = time.time() if timestamp is None else timestamp

    # genanki's own write_to_file leaves its temporary database behind, so assemble the zip here
    # (an .apkg is a zip holding collection.anki2 plus a media manifest).
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "collection.anki2"
        conn = sqlite3.connect(db_path)
        try:
            package.write_to_db(conn.cursor(), ts, itertools.count(int(ts * 1000)))
            conn.commit()
        finally:
            conn.close()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(db_path, "collection.anki2")
            z.writestr("media", "{}")
    return buf.getvalue()
