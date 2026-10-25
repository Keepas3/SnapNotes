"""Decks and cards: the persistent flashcard library."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from app.core.db import Database

CARD_STATUSES = ("draft", "approved", "rejected")
CARD_DIFFICULTIES = ("easy", "medium", "hard")
MAX_FOLDER_NAME = 80
MANUAL_CARD_TYPES = ("basic", "cloze")
MANUAL_SOURCE = "Added by you"  # `source_ref` of a card the user typed in, and what the app and exports show for its origin "manual"
_CLOZE_BLANK = re.compile(r"\{\{c\d+::.+?\}\}", re.DOTALL)


class FolderNotFound(LookupError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


MAX_CHOICES = 6  # wrong answers kept per card: the app shows at most 4 of them next to the right one
MAX_CHOICE_LENGTH = 200


def _same_text(a: str, b: str) -> bool:
    """Equal ignoring case, spacing and punctuation: 'Mitochondria.' and ' mitochondria ' are the same answer."""
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()

    return norm(a) == norm(b)


def clean_choices(choices: Iterable[str], correct: str, card_type: str) -> list[str]:
    """The wrong answers to store for a card: trimmed, with blanks and repeats dropped. Raises ValueError (a sentence for the user)
    when one is the right answer, is too long, there are too many, or the card is fill-in-the-blank (no options for those)."""
    cleaned: list[str] = []
    for raw in choices:
        text = " ".join(str(raw).split())
        if not text or any(_same_text(text, kept) for kept in cleaned):
            continue
        if len(text) > MAX_CHOICE_LENGTH:
            raise ValueError(f"Keep each wrong answer under {MAX_CHOICE_LENGTH} characters.")
        if _same_text(text, correct):
            raise ValueError(f"“{text[:40]}” is the right answer, so it cannot be one of the wrong ones.")
        cleaned.append(text)
    if cleaned and card_type == "cloze":
        raise ValueError("Multiple choice is not available for fill-in-the-blank cards.")
    if len(cleaned) > MAX_CHOICES:
        raise ValueError(f"A card can have at most {MAX_CHOICES} wrong answers.")
    return cleaned


def card_from_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["choices"] = json.loads(d.pop("choices_json", None) or "[]")
    d["tags"] = json.loads(d.pop("tags_json"))
    d["atomic"] = bool(d["atomic"])
    d["grounded"] = bool(d["grounded"])
    return d


def deck_from_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["counts"] = {"draft": d.pop("n_draft") or 0, "approved": d.pop("n_approved") or 0, "rejected": d.pop("n_rejected") or 0}
    return d


_DECK_SELECT = """
SELECT d.*,
       SUM(c.status = 'draft') AS n_draft,
       SUM(c.status = 'approved') AS n_approved,
       SUM(c.status = 'rejected') AS n_rejected
FROM decks d LEFT JOIN cards c ON c.deck_id = d.id
"""


class Library:
    def __init__(self, db: Database):
        self.db = db

    # -- decks ---------------------------------------------------------------------------

    def list_decks(self) -> list[dict[str, Any]]:
        with self.db.conn() as c:
            rows = c.execute(f"{_DECK_SELECT} GROUP BY d.id ORDER BY d.updated_at DESC").fetchall()
        return [deck_from_row(r) for r in rows]

    def get_deck(self, deck_id: str) -> dict[str, Any] | None:
        with self.db.conn() as c:
            row = c.execute(f"{_DECK_SELECT} WHERE d.id = ? GROUP BY d.id", (deck_id,)).fetchone()
        return deck_from_row(row) if row else None

    def create_deck(self, name: str, topic: str = "") -> dict[str, Any]:
        deck_id, ts = new_id(), now()
        with self.db.tx() as c:
            c.execute("INSERT INTO decks(id, name, topic, created_at, updated_at) VALUES (?,?,?,?,?)", (deck_id, name, topic, ts, ts))
        return self.get_deck(deck_id)  # type: ignore[return-value]

    def update_deck(self, deck_id: str, *, name: str | None = None, topic: str | None = None) -> dict[str, Any] | None:
        with self.db.tx() as c:
            cur = c.execute(
                "UPDATE decks SET name = COALESCE(?, name), topic = COALESCE(?, topic), updated_at = ? WHERE id = ?",
                (name, topic, now(), deck_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get_deck(deck_id)

    def delete_deck(self, deck_id: str) -> bool:
        with self.db.tx() as c:
            return c.execute("DELETE FROM decks WHERE id = ?", (deck_id,)).rowcount > 0

    # -- folders -------------------------------------------------------------------------

    def list_folders(self) -> list[dict[str, Any]]:
        with self.db.conn() as c:
            rows = c.execute(
                "SELECT f.id, f.name, f.created_at, f.updated_at, COUNT(d.id) AS deck_count"
                " FROM folders f LEFT JOIN decks d ON d.folder_id = f.id GROUP BY f.id ORDER BY lower(f.name), f.id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_folder(self, folder_id: str) -> dict[str, Any] | None:
        return next((f for f in self.list_folders() if f["id"] == folder_id), None)

    @staticmethod
    def _folder_name(c: sqlite3.Connection, name: str, own_id: str | None) -> str:
        """The trimmed name, or ValueError (a sentence for the user) if it is empty, too long or already used by another folder."""
        name = name.strip()
        if not name:
            raise ValueError("Give the folder a name.")
        if len(name) > MAX_FOLDER_NAME:
            raise ValueError(f"Use at most {MAX_FOLDER_NAME} characters for a folder name.")
        clash = c.execute("SELECT 1 FROM folders WHERE lower(name) = lower(?) AND id IS NOT ?", (name, own_id)).fetchone()
        if clash:
            raise ValueError("You already have a folder with that name.")
        return name

    def create_folder(self, name: str) -> dict[str, Any]:
        folder_id, ts = new_id(), now()
        with self.db.tx() as c:
            name = self._folder_name(c, name, None)
            c.execute("INSERT INTO folders(id, name, created_at, updated_at) VALUES (?,?,?,?)", (folder_id, name, ts, ts))
        return self.get_folder(folder_id)  # type: ignore[return-value]

    def rename_folder(self, folder_id: str, name: str) -> dict[str, Any] | None:
        with self.db.tx() as c:
            if c.execute("SELECT 1 FROM folders WHERE id = ?", (folder_id,)).fetchone() is None:
                return None
            name = self._folder_name(c, name, folder_id)
            c.execute("UPDATE folders SET name = ?, updated_at = ? WHERE id = ?", (name, now(), folder_id))
        return self.get_folder(folder_id)

    def delete_folder(self, folder_id: str) -> bool:
        """Remove a folder. Its decks are kept: they are taken out of the folder first (each deck's own change is recorded, so
        phones hear about it), never deleted along with it."""
        ts = now()
        with self.db.tx() as c:
            c.execute("UPDATE decks SET folder_id = NULL, updated_at = ? WHERE folder_id = ?", (ts, folder_id))
            return c.execute("DELETE FROM folders WHERE id = ?", (folder_id,)).rowcount > 0

    def set_deck_folder(self, deck_id: str, folder_id: str | None) -> dict[str, Any] | None:
        """Put a deck in a folder, or take it out with `None`. Returns the deck, or None when the deck does not exist;
        raises FolderNotFound when the folder does not."""
        with self.db.tx() as c:
            if c.execute("SELECT 1 FROM decks WHERE id = ?", (deck_id,)).fetchone() is None:
                return None
            if folder_id is not None and c.execute("SELECT 1 FROM folders WHERE id = ?", (folder_id,)).fetchone() is None:
                raise FolderNotFound(folder_id)
            c.execute("UPDATE decks SET folder_id = ?, updated_at = ? WHERE id = ?", (folder_id, now(), deck_id))
        return self.get_deck(deck_id)

    def merge_decks(self, into_id: str, from_id: str) -> tuple[dict[str, Any], int] | None:
        """Move every card of `from_id` into `into_id`, then remove `from_id`, in one transaction.

        Returns the destination deck and how many cards moved, or None when either deck does not exist. Cards keep their ids
        and their status, so review logs and schedules (which hang off the card id) are untouched; only the deck changes.
        Uploads that were headed for, or produced, the removed deck are pointed at the destination so nothing is orphaned."""
        if into_id == from_id:
            raise ValueError("A deck cannot be merged into itself")
        ts = now()
        with self.db.tx() as c:
            found = {r["id"] for r in c.execute("SELECT id FROM decks WHERE id IN (?, ?)", (into_id, from_id)).fetchall()}
            if found != {into_id, from_id}:
                return None
            moved = c.execute("UPDATE cards SET deck_id = ?, updated_at = ? WHERE deck_id = ?", (into_id, ts, from_id)).rowcount
            c.execute("UPDATE jobs SET deck_id = ? WHERE deck_id = ?", (into_id, from_id))
            c.execute("UPDATE jobs SET target_deck_id = ? WHERE target_deck_id = ?", (into_id, from_id))
            c.execute("UPDATE decks SET updated_at = ? WHERE id = ?", (ts, into_id))
            c.execute("DELETE FROM decks WHERE id = ?", (from_id,))
        return self.get_deck(into_id), moved  # type: ignore[return-value]

    # -- cards ---------------------------------------------------------------------------

    def list_cards(self, *, deck_id: str | None = None, job_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        for col, val in (("deck_id", deck_id), ("job_id", job_id), ("status", status)):
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        sql = "SELECT * FROM cards" + (f" WHERE {' AND '.join(where)}" if where else "") + " ORDER BY created_at, rowid"
        with self.db.conn() as c:
            return [card_from_row(r) for r in c.execute(sql, args).fetchall()]

    def get_card(self, card_id: str) -> dict[str, Any] | None:
        with self.db.conn() as c:
            row = c.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        return card_from_row(row) if row else None

    def update_card(self, card_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        """Edit front/back/tags/difficulty/status. Unknown keys are ignored; invalid values raise ValueError."""
        if fields.get("back") is not None and not str(fields["back"]).strip():
            existing = self.get_card(card_id)
            if existing is not None and existing["type"] != "cloze":  # only a fill-in-the-blank card may have no explanation
                raise ValueError("Write the back of the card.")
        sets, args = [], []
        for key in ("front", "back", "difficulty", "status"):
            if key in fields and fields[key] is not None:
                if key == "status" and fields[key] not in CARD_STATUSES:
                    raise ValueError(f"status must be one of {CARD_STATUSES}")
                if key == "difficulty" and fields[key] not in CARD_DIFFICULTIES:
                    raise ValueError(f"difficulty must be one of {CARD_DIFFICULTIES}")
                sets.append(f"{key} = ?")
                args.append(fields[key])
        if fields.get("tags") is not None:
            sets.append("tags_json = ?")
            args.append(json.dumps(fields["tags"]))
        if fields.get("choices") is not None:
            existing = self.get_card(card_id)
            if existing is None:
                return None
            # Checked against the right answer the card will have after this edit.
            correct = fields["back"] if fields.get("back") is not None else existing["back"]
            sets.append("choices_json = ?")
            args.append(json.dumps(clean_choices(fields["choices"], correct, existing["type"])))
        if not sets:
            return self.get_card(card_id)
        sets.append("updated_at = ?")
        args += [now(), card_id]
        with self.db.tx() as c:
            if c.execute(f"UPDATE cards SET {', '.join(sets)} WHERE id = ?", args).rowcount == 0:
                return None
        return self.get_card(card_id)

    def set_choices(self, items: Iterable[tuple[str, list[str]]]) -> int:
        """Save the wrong answers of several cards at once, all or nothing (so a reviewed deck is never half saved). Raises ValueError
        for a refused answer and LookupError for a card that does not exist; returns how many cards were saved."""
        saved = 0
        with self.db.tx() as c:
            for card_id, choices in items:
                row = c.execute("SELECT type, back FROM cards WHERE id = ?", (card_id,)).fetchone()
                if row is None:
                    raise LookupError(card_id)
                cleaned = clean_choices(choices, row["back"], row["type"])
                c.execute("UPDATE cards SET choices_json = ?, updated_at = ? WHERE id = ?", (json.dumps(cleaned), now(), card_id))
                saved += 1
        return saved

    def create_card(self, deck_id: str, *, type: str, front: str, back: str = "", choices: Iterable[str] = ()) -> dict[str, Any] | None:
        """A card the user typed in. It is approved straight away (there is nothing to review), has no job, and is marked
        `origin = "manual"`. Returns None for an unknown deck; raises ValueError (a sentence for the user) for bad text."""
        front, back = front.strip(), back.strip()
        if type not in MANUAL_CARD_TYPES:
            raise ValueError(f"type must be one of {MANUAL_CARD_TYPES}")
        if not front:
            raise ValueError("Write the front of the card.")
        if type == "cloze":
            if not _CLOZE_BLANK.search(front):
                raise ValueError("A fill-in-the-blank card needs a blank: wrap the answer like {{c1::answer}}.")
        elif not back:
            raise ValueError("Write the back of the card.")
        wrong = clean_choices(choices, back, type)
        card_id, ts = new_id(), now()
        with self.db.tx() as c:
            if c.execute("SELECT 1 FROM decks WHERE id = ?", (deck_id,)).fetchone() is None:
                return None
            c.execute(
                "INSERT INTO cards(id, deck_id, job_id, type, front, back, origin, source_ref, source_snippet, tags_json,"
                " difficulty, status, atomic, grounded, issue, duplicate_of, created_at, updated_at, choices_json)"
                " VALUES (?,?,NULL,?,?,?,'manual',?,'','[]','medium','approved',1,1,'',NULL,?,?,?)",
                (card_id, deck_id, type, front, back, MANUAL_SOURCE, ts, ts, json.dumps(wrong)),
            )
            c.execute("UPDATE decks SET updated_at = ? WHERE id = ?", (ts, deck_id))
        return self.get_card(card_id)

    def delete_card(self, card_id: str) -> bool:
        with self.db.tx() as c:
            return c.execute("DELETE FROM cards WHERE id = ?", (card_id,)).rowcount > 0

    def review(self, approve_ids: Iterable[str], reject_ids: Iterable[str]) -> int:
        """Bulk approve/reject (one round trip after swiping through a draft deck). Returns rows changed."""
        ts, changed = now(), 0
        with self.db.tx() as c:
            for status, ids in (("approved", list(approve_ids)), ("rejected", list(reject_ids))):
                for card_id in ids:
                    changed += c.execute("UPDATE cards SET status = ?, updated_at = ? WHERE id = ?", (status, ts, card_id)).rowcount
        return changed
