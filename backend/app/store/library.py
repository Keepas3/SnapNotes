"""Decks and cards: the persistent flashcard library."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from app.core.db import Database

CARD_STATUSES = ("draft", "approved", "rejected")
CARD_DIFFICULTIES = ("easy", "medium", "hard")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def card_from_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
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
        if not sets:
            return self.get_card(card_id)
        sets.append("updated_at = ?")
        args += [now(), card_id]
        with self.db.tx() as c:
            if c.execute(f"UPDATE cards SET {', '.join(sets)} WHERE id = ?", args).rowcount == 0:
                return None
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
