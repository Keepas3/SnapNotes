"""Incremental sync between the server library and a device.

Pull is cursor based: every deck, card and card_state row carries the value of a global change counter (`seq`,
stamped by triggers), and deletions leave tombstones. A device asks for "everything after cursor N".

Push is an append-only log of reviews. The device computes the new FSRS state when it records a review and
sends that state along; the server keeps the log and the state from the most recent review of each card.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.db import Database
from app.store.library import now

MIN_REVIEW_TIME = datetime(2020, 1, 1, tzinfo=timezone.utc)
MAX_CLOCK_SKEW = timedelta(days=1)  # a review "from the future" means a wrong device clock; refuse it


def parse_ts(value: str) -> datetime:
    """Accept ISO-8601 with a trailing Z or an offset; naive values are taken as UTC."""
    dt = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    """Fixed-width UTC timestamp (YYYY-MM-DDTHH:MM:SS.mmmZ), so string order equals time order."""
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _card_for_phone(row: Any) -> dict[str, Any]:
    """A card row as sent to a phone: the wrong answers stored as JSON become a list."""
    card = dict(row)
    card["choices"] = json.loads(card.pop("choices_json", None) or "[]")
    return card


@dataclass
class ReviewLogData:
    id: str
    card_id: str
    rating: int
    reviewed_at: str
    elapsed_ms: int
    state_before: int
    state_after: dict[str, Any]


@dataclass
class PushResult:
    accepted: int = 0
    duplicates: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (log id, reason)


@dataclass
class PullResult:
    cursor: int
    has_more: bool
    decks: list[dict[str, Any]]
    cards: list[dict[str, Any]]
    states: list[dict[str, Any]]
    deleted: list[dict[str, str]]
    folders: list[dict[str, Any]] = field(default_factory=list)


class SyncStore:
    def __init__(self, db: Database):
        self.db = db

    # -- database identity -----------------------------------------------------------------

    def epoch(self) -> str:
        """Random id created with the database. A device that last synced with a different epoch is talking to a
        different database (a rebuilt volume, a restore) and must start from scratch."""
        with self.db.conn() as c:
            return c.execute("SELECT value FROM sync_meta WHERE key = 'epoch'").fetchone()[0]

    def rotate_epoch(self) -> str:
        """Give the database a new identity, so every device re-syncs from zero and re-uploads its review history.
        Run this after restoring an older snapshot (see DEPLOY.md): the restored data has the old epoch."""
        new = uuid.uuid4().hex
        with self.db.tx() as c:
            c.execute("UPDATE sync_meta SET value = ? WHERE key = 'epoch'", (new,))
        return new

    def max_seq(self) -> int:
        with self.db.conn() as c:
            return c.execute("SELECT seq FROM sync_counter WHERE id = 1").fetchone()[0]

    def needs_reset(self, client_epoch: str | None, cursor: int) -> bool:
        """True when a device's saved position cannot belong to this database:
        - it was last told a different epoch, or
        - its cursor is ahead of anything this database has ever numbered (the server was rolled back)."""
        return (client_epoch is not None and client_epoch != self.epoch()) or cursor > self.max_seq()

    # -- push ----------------------------------------------------------------------------

    @staticmethod
    def _problem(log: ReviewLogData, current: datetime) -> str | None:
        if not 1 <= log.rating <= 4:
            return "bad_rating"
        try:
            reviewed = parse_ts(log.reviewed_at)
        except ValueError:
            return "bad_time"
        if reviewed < MIN_REVIEW_TIME or reviewed > current + MAX_CLOCK_SKEW:
            return "bad_time"
        return None

    def push_logs(self, device_id: str, logs: list[ReviewLogData]) -> PushResult:
        """Store reviews idempotently (by log id). A log only moves a card's state forward in time, so replaying
        an old log, or one from a slower device, never rewinds the schedule."""
        result, current, received = PushResult(), datetime.now(timezone.utc), now()
        with self.db.tx() as c:
            for log in logs:
                problem = self._problem(log, current)
                if problem is None and c.execute("SELECT 1 FROM cards WHERE id = ?", (log.card_id,)).fetchone() is None:
                    problem = "unknown_card"
                if problem:
                    result.rejected.append((log.id, problem))
                    continue

                reviewed = parse_ts(log.reviewed_at)
                state = dict(log.state_after)
                state["last_review"] = iso_z(reviewed)
                try:
                    state["due"] = iso_z(parse_ts(str(state["due"])))
                except (KeyError, ValueError):
                    result.rejected.append((log.id, "bad_state"))
                    continue

                inserted = c.execute(
                    "INSERT OR IGNORE INTO review_logs(id, card_id, rating, reviewed_at, elapsed_ms, state_before,"
                    " state_after_json, device_id, received_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (log.id, log.card_id, log.rating, iso_z(reviewed), log.elapsed_ms, log.state_before, json.dumps(state), device_id, received),
                ).rowcount
                if inserted == 0:
                    result.duplicates += 1
                    continue
                result.accepted += 1

                existing = c.execute("SELECT last_review FROM card_state WHERE card_id = ?", (log.card_id,)).fetchone()
                if existing is None or parse_ts(existing["last_review"]) < reviewed:
                    c.execute(
                        "INSERT INTO card_state(card_id, state_json, due, last_review) VALUES (?,?,?,?)"
                        " ON CONFLICT(card_id) DO UPDATE SET state_json = excluded.state_json, due = excluded.due,"
                        " last_review = excluded.last_review",
                        (log.card_id, json.dumps(state), state["due"], state["last_review"]),
                    )
        return result

    # -- pull ----------------------------------------------------------------------------

    def pull(self, cursor: int, limit: int) -> PullResult:
        """Everything changed after `cursor`, at most `limit` rows per kind.

        If any kind hits the limit the returned cursor stops at the earliest such boundary and `has_more` is set. Rows of
        every kind past that cursor are left for the next page, so changes always arrive in the order they happened. That
        matters for a merged deck: its cards are moved before it is deleted, and a device deletes a deck's cards when it is
        told the deck is gone, so the deletion must never arrive ahead of the moves."""
        with self.db.conn() as c:
            c.execute("BEGIN")  # one snapshot for all the reads below
            try:
                max_seq = c.execute("SELECT seq FROM sync_counter WHERE id = 1").fetchone()[0]
                folders = c.execute(
                    "SELECT id, name, updated_at, seq FROM folders WHERE seq > ? ORDER BY seq LIMIT ?", (cursor, limit)
                ).fetchall()
                decks = c.execute(
                    "SELECT id, name, topic, folder_id, updated_at, seq FROM decks WHERE seq > ? ORDER BY seq LIMIT ?", (cursor, limit)
                ).fetchall()
                cards = c.execute(
                    "SELECT id, deck_id, type, front, back, origin, source_ref, source_snippet, status, choices_json, created_at, updated_at, seq"
                    " FROM cards WHERE seq > ? ORDER BY seq LIMIT ?",
                    (cursor, limit),
                ).fetchall()
                changed_states = c.execute(
                    "SELECT card_id, state_json, last_review, seq FROM card_state WHERE seq > ? ORDER BY seq LIMIT ?", (cursor, limit)
                ).fetchall()
                tombs = c.execute("SELECT seq, kind, id FROM tombstones WHERE seq > ? ORDER BY seq LIMIT ?", (cursor, limit)).fetchall()

                # Only the rows selected by seq (not the schedules joined on to cards) decide where this page ends.
                limited = [rows[-1]["seq"] for rows in (folders, decks, cards, changed_states, tombs) if len(rows) >= limit]
                end = min(limited) if limited else max_seq
                folders, decks, cards, changed_states, tombs = (
                    [r for r in rows if r["seq"] <= end] for rows in (folders, decks, cards, changed_states, tombs)
                )

                # A card always travels with its schedule, so a device never studies it as "new" by mistake.
                states = list(changed_states)
                have = {s["card_id"] for s in states}
                missing = [card["id"] for card in cards if card["id"] not in have]
                for i in range(0, len(missing), 500):
                    chunk = missing[i : i + 500]
                    states += c.execute(
                        f"SELECT card_id, state_json, last_review, seq FROM card_state WHERE card_id IN ({','.join('?' * len(chunk))})", chunk
                    ).fetchall()
            finally:
                c.execute("COMMIT")

        return PullResult(
            cursor=end,
            has_more=bool(limited),
            folders=[dict(f) for f in folders],
            decks=[dict(d) for d in decks],
            cards=[_card_for_phone(card) for card in cards],
            states=[{"card_id": s["card_id"], "last_review": s["last_review"], "state": json.loads(s["state_json"])} for s in states],
            deleted=[{"kind": t["kind"], "id": t["id"]} for t in tombs],
        )
