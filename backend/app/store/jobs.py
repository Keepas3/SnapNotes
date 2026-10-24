"""Documents, jobs and the web evidence awaiting approval. All state transitions live here so the API
and the worker cannot disagree about what is allowed."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

from app.core.db import Database
from app.llm.client import UsageRow
from app.llm.schemas import EvidenceItem, ReviewedCard, UnderstandResult
from app.store.library import new_id, now


class NotFound(LookupError):
    pass


class InvalidState(RuntimeError):
    pass


_JOB_SELECT = "SELECT j.*, d.filename, d.kind FROM jobs j JOIN documents d ON d.id = j.document_id"


def _job(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["auto_approve"] = bool(d["auto_approve"])
    raw = d.pop("understanding_json")
    d["understanding"] = json.loads(raw) if raw else None
    d["usage"] = json.loads(d.pop("usage_json"))
    return d


class JobStore:
    def __init__(self, db: Database):
        self.db = db

    # -- creation and lookup -------------------------------------------------------------

    def create(self, *, doc_id: str, filename: str, kind: str, stored_names: list[str], auto_approve: bool) -> str:
        job_id, ts = new_id(), now()
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO documents(id, filename, kind, stored_names_json, created_at) VALUES (?,?,?,?,?)",
                (doc_id, filename, kind, json.dumps(stored_names), ts),
            )
            c.execute(
                "INSERT INTO jobs(id, document_id, status, auto_approve, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (job_id, doc_id, "queued", int(auto_approve), ts, ts),
            )
        return job_id

    def count_since(self, iso_ts: str) -> int:
        with self.db.conn() as c:
            return c.execute("SELECT COUNT(*) FROM jobs WHERE created_at >= ?", (iso_ts,)).fetchone()[0]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.db.conn() as c:
            row = c.execute(f"{_JOB_SELECT} WHERE j.id = ?", (job_id,)).fetchone()
        return _job(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.conn() as c:
            rows = c.execute(f"{_JOB_SELECT} ORDER BY j.created_at DESC, j.rowid DESC LIMIT ?", (limit,)).fetchall()
        return [_job(r) for r in rows]

    def document(self, doc_id: str) -> dict[str, Any] | None:
        with self.db.conn() as c:
            row = c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["stored_names"] = json.loads(d.pop("stored_names_json"))
        return d

    # -- worker transitions --------------------------------------------------------------

    def claim_next(self) -> dict[str, Any] | None:
        """Atomically move the oldest runnable job into its running state."""
        with self.db.tx() as c:
            row = c.execute("SELECT id, status FROM jobs WHERE status IN ('queued','approved') ORDER BY created_at, rowid LIMIT 1").fetchone()
            if row is None:
                return None
            running, phase = ("researching", "research") if row["status"] == "queued" else ("generating", "generate")
            c.execute("UPDATE jobs SET status = ?, phase = ?, stage = NULL, error = NULL, updated_at = ? WHERE id = ?", (running, phase, now(), row["id"]))
        return self.get(row["id"])

    def set_stage(self, job_id: str, stage: str) -> None:
        with self.db.conn() as c:
            c.execute("UPDATE jobs SET stage = ?, updated_at = ? WHERE id = ?", (stage, now(), job_id))

    def save_research(self, job_id: str, understanding: UnderstandResult, evidence: list[EvidenceItem], note: str = "") -> str:
        """Store phase-1 output and decide the next status. Returns it.

        `note` says why a wanted web search gave nothing; it is kept inside the understanding JSON (no new column) and shown in the app."""
        stored = json.loads(understanding.model_dump_json())
        if note:
            stored["research_note"] = note
        with self.db.tx() as c:
            auto = c.execute("SELECT auto_approve FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if auto is None:
                raise NotFound(job_id)
            c.execute("DELETE FROM evidence WHERE job_id = ?", (job_id,))
            c.executemany(
                "INSERT INTO evidence(job_id, claim, snippet, url, title, approved) VALUES (?,?,?,?,?,1)",
                [(job_id, e.claim, e.snippet, e.url, e.title) for e in evidence],
            )
            # Nothing to approve (or the user opted out of the approval step): go straight to generation.
            status = "approved" if (auto["auto_approve"] or not evidence) else "awaiting_approval"
            c.execute(
                "UPDATE jobs SET status = ?, stage = NULL, understanding_json = ?, updated_at = ? WHERE id = ?",
                (status, json.dumps(stored), now(), job_id),
            )
        return status

    def approved_evidence(self, job_id: str) -> list[EvidenceItem]:
        with self.db.conn() as c:
            rows = c.execute("SELECT * FROM evidence WHERE job_id = ? AND approved = 1 ORDER BY id", (job_id,)).fetchall()
        return [EvidenceItem(claim=r["claim"], snippet=r["snippet"], url=r["url"], title=r["title"]) for r in rows]

    def suspect_claims(self, job_id: str) -> list[str]:
        """Statements in the notes that the fact check flagged in phase one (empty when it was off, or for jobs from before it existed)."""
        with self.db.conn() as c:
            row = c.execute("SELECT understanding_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or not row["understanding_json"]:
            return []
        return list(json.loads(row["understanding_json"]).get("suspect_claims") or [])

    def save_generation(self, job_id: str, reviewed: list[ReviewedCard]) -> str:
        """Create the deck and its cards and complete the job in one transaction. Returns the deck id."""
        deck_id, ts = new_id(), now()
        card_ids = [new_id() for _ in reviewed]
        with self.db.tx() as c:
            row = c.execute("SELECT understanding_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFound(job_id)
            understanding = json.loads(row["understanding_json"]) if row["understanding_json"] else {}
            name = (understanding.get("main_topic") or "New deck")[:80]
            c.execute(
                "INSERT INTO decks(id, name, topic, created_at, updated_at) VALUES (?,?,?,?,?)",
                (deck_id, name, understanding.get("subject_area", ""), ts, ts),
            )
            for cid, rc in zip(card_ids, reviewed):
                card = rc.card
                dup = card_ids[rc.duplicate_of] if rc.duplicate_of is not None else None
                c.execute(
                    "INSERT INTO cards(id, deck_id, job_id, type, front, back, origin, source_ref, source_snippet, tags_json,"
                    " difficulty, status, atomic, grounded, issue, duplicate_of, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cid, deck_id, job_id, card.type, card.front, card.back, card.origin, card.source_ref,
                        card.source_snippet, json.dumps(card.tags), card.difficulty,
                        # Ungrounded cards and duplicates start out rejected; the client can still show them as "filtered".
                        "draft" if rc.keep else "rejected",
                        int(rc.atomic), int(rc.grounded), rc.issue, dup, ts, ts,
                    ),
                )
            c.execute("UPDATE jobs SET status = 'completed', stage = NULL, deck_id = ?, error = NULL, updated_at = ? WHERE id = ?", (deck_id, ts, job_id))
        return deck_id

    def add_usage(self, job_id: str, rows: Iterable[UsageRow]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.db.tx() as c:
            cur = c.execute("SELECT usage_json, cost_usd FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if cur is None:  # job deleted while running
                return
            usage = json.loads(cur["usage_json"]) + [r.__dict__ | {"cost": r.cost} for r in rows]
            cost = cur["cost_usd"] + sum(r.cost or 0.0 for r in rows)
            c.execute("UPDATE jobs SET usage_json = ?, cost_usd = ?, updated_at = ? WHERE id = ?", (json.dumps(usage), cost, now(), job_id))

    def fail(self, job_id: str, message: str) -> None:
        # `stage` is kept on purpose: clients use it to show which step failed.
        with self.db.conn() as c:
            c.execute("UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE id = ?", (message, now(), job_id))

    # -- user-driven transitions ---------------------------------------------------------

    def evidence(self, job_id: str) -> list[dict[str, Any]]:
        with self.db.conn() as c:
            rows = c.execute("SELECT * FROM evidence WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
        return [dict(r) | {"approved": bool(r["approved"])} for r in rows]

    def apply_approval(self, job_id: str, rejected_ids: list[int]) -> dict[str, int]:
        with self.db.tx() as c:
            job = c.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise NotFound(job_id)
            if job["status"] != "awaiting_approval":
                raise InvalidState(f"Job is {job['status']}, not awaiting_approval")
            known = {r["id"] for r in c.execute("SELECT id FROM evidence WHERE job_id = ?", (job_id,)).fetchall()}
            unknown = set(rejected_ids) - known
            if unknown:
                raise ValueError(f"Unknown evidence ids: {sorted(unknown)}")
            c.execute("UPDATE evidence SET approved = 1 WHERE job_id = ?", (job_id,))
            c.executemany("UPDATE evidence SET approved = 0 WHERE id = ? AND job_id = ?", [(i, job_id) for i in set(rejected_ids)])
            c.execute("UPDATE jobs SET status = 'approved', updated_at = ? WHERE id = ?", (now(), job_id))
        return {"approved": len(known) - len(set(rejected_ids)), "rejected": len(set(rejected_ids))}

    def retry(self, job_id: str) -> str:
        """Re-queue a failed job at the phase that failed. Returns the new status."""
        with self.db.tx() as c:
            job = c.execute("SELECT status, phase FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise NotFound(job_id)
            if job["status"] != "failed":
                raise InvalidState(f"Job is {job['status']}, only failed jobs can be retried")
            if job["phase"] == "generate":
                status = "approved"  # keeps the evidence approvals already made
            else:
                status = "queued"
                c.execute("DELETE FROM evidence WHERE job_id = ?", (job_id,))
                c.execute("UPDATE jobs SET understanding_json = NULL WHERE id = ?", (job_id,))
            c.execute("UPDATE jobs SET status = ?, error = NULL, stage = NULL, updated_at = ? WHERE id = ?", (status, now(), job_id))
        return status

    def recover_interrupted(self) -> int:
        """After a restart, jobs left mid-run go back to the start of their phase (each phase saves atomically)."""
        with self.db.tx() as c:
            a = c.execute("UPDATE jobs SET status = 'queued', stage = NULL WHERE status = 'researching'").rowcount
            b = c.execute("UPDATE jobs SET status = 'approved', stage = NULL WHERE status = 'generating'").rowcount
        return a + b

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document, its jobs and evidence, and every card that was never approved.
        Approved cards stay in their deck (their job link is cleared)."""
        with self.db.tx() as c:
            if c.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone() is None:
                return False
            c.execute(
                "DELETE FROM cards WHERE status != 'approved' AND job_id IN (SELECT id FROM jobs WHERE document_id = ?)",
                (doc_id,),
            )
            c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return True
