import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.db import Database
from app.main import create_app
from app.store.library import now as ts_now
from tests.test_api import H, TOKEN, upload, wait_for

DEVICE = "phone-1"


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class Env:
    """A running app plus helpers that seed decks and cards straight into the database."""

    def __init__(self, tmp_path, **overrides):
        self.app = create_app(Settings(data_dir=tmp_path, api_token=TOKEN, demo_mode=True, demo_delay=0, **overrides))
        self.client = TestClient(self.app)
        self.db: Database = self.app.state.library.db

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, *exc):
        return self.client.__exit__(*exc)

    def deck(self, name="Biology") -> str:
        return self.client.post("/decks", headers=H, json={"name": name}).json()["id"]

    def card(self, deck_id: str, front="Q?", status="approved") -> str:
        card_id, ts = uuid.uuid4().hex, ts_now()
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO cards(id, deck_id, job_id, type, front, back, origin, source_ref, source_snippet, tags_json,"
                " difficulty, status, atomic, grounded, issue, duplicate_of, created_at, updated_at)"
                " VALUES (?,?,NULL,'basic',?,'A','notes','p.1','snip','[]','easy',?,1,1,'',NULL,?,?)",
                (card_id, deck_id, front, status, ts, ts),
            )
        return card_id

    def sync(self, cursor=0, logs=None, limit=500, device=DEVICE):
        res = self.client.post("/sync", headers=H, json={"device_id": device, "cursor": cursor, "logs": logs or [], "limit": limit})
        assert res.status_code == 200, res.text
        return res.json()


def state(due: datetime, reps=1, fsrs_state=2) -> dict:
    return {"due": iso(due), "stability": 4.2, "difficulty": 5.1, "elapsed_days": 0, "scheduled_days": 4,
            "learning_steps": 0, "reps": reps, "lapses": 0, "state": fsrs_state, "last_review": None}


def log(card_id: str, reviewed: datetime, rating=3, log_id=None, due_in_days=4, **extra) -> dict:
    return {"id": log_id or uuid.uuid4().hex, "card_id": card_id, "rating": rating, "reviewed_at": iso(reviewed),
            "elapsed_ms": 2500, "state_before": 0, "state_after": state(reviewed + timedelta(days=due_in_days)), **extra}


NOW = datetime.now(timezone.utc)


# --- pull ----------------------------------------------------------------------------------


def test_first_sync_returns_everything_then_nothing(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        cards = [env.card(deck, f"Q{i}") for i in range(3)]

        first = env.sync()
        assert [d["name"] for d in first["decks"]] == ["Biology"]
        assert sorted(c["id"] for c in first["cards"]) == sorted(cards)
        assert first["has_more"] is False and first["cursor"] > 0
        assert {"front", "back", "origin", "source_ref", "status"} <= set(first["cards"][0])

        second = env.sync(cursor=first["cursor"])
        assert second["decks"] == second["cards"] == second["states"] == second["deleted"] == []
        assert second["cursor"] == first["cursor"]


def test_changes_after_the_cursor_are_delivered_once(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        a, b = env.card(deck, "A"), env.card(deck, "B")
        cursor = env.sync()["cursor"]

        env.client.patch(f"/cards/{a}", headers=H, json={"front": "A edited"})
        env.client.patch(f"/decks/{deck}", headers=H, json={"name": "Renamed"})
        changed = env.sync(cursor=cursor)
        assert [c["id"] for c in changed["cards"]] == [a] and changed["cards"][0]["front"] == "A edited"
        assert [d["name"] for d in changed["decks"]] == ["Renamed"]
        assert env.sync(cursor=changed["cursor"])["cards"] == []


def test_status_changes_are_delivered_so_devices_can_drop_unapproved_cards(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        card = env.card(deck, status="draft")
        cursor = env.sync()["cursor"]
        env.client.post("/cards/review", headers=H, json={"approve_ids": [card]})
        approved = env.sync(cursor=cursor)
        assert [(c["id"], c["status"]) for c in approved["cards"]] == [(card, "approved")]
        cursor = approved["cursor"]
        env.client.patch(f"/cards/{card}", headers=H, json={"status": "rejected"})
        assert [(c["id"], c["status"]) for c in env.sync(cursor=cursor)["cards"]] == [(card, "rejected")]


def test_deletes_leave_tombstones_including_cascades(tmp_path):
    with Env(tmp_path) as env:
        deck, other = env.deck("Keep"), env.deck("Drop")
        keep_card, gone, cascaded = env.card(deck), env.card(deck), env.card(other)
        cursor = env.sync()["cursor"]

        env.client.delete(f"/cards/{gone}", headers=H)
        env.client.delete(f"/decks/{other}", headers=H)  # also removes `cascaded`
        deleted = {(d["kind"], d["id"]) for d in env.sync(cursor=cursor)["deleted"]}
        assert deleted == {("card", gone), ("card", cascaded), ("deck", other)}
        assert keep_card not in {i for _, i in deleted}


def test_deleting_a_document_tombstones_its_unapproved_cards(tmp_path):
    with Env(tmp_path, delete_originals=False) as env:
        body = upload(env.client, auto_approve="true").json()
        wait_for(env.client, body["job_id"], "completed")
        cards = env.client.get(f"/jobs/{body['job_id']}/cards", headers=H).json()
        env.client.post("/cards/review", headers=H, json={"approve_ids": [cards[0]["id"]]})
        cursor = env.sync()["cursor"]

        assert env.client.delete(f"/documents/{body['document_id']}", headers=H).status_code == 204
        after = env.sync(cursor=cursor)
        removed = {d["id"] for d in after["deleted"] if d["kind"] == "card"}
        assert removed == {c["id"] for c in cards} - {cards[0]["id"]}
        # the approved card survives (its job link was cleared, which shows up as a plain update)
        assert cards[0]["id"] in {c["id"] for c in after["cards"]}


def test_pages_cover_everything_with_a_monotonic_cursor(tmp_path):
    with Env(tmp_path) as env:
        decks = [env.deck(f"D{i}") for i in range(3)]
        cards = [env.card(decks[i % 3], f"Q{i}") for i in range(7)]
        env.sync(logs=[log(cards[0], NOW), log(cards[1], NOW)])
        # also create some tombstones so every kind has to paginate
        extra = [env.card(decks[0], f"X{i}") for i in range(3)]
        for card in extra:
            env.client.delete(f"/cards/{card}", headers=H)

        seen_cards, seen_decks, seen_states, seen_deleted, cursor, pages = set(), set(), set(), set(), 0, 0
        while True:
            page = env.sync(cursor=cursor, limit=2)
            assert page["cursor"] >= cursor
            cursor, pages = page["cursor"], pages + 1
            seen_cards |= {c["id"] for c in page["cards"]}
            seen_decks |= {d["id"] for d in page["decks"]}
            seen_states |= {s["card_id"] for s in page["states"]}
            seen_deleted |= {d["id"] for d in page["deleted"]}
            if not page["has_more"]:
                break
            assert pages < 50
        assert set(cards) <= seen_cards and set(decks) <= seen_decks
        assert {cards[0], cards[1]} <= seen_states
        assert set(extra) <= seen_deleted
        assert pages > 1


def test_a_card_always_arrives_with_its_schedule(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        card = env.card(deck)
        env.sync(logs=[log(card, NOW)])
        # a client that has never seen the card gets its state in the same response, even with a cursor past the state
        top = env.sync()["cursor"]
        env.client.patch(f"/cards/{card}", headers=H, json={"front": "edited"})
        res = env.sync(cursor=top)
        assert [c["id"] for c in res["cards"]] == [card]
        assert [s["card_id"] for s in res["states"]] == [card]


# --- push ----------------------------------------------------------------------------------


def test_pushed_reviews_are_idempotent_and_set_the_card_state(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        card = env.card(deck)
        entry = log(card, NOW - timedelta(minutes=5), rating=4, due_in_days=10)

        first = env.sync(logs=[entry])
        assert (first["accepted"], first["duplicates"], first["rejected"]) == (1, 0, [])
        assert first["states"][0]["card_id"] == card and first["states"][0]["state"]["scheduled_days"] == 4

        again = env.sync(logs=[entry], cursor=first["cursor"])  # a retried push after a lost response
        assert (again["accepted"], again["duplicates"]) == (0, 1)
        with env.db.conn() as c:
            assert c.execute("SELECT COUNT(*) FROM review_logs").fetchone()[0] == 1


def test_an_older_review_never_rewinds_the_schedule(tmp_path):
    with Env(tmp_path) as env:
        card = env.card(env.deck())
        newer = log(card, NOW - timedelta(minutes=1), rating=3, due_in_days=9)
        older = log(card, NOW - timedelta(hours=3), rating=1, due_in_days=1)
        env.sync(logs=[newer])
        res = env.sync(logs=[older])  # e.g. a second device that was offline
        assert res["accepted"] == 1  # the review itself is kept...
        with env.db.conn() as c:
            row = c.execute("SELECT state_json, last_review FROM card_state WHERE card_id = ?", (card,)).fetchone()
            assert c.execute("SELECT COUNT(*) FROM review_logs").fetchone()[0] == 2
        assert json.loads(row["state_json"])["scheduled_days"] == 4 and row["last_review"] == newer["reviewed_at"]
        # ...and the stored due date is the newer review's (9 days out), not the older one's
        assert json.loads(row["state_json"])["due"] == newer["state_after"]["due"]


def test_state_timestamps_are_normalised(tmp_path):
    with Env(tmp_path) as env:
        card = env.card(env.deck())
        entry = log(card, NOW)
        entry["reviewed_at"] = NOW.astimezone(timezone(timedelta(hours=-7))).isoformat()  # offset form, microseconds
        entry["state_after"]["due"] = (NOW + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        res = env.sync(logs=[entry])
        st = res["states"][0]["state"]
        assert st["last_review"] == res["states"][0]["last_review"]
        assert st["last_review"].endswith("Z") and len(st["last_review"]) == 24
        assert st["due"].endswith("Z") and len(st["due"]) == 24


def test_bad_logs_are_rejected_individually_and_the_rest_applied(tmp_path):
    with Env(tmp_path) as env:
        card = env.card(env.deck())
        good = log(card, NOW)
        bad_rating = log(card, NOW, rating=9)
        unknown = log("no-such-card", NOW)
        future = log(card, NOW + timedelta(days=30))
        garbage_time = log(card, NOW)
        garbage_time["reviewed_at"] = "yesterday-ish"
        bad_state = log(card, NOW)
        bad_state["state_after"]["due"] = "soon"

        res = env.sync(logs=[bad_rating, unknown, future, garbage_time, bad_state, good])
        reasons = {r["id"]: r["reason"] for r in res["rejected"]}
        assert reasons == {
            bad_rating["id"]: "bad_rating", unknown["id"]: "unknown_card", future["id"]: "bad_time",
            garbage_time["id"]: "bad_time", bad_state["id"]: "bad_state",
        }
        assert res["accepted"] == 1


def test_odd_cosmetic_values_are_clamped_not_fatal(tmp_path):
    with Env(tmp_path) as env:
        card = env.card(env.deck())
        entry = log(card, NOW, elapsed_ms=-5, state_before=99)
        assert env.sync(logs=[entry])["accepted"] == 1
        with env.db.conn() as c:
            row = c.execute("SELECT elapsed_ms, state_before FROM review_logs").fetchone()
        assert (row["elapsed_ms"], row["state_before"]) == (0, 3)


def test_request_size_and_shape_limits(tmp_path):
    with Env(tmp_path) as env:
        card = env.card(env.deck())
        too_many = [log(card, NOW) for _ in range(501)]
        assert env.client.post("/sync", headers=H, json={"device_id": DEVICE, "logs": too_many}).status_code == 422
        assert env.client.post("/sync", headers=H, json={"cursor": 0}).status_code == 422  # no device id
        assert env.client.post("/sync", headers=H, json={"device_id": DEVICE, "cursor": -1}).status_code == 422
        assert env.client.post("/sync", json={"device_id": DEVICE}).status_code == 401


def test_deleting_a_card_drops_its_logs_and_state(tmp_path):
    with Env(tmp_path) as env:
        card = env.card(env.deck())
        env.sync(logs=[log(card, NOW)])
        env.client.delete(f"/cards/{card}", headers=H)
        with env.db.conn() as c:
            assert c.execute("SELECT COUNT(*) FROM review_logs").fetchone()[0] == 0
            assert c.execute("SELECT COUNT(*) FROM card_state").fetchone()[0] == 0
        late = env.sync(logs=[log(card, NOW)])  # a device that was offline still has a pending review for it
        assert [r["reason"] for r in late["rejected"]] == ["unknown_card"]


# --- schema migration ----------------------------------------------------------------------


def test_databases_from_before_sync_are_upgraded_in_place(tmp_path):
    path = tmp_path / "old.db"
    # Build a version-1 database (no seq columns) with data in it, exactly as the previous milestone left it.
    from app.core.db import SCHEMA

    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO decks(id, name, topic, created_at, updated_at) VALUES ('d1','Old deck','','t','t')")
    for i in range(3):
        conn.execute(
            "INSERT INTO cards(id, deck_id, type, front, back, origin, source_ref, source_snippet, tags_json, difficulty,"
            " status, atomic, grounded, created_at, updated_at) VALUES (?, 'd1','basic','f','b','notes','','','[]','easy','approved',1,1,'t','t')",
            (f"c{i}",),
        )
    conn.commit()
    conn.close()

    db = Database(path)
    db.init()
    db.init()  # running it again is a no-op

    with db.conn() as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == 3
        seqs = [r[0] for r in c.execute("SELECT seq FROM decks UNION ALL SELECT seq FROM cards ORDER BY 1")]
        assert seqs == [1, 2, 3, 4]  # distinct, so a first sync can page through them
        assert c.execute("SELECT seq FROM sync_counter").fetchone()[0] == 4
        c.execute("UPDATE cards SET front = 'edited' WHERE id = 'c0'")  # triggers are live
        assert c.execute("SELECT seq FROM cards WHERE id = 'c0'").fetchone()[0] == 5


def test_every_write_gets_a_fresh_seq_and_never_loops(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        card = env.card(deck)
        with env.db.conn() as c:
            before = c.execute("SELECT seq FROM sync_counter").fetchone()[0]
            c.execute("UPDATE cards SET front = 'x' WHERE id = ?", (card,))
            after = c.execute("SELECT seq FROM sync_counter").fetchone()[0]
            assert after == before + 1  # exactly one bump per statement: the trigger does not retrigger itself
            assert c.execute("SELECT seq FROM cards WHERE id = ?", (card,)).fetchone()[0] == after
