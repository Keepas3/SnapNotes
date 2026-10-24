"""What happens when a phone's idea of the server no longer matches the server (restored backup, rebuilt volume)."""
import uuid
from datetime import datetime, timedelta, timezone

from app.admin import main as admin
from app.core.db import Database
from app.store.sync import SyncStore
from tests.test_api import H
from tests.test_sync import DEVICE, Env, log


def raw_sync(env: Env, **body):
    res = env.client.post("/sync", headers=H, json={"device_id": DEVICE, "cursor": 0, "logs": [], **body})
    assert res.status_code == 200, res.text
    return res.json()


def seeded(env: Env, n_cards=3):
    deck = env.deck()
    return deck, [env.card(deck, f"Q{i}") for i in range(n_cards)]


def test_every_response_names_the_database_and_a_normal_sync_is_not_a_reset(tmp_path):
    with Env(tmp_path) as env:
        seeded(env)
        first = raw_sync(env)  # a first sync sends no epoch
        assert first["epoch"] and first["reset"] is False

        again = raw_sync(env, cursor=first["cursor"], epoch=first["epoch"])
        assert again["reset"] is False and again["epoch"] == first["epoch"]
        assert again["cards"] == []  # nothing new, so this really was an incremental sync


def test_an_app_that_does_not_send_an_epoch_yet_keeps_working(tmp_path):
    with Env(tmp_path) as env:
        _, cards = seeded(env)
        first = raw_sync(env)
        second = raw_sync(env, cursor=first["cursor"])  # older app: cursor only
        assert second["reset"] is False and second["cards"] == []
        assert len(first["cards"]) == len(cards)


def test_a_different_epoch_means_a_different_database_and_forces_a_full_download(tmp_path):
    with Env(tmp_path) as env:
        deck, cards = seeded(env)
        caught_up = raw_sync(env)
        res = raw_sync(env, cursor=caught_up["cursor"], epoch="the-epoch-of-some-other-database")

        assert res["reset"] is True and res["epoch"] == caught_up["epoch"]
        # Even though the phone claimed to be fully caught up, it receives everything again.
        assert [d["id"] for d in res["decks"]] == [deck]
        assert sorted(c["id"] for c in res["cards"]) == sorted(cards)


def test_a_cursor_ahead_of_the_database_means_the_server_was_rolled_back(tmp_path):
    with Env(tmp_path) as env:
        _, cards = seeded(env)
        epoch = raw_sync(env)["epoch"]
        res = raw_sync(env, cursor=10_000, epoch=epoch)  # same epoch, but a position this database never reached

        assert res["reset"] is True
        assert sorted(c["id"] for c in res["cards"]) == sorted(cards)
        assert 0 < res["cursor"] < 10_000  # the phone's cursor is pulled back to a real position


def test_a_reset_is_reported_only_once_the_phone_has_adopted_the_new_position(tmp_path):
    with Env(tmp_path) as env:
        seeded(env)
        reset = raw_sync(env, cursor=999, epoch="stale")
        assert reset["reset"] is True
        follow_up = raw_sync(env, cursor=reset["cursor"], epoch=reset["epoch"])
        assert follow_up["reset"] is False and follow_up["cards"] == []


def test_a_reset_request_still_stores_the_reviews_it_carries(tmp_path):
    with Env(tmp_path) as env:
        _, cards = seeded(env)
        entry = log(cards[0], datetime.now(timezone.utc) - timedelta(minutes=1))
        res = raw_sync(env, cursor=999, epoch="stale", logs=[entry])
        assert res["reset"] is True and res["accepted"] == 1
        assert [s["card_id"] for s in res["states"]] == [cards[0]]  # and the schedule comes straight back


def test_a_paginated_reset_download_covers_everything_and_flags_only_the_first_page(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        cards = [env.card(deck, f"Q{i}") for i in range(7)]
        page = raw_sync(env, cursor=999, epoch="stale", limit=2)
        assert page["reset"] is True and page["has_more"] is True

        seen, epoch, cursor, pages = {c["id"] for c in page["cards"]}, page["epoch"], page["cursor"], 1
        while page["has_more"]:
            page = raw_sync(env, cursor=cursor, epoch=epoch, limit=2)
            assert page["reset"] is False
            seen |= {c["id"] for c in page["cards"]}
            cursor, pages = page["cursor"], pages + 1
            assert pages < 30
        assert seen == set(cards)


def test_the_operator_command_makes_every_phone_start_over(tmp_path, monkeypatch, capsys):
    """The documented restore procedure: after restoring a snapshot, rotate the epoch."""
    monkeypatch.setenv("SNAPNOTES_DATA_DIR", str(tmp_path))
    with Env(tmp_path) as env:
        seeded(env)
        before = raw_sync(env)
        assert raw_sync(env, cursor=before["cursor"], epoch=before["epoch"])["reset"] is False

        assert admin(["show-epoch"]) == 0
        assert capsys.readouterr().out.strip() == before["epoch"]
        assert admin(["rotate-epoch"]) == 0
        out = capsys.readouterr().out
        assert before["epoch"] in out

        after = raw_sync(env, cursor=before["cursor"], epoch=before["epoch"])
        assert after["reset"] is True and after["epoch"] != before["epoch"] and out.count(after["epoch"]) == 1
        assert raw_sync(env, cursor=after["cursor"], epoch=after["epoch"])["reset"] is False


def test_the_epoch_is_stable_across_restarts_and_unique_per_database(tmp_path):
    one, two = Database(tmp_path / "one.db"), Database(tmp_path / "two.db")
    for db in (one, two):
        db.init()
    first = SyncStore(one).epoch()
    one.init()  # a restart
    assert SyncStore(one).epoch() == first
    assert SyncStore(two).epoch() != first


def test_a_version_2_database_gains_an_epoch_and_keeps_its_data(tmp_path):
    db = Database(tmp_path / "v2.db")
    db.init()
    deck_id = uuid.uuid4().hex
    with db.tx() as c:
        c.execute("INSERT INTO decks(id, name, topic, created_at, updated_at) VALUES (?, 'Kept', '', 't', 't')", (deck_id,))
        c.execute("DROP TABLE sync_meta")
        c.execute("PRAGMA user_version = 2")  # what the previous release left behind

    db.init()
    with db.conn() as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == 4
        assert c.execute("SELECT name FROM decks WHERE id = ?", (deck_id,)).fetchone()[0] == "Kept"
    assert len(SyncStore(db).epoch()) == 32
