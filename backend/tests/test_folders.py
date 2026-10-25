"""Folders that group decks: the API, what happens to decks when a folder goes, and how phones hear about all of it."""
import sqlite3

from app.core.db import SCHEMA, SYNC_MIGRATION_STATEMENTS, SYNC_TRIGGER_STATEMENTS, Database
from tests.test_api import H
from tests.test_sync import Env


def folder(env, name="Biology") -> dict:
    res = env.client.post("/folders", headers=H, json={"name": name})
    assert res.status_code == 201, res.text
    return res.json()


def file_deck(env, deck_id, folder_id):
    return env.client.patch(f"/decks/{deck_id}", headers=H, json={"folder_id": folder_id})


# --- the API -------------------------------------------------------------------------------


def test_folders_can_be_made_listed_renamed_and_removed(tmp_path):
    with Env(tmp_path) as env:
        bio, prog = folder(env, "Biology"), folder(env, "programming")
        assert (bio["name"], bio["deck_count"]) == ("Biology", 0)
        assert [f["name"] for f in env.client.get("/folders", headers=H).json()] == ["Biology", "programming"]  # alphabetical, ignoring case

        renamed = env.client.patch(f"/folders/{prog['id']}", headers=H, json={"name": "  Code  "})
        assert renamed.status_code == 200 and renamed.json()["name"] == "Code"
        assert env.client.delete(f"/folders/{prog['id']}", headers=H).status_code == 204
        assert [f["name"] for f in env.client.get("/folders", headers=H).json()] == ["Biology"]


def test_folder_names_are_checked_in_plain_words(tmp_path):
    with Env(tmp_path) as env:
        bio = folder(env, "Biology")
        other = folder(env, "Chemistry")

        def post(name):
            return env.client.post("/folders", headers=H, json={"name": name})

        assert post("   ").status_code == 422 and "name" in post("   ").json()["detail"]
        assert post("x" * 81).status_code == 422 and "80" in post("x" * 81).json()["detail"]
        assert post("BIOLOGY").status_code == 409  # the same name in another case is still the same name
        assert env.client.patch(f"/folders/{other['id']}", headers=H, json={"name": "biology"}).status_code == 409
        assert env.client.patch(f"/folders/{bio['id']}", headers=H, json={"name": "BIOLOGY"}).status_code == 200  # but its own name is fine
        assert env.client.patch("/folders/nope", headers=H, json={"name": "x"}).status_code == 404
        assert env.client.delete("/folders/nope", headers=H).status_code == 404


def test_a_deck_can_be_filed_moved_and_taken_out(tmp_path):
    with Env(tmp_path) as env:
        deck, one, two = env.deck("Bio I"), folder(env, "One"), folder(env, "Two")
        assert env.client.get(f"/decks/{deck}", headers=H).json()["folder_id"] is None

        assert file_deck(env, deck, one["id"]).json()["folder_id"] == one["id"]
        assert file_deck(env, deck, two["id"]).json()["folder_id"] == two["id"]  # moving it to another folder replaces the first
        counts = {f["name"]: f["deck_count"] for f in env.client.get("/folders", headers=H).json()}
        assert counts == {"One": 0, "Two": 1}
        assert file_deck(env, deck, None).json()["folder_id"] is None  # an explicit null takes it out

        # leaving folder_id out of an edit changes nothing about it
        file_deck(env, deck, one["id"])
        renamed = env.client.patch(f"/decks/{deck}", headers=H, json={"name": "Bio 1"}).json()
        assert (renamed["name"], renamed["folder_id"]) == ("Bio 1", one["id"])
        assert env.client.get("/decks", headers=H).json()[0]["folder_id"] == one["id"]


def test_filing_into_a_missing_folder_or_a_missing_deck_is_a_404_and_changes_nothing(tmp_path):
    with Env(tmp_path) as env:
        deck, home = env.deck(), folder(env)
        file_deck(env, deck, home["id"])
        assert file_deck(env, deck, "no-such-folder").status_code == 404
        assert file_deck(env, "no-such-deck", home["id"]).status_code == 404
        assert env.client.get(f"/decks/{deck}", headers=H).json()["folder_id"] == home["id"]


def test_deleting_a_folder_keeps_its_decks_and_cards(tmp_path):
    with Env(tmp_path) as env:
        deck, home = env.deck("Bio I"), folder(env)
        env.card(deck, "Q1")
        file_deck(env, deck, home["id"])

        assert env.client.delete(f"/folders/{home['id']}", headers=H).status_code == 204
        kept = env.client.get(f"/decks/{deck}", headers=H).json()
        assert (kept["name"], kept["folder_id"], kept["counts"]["approved"]) == ("Bio I", None, 1)


def test_merging_decks_leaves_the_destination_in_its_folder(tmp_path):
    with Env(tmp_path) as env:
        keep, gone, home = env.deck("Keep"), env.deck("Gone"), folder(env)
        file_deck(env, keep, home["id"])
        env.card(gone, "Q")
        merged = env.client.post(f"/decks/{keep}/merge", headers=H, json={"from_deck_id": gone}).json()["deck"]
        assert merged["folder_id"] == home["id"]


# --- sync ----------------------------------------------------------------------------------


def test_a_phone_is_sent_folders_and_each_decks_folder(tmp_path):
    with Env(tmp_path) as env:
        deck, home = env.deck("Bio I"), folder(env)
        first = env.sync()
        assert [f["name"] for f in first["folders"]] == ["Biology"]  # made before the first sync, so it is in it

        file_deck(env, deck, home["id"])
        full = env.sync()
        assert [(f["id"], f["name"]) for f in full["folders"]] == [(home["id"], "Biology")]
        assert {d["id"]: d["folder_id"] for d in full["decks"]} == {deck: home["id"]}

        cursor = full["cursor"]
        env.client.patch(f"/folders/{home['id']}", headers=H, json={"name": "Life science"})
        change = env.sync(cursor=cursor)
        assert [f["name"] for f in change["folders"]] == ["Life science"] and change["decks"] == []  # only the folder changed


def test_deleting_a_folder_tells_phones_about_the_folder_and_about_each_deck_it_held(tmp_path):
    with Env(tmp_path) as env:
        d1, d2, home = env.deck("A"), env.deck("B"), folder(env)
        file_deck(env, d1, home["id"])
        file_deck(env, d2, home["id"])
        cursor = env.sync()["cursor"]

        env.client.delete(f"/folders/{home['id']}", headers=H)
        change = env.sync(cursor=cursor)
        assert {"kind": "folder", "id": home["id"]} in change["deleted"]
        assert {d["id"]: d["folder_id"] for d in change["decks"]} == {d1: None, d2: None}  # the phone must hear that they are unfiled


def test_a_folder_deletion_never_arrives_before_the_decks_it_unfiled(tmp_path):
    """Pages end at one cursor for every kind, so a phone never drops a folder while some of its decks still point at it."""
    with Env(tmp_path) as env:
        home = folder(env)
        decks = [env.deck(f"D{i}") for i in range(5)]
        for d in decks:
            file_deck(env, d, home["id"])
        cursor = env.sync()["cursor"]
        env.client.delete(f"/folders/{home['id']}", headers=H)

        unfiled: set[str] = set()
        told_gone, pages = False, 0
        while True:
            page = env.sync(cursor=cursor, limit=2)
            pages += 1
            unfiled |= {d["id"] for d in page["decks"] if d["folder_id"] is None}
            if {"kind": "folder", "id": home["id"]} in page["deleted"]:
                assert unfiled >= set(decks), "the folder was announced gone before all its decks were unfiled"
                told_gone = True
            cursor = page["cursor"]
            if not page["has_more"]:
                break
        assert pages > 1 and told_gone


# --- the database upgrade ------------------------------------------------------------------


def test_a_version_4_database_gains_folders_and_keeps_its_decks(tmp_path):
    path = tmp_path / "v4.db"
    conn = sqlite3.connect(path, isolation_level=None)
    conn.executescript(SCHEMA)
    for statement in (*SYNC_MIGRATION_STATEMENTS, *SYNC_TRIGGER_STATEMENTS):
        conn.execute(statement)
    conn.execute("ALTER TABLE jobs ADD COLUMN target_deck_id TEXT REFERENCES decks(id) ON DELETE SET NULL")
    conn.execute("CREATE TABLE sync_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO sync_meta(key, value) VALUES ('epoch', 'abc')")
    conn.execute("INSERT INTO decks(id, name, topic, created_at, updated_at) VALUES ('d1', 'Old deck', '', 't', 't')")
    conn.execute("PRAGMA user_version = 4")
    assert "folder_id" not in {r[1] for r in conn.execute("PRAGMA table_info(decks)")}
    conn.close()

    db = Database(path)
    db.init()
    db.init()  # a restart changes nothing
    with db.conn() as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == 6
        assert c.execute("SELECT name, folder_id FROM decks WHERE id = 'd1'").fetchone()["folder_id"] is None
        c.execute("INSERT INTO folders(id, name, created_at, updated_at) VALUES ('f1', 'New', 't', 't')")
        assert c.execute("SELECT seq FROM folders WHERE id = 'f1'").fetchone()[0] > 0  # the sync triggers are live
        c.execute("DELETE FROM folders WHERE id = 'f1'")
        assert c.execute("SELECT COUNT(*) FROM tombstones WHERE kind = 'folder' AND id = 'f1'").fetchone()[0] == 1
