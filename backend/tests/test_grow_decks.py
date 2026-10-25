"""Growing a deck: add notes to it, merge another deck into it, and type a card by hand."""
import sqlite3
import uuid
from datetime import timedelta

import pytest

from app.core.db import SCHEMA, SYNC_MIGRATION_STATEMENTS, SYNC_TRIGGER_STATEMENTS, Database
from app.export.anki import card_tags, source_html
from app.llm.schemas import ReviewedCard
from app.llm.verify import find_duplicates_of_existing
from app.store.jobs import JobStore
from app.store.library import Library
from tests.test_api import CARDS, H, UNDERSTANDING, card, make_client, make_sdk, upload, wait_for
from tests.test_sync import NOW, Env, log


def store_pair(tmp_path):
    """The stores on their own, with no worker running: for tests that drive a job's steps by hand."""
    db = Database(tmp_path / "t.db")
    db.init()
    return JobStore(db), Library(db), db


def start_job(store: JobStore, **kw) -> str:
    return store.create(doc_id=uuid.uuid4().hex, filename="n.md", kind="md", stored_names=["00_n.md"], auto_approve=True, **kw)


def reviewed(front: str) -> ReviewedCard:
    return ReviewedCard(card=card(front), atomic=True, grounded=True)


def add_card(client, deck_id, front="What organelle produces ATP?", back="Mitochondria", type="basic"):
    res = client.post(f"/decks/{deck_id}/cards", headers=H, json={"type": type, "front": front, "back": back})
    assert res.status_code == 201, res.text
    return res.json()


# --- merging decks -------------------------------------------------------------------------


def test_merge_moves_every_card_and_keeps_study_history(tmp_path):
    with Env(tmp_path) as env:
        keep, gone = env.deck("Keep"), env.deck("Gone")
        a1 = env.card(keep, "A1")
        studied = env.card(gone, "B1")
        drafted, set_aside = env.card(gone, "B2", status="draft"), env.card(gone, "B3", status="rejected")
        env.sync(logs=[log(studied, NOW - timedelta(hours=1))])  # a review of a card in the deck that is about to go
        with env.db.conn() as c:
            logs_before = c.execute("SELECT COUNT(*) FROM review_logs").fetchone()[0]
            state_before = tuple(c.execute("SELECT state_json, due FROM card_state WHERE card_id = ?", (studied,)).fetchone())
        assert logs_before == 1

        res = env.client.post(f"/decks/{keep}/merge", headers=H, json={"from_deck_id": gone})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["moved"] == 3
        assert (body["deck"]["id"], body["deck"]["name"]) == (keep, "Keep")  # the destination keeps its title
        assert body["deck"]["counts"] == {"draft": 1, "approved": 2, "rejected": 1}  # every status moved, none promoted

        assert env.client.get(f"/decks/{gone}", headers=H).status_code == 404
        in_deck = {c["id"]: c["status"] for c in env.client.get(f"/decks/{keep}/cards", headers=H).json()}
        assert in_deck == {a1: "approved", studied: "approved", drafted: "draft", set_aside: "rejected"}
        with env.db.conn() as c:  # nothing about the studied card was touched
            assert c.execute("SELECT COUNT(*) FROM review_logs").fetchone()[0] == logs_before
            assert tuple(c.execute("SELECT state_json, due FROM card_state WHERE card_id = ?", (studied,)).fetchone()) == state_before


def test_merge_refuses_bad_requests_and_changes_nothing(tmp_path):
    with Env(tmp_path) as env:
        one, two = env.deck("One"), env.deck("Two")
        env.card(one, "Q1")
        env.card(two, "Q2")

        def merge(into, source):
            return env.client.post(f"/decks/{into}/merge", headers=H, json={"from_deck_id": source})

        assert merge(one, one).status_code == 409  # into itself would delete the deck's own cards
        assert merge("nope", two).status_code == 404
        assert merge(one, "nope").status_code == 404
        assert env.client.post(f"/decks/{one}/merge", headers=H, json={}).status_code == 422
        decks = {d["id"]: d["counts"]["approved"] for d in env.client.get("/decks", headers=H).json()}
        assert decks == {one: 1, two: 1}


def test_a_merge_cannot_delete_a_deck_into_itself_even_when_called_directly(tmp_path):
    _, library, _ = store_pair(tmp_path)
    deck = library.create_deck("Solo")["id"]
    with pytest.raises(ValueError):
        library.merge_decks(deck, deck)
    assert library.get_deck(deck) is not None


def test_a_deletion_never_arrives_before_the_moves_that_came_first(tmp_path):
    """The app drops a deck's cards when told the deck is gone, so the page that says so must not outrun the moved cards."""
    with Env(tmp_path) as env:
        keep, gone = env.deck("Keep"), env.deck("Gone")
        env.card(keep, "K")
        moved = {env.card(gone, f"G{i}") for i in range(5)}
        cursor = env.sync()["cursor"]
        assert env.client.post(f"/decks/{keep}/merge", headers=H, json={"from_deck_id": gone}).status_code == 200

        phone: dict[str, str] = {}  # card id -> deck id, as the app would hold it
        announced_gone, pages = False, 0
        while True:
            page = env.sync(cursor=cursor, limit=2)
            pages += 1
            for c in page["cards"]:
                phone[c["id"]] = c["deck_id"]
            for d in page["deleted"]:
                assert d["kind"] == "deck" and d["id"] == gone  # no card was deleted, only the emptied deck
                assert set(phone) >= moved, "the deck was announced gone before all its cards had moved"
                announced_gone = True
            assert page["cursor"] >= cursor
            cursor = page["cursor"]
            if not page["has_more"]:
                break
        assert pages > 1 and announced_gone
        assert phone == {card_id: keep for card_id in moved}


# --- adding notes to a deck ----------------------------------------------------------------


def test_uploading_to_an_unknown_deck_is_refused_before_anything_is_saved(tmp_path):
    with make_client(tmp_path) as client:
        assert upload(client, deck_id="no-such-deck").status_code == 404
        assert client.get("/jobs", headers=H).json() == []
        upload_dir = client.app.state.cfg.upload_dir
        assert not upload_dir.exists() or not any(upload_dir.iterdir())


def test_a_blank_deck_field_means_a_new_deck(tmp_path):
    with make_client(tmp_path, make_sdk(n_jobs=1)) as client:
        job = upload(client, deck_id="", auto_approve="true").json()["job_id"]
        done = wait_for(client, job, "completed")
        assert done["target_deck_id"] is None
        assert [d["name"] for d in client.get("/decks", headers=H).json()] == ["Cell biology"]


def test_added_cards_join_the_existing_deck_and_leave_its_name_alone(tmp_path):
    with make_client(tmp_path, make_sdk(n_jobs=1)) as client:
        deck = client.post("/decks", headers=H, json={"name": "My biology", "topic": "mine"}).json()["id"]
        job = upload(client, deck_id=deck, auto_approve="true").json()["job_id"]
        done = wait_for(client, job, "completed")

        assert done["target_deck_id"] == deck and done["deck_id"] == deck
        decks = client.get("/decks", headers=H).json()
        assert [(d["name"], d["topic"]) for d in decks] == [("My biology", "mine")]  # no second deck, no rename
        assert decks[0]["counts"] == {"draft": 2, "approved": 0, "rejected": 1}  # three cards, one a repeat inside the upload
        assert {c["deck_id"] for c in client.get(f"/jobs/{job}/cards", headers=H).json()} == {deck}


def test_a_card_that_repeats_one_in_the_deck_is_set_aside_and_says_so(tmp_path):
    with make_client(tmp_path, make_sdk(n_jobs=1)) as client:
        deck = client.post("/decks", headers=H, json={"name": "Bio"}).json()["id"]
        existing = add_card(client, deck, "What organelle produces ATP?")  # approved; same question as CARDS[0]
        job = upload(client, deck_id=deck, auto_approve="true").json()["job_id"]
        wait_for(client, job, "completed")

        new = {c["front"]: c for c in client.get(f"/jobs/{job}/cards", headers=H).json()}
        repeat = new["What organelle produces ATP?"]
        assert (repeat["status"], repeat["duplicate_of"]) == ("rejected", existing["id"])
        assert "Already in this deck." in repeat["issue"]
        assert new["Define osmosis"]["status"] == "draft"  # genuinely new
        assert existing["status"] == "approved"  # the old card is untouched


def test_a_card_the_user_set_aside_does_not_block_a_new_one(tmp_path):
    with make_client(tmp_path, make_sdk(n_jobs=1)) as client:
        deck = client.post("/decks", headers=H, json={"name": "Bio"}).json()["id"]
        old = add_card(client, deck, "What organelle produces ATP?")
        assert client.patch(f"/cards/{old['id']}", headers=H, json={"status": "rejected"}).status_code == 200
        job = upload(client, deck_id=deck, auto_approve="true").json()["job_id"]
        wait_for(client, job, "completed")

        first = next(c for c in client.get(f"/jobs/{job}/cards", headers=H).json() if c["front"] == "What organelle produces ATP?")
        assert (first["status"], first["duplicate_of"]) == ("draft", None)


def test_new_questions_are_not_mistaken_for_repeats():
    cards = [card("Define osmosis"), card("What does the Golgi apparatus do?"), card("")]
    existing = [("old1", "Define osmosis?"), ("old2", "What does the nucleus do?"), ("old3", "  ")]
    assert find_duplicates_of_existing(cards, existing) == {0: "old1"}  # punctuation and case do not matter; a different organelle does
    assert find_duplicates_of_existing(cards, []) == {}


def test_a_deck_deleted_before_generation_finishes_gets_a_new_deck(tmp_path):
    store, library, _ = store_pair(tmp_path)
    deck = library.create_deck("Doomed")["id"]
    job = start_job(store, target_deck_id=deck)
    store.save_research(job, UNDERSTANDING, [])
    assert library.delete_deck(deck)
    assert store.get(job)["target_deck_id"] is None  # the link clears itself

    made = store.save_generation(job, [reviewed("Q1")])
    assert made != deck
    assert library.get_deck(made)["name"] == "Cell biology"


def test_a_merge_redirects_uploads_aimed_at_the_removed_deck(tmp_path):
    store, library, _ = store_pair(tmp_path)
    keep, gone = library.create_deck("Keep")["id"], library.create_deck("Gone")["id"]
    waiting, finished = start_job(store, target_deck_id=gone), start_job(store, target_deck_id=gone)
    store.save_research(finished, UNDERSTANDING, [])
    assert store.save_generation(finished, [reviewed("Q1")]) == gone  # it added to "Gone"

    library.merge_decks(keep, gone)
    assert store.get(waiting)["target_deck_id"] == keep  # still headed somewhere that exists
    assert store.get(finished)["deck_id"] == keep  # and "open deck" from a finished job still works
    assert [c["front"] for c in library.list_cards(deck_id=keep)] == ["Q1"]


# --- typing a card -------------------------------------------------------------------------


def test_a_typed_card_is_approved_labelled_and_reaches_the_phone(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck("Mine")
        cursor = env.sync()["cursor"]
        made = add_card(env.client, deck, "Capital of France?", "Paris")

        assert (made["origin"], made["status"], made["type"]) == ("manual", "approved", "basic")
        assert (made["source_ref"], made["job_id"], made["issue"]) == ("Added by you", None, "")
        assert made["atomic"] and made["grounded"]
        assert env.client.get(f"/decks/{deck}", headers=H).json()["counts"]["approved"] == 1

        page = env.sync(cursor=cursor)
        assert [(c["id"], c["status"], c["origin"]) for c in page["cards"]] == [(made["id"], "approved", "manual")]
        assert page["states"] == []  # never studied, so the phone starts it as new


def test_a_typed_fill_in_the_blank_card_needs_a_blank_but_not_an_explanation(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        ok = add_card(env.client, deck, "The {{c1::mitochondrion}} makes ATP.", "", type="cloze")
        assert (ok["type"], ok["back"]) == ("cloze", "")

        res = env.client.post(f"/decks/{deck}/cards", headers=H, json={"type": "cloze", "front": "No blank here", "back": "x"})
        assert res.status_code == 422 and "blank" in res.json()["detail"]


def test_a_typed_fill_in_the_blank_card_can_be_edited_again_without_an_explanation(tmp_path):
    """The app's editor sends the trimmed back with every save; for such a card that is an empty string."""
    with Env(tmp_path) as env:
        deck = env.deck()
        made = add_card(env.client, deck, "The {{c1::cell}} divides.", "", type="cloze")

        res = env.client.patch(f"/cards/{made['id']}", headers=H, json={"front": "The {{c1::cell}} divides fast.", "back": "", "difficulty": "easy"})
        assert res.status_code == 200, res.text
        assert (res.json()["front"], res.json()["back"], res.json()["difficulty"]) == ("The {{c1::cell}} divides fast.", "", "easy")


def test_an_ordinary_card_still_cannot_be_saved_with_an_empty_back(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        basic = add_card(env.client, deck, "Capital of France?", "Paris")

        for empty in ("", "   "):
            res = env.client.patch(f"/cards/{basic['id']}", headers=H, json={"back": empty})
            assert res.status_code == 422 and res.json()["detail"] == "Write the back of the card."
        assert env.client.get(f"/cards/{basic['id']}", headers=H).json()["back"] == "Paris"  # the refused edit changed nothing
        assert env.client.patch(f"/cards/{basic['id']}", headers=H, json={"front": ""}).status_code == 422  # the front is still required
        assert env.client.patch(f"/cards/{basic['id']}", headers=H, json={"difficulty": "hard"}).status_code == 200  # omitting back is fine


def test_a_typed_card_must_have_text_and_a_real_deck(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()

        def post(deck_id, body):
            return env.client.post(f"/decks/{deck_id}/cards", headers=H, json=body)

        assert "front" in post(deck, {"front": "   ", "back": "x"}).json()["detail"]
        assert "back" in post(deck, {"front": "Q?", "back": " "}).json()["detail"]
        assert post(deck, {"type": "definition", "front": "Q?", "back": "A"}).status_code == 422  # only basic and cloze are typed by hand
        assert post(deck, {"front": "x" * 2001, "back": "A"}).status_code == 422
        assert post("nope", {"front": "Q?", "back": "A"}).status_code == 404
        assert env.client.get(f"/decks/{deck}/cards", headers=H).json() == []  # none of the above left a card behind


def test_a_typed_card_exports_with_its_origin(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck("Mine")
        add_card(env.client, deck, "Capital of France?", "Paris")

        assert source_html({"origin": "manual", "source_ref": "Added by you"}) == "Added by you"
        assert "source-manual" in card_tags({"origin": "manual", "tags": []})
        csv = env.client.get(f"/decks/{deck}/export", headers=H, params={"format": "csv"})
        assert csv.status_code == 200 and "manual" in csv.text and "Added by you" in csv.text
        assert env.client.get(f"/decks/{deck}/export", headers=H).status_code == 200  # the Anki package builds


# --- the database upgrade ------------------------------------------------------------------


def test_a_version_3_database_gains_the_target_deck_column_and_keeps_its_jobs(tmp_path):
    path = tmp_path / "v3.db"
    conn = sqlite3.connect(path, isolation_level=None)
    conn.executescript(SCHEMA)  # the original tables, without the newer columns
    for statement in (*SYNC_MIGRATION_STATEMENTS, *SYNC_TRIGGER_STATEMENTS):
        conn.execute(statement)
    conn.execute("CREATE TABLE sync_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO sync_meta(key, value) VALUES ('epoch', 'abc')")
    conn.execute("INSERT INTO documents(id, filename, kind, stored_names_json, created_at) VALUES ('d', 'n.md', 'md', '[]', 't')")
    conn.execute("INSERT INTO jobs(id, document_id, status, created_at, updated_at) VALUES ('j', 'd', 'completed', 't', 't')")
    conn.execute("PRAGMA user_version = 3")
    assert "target_deck_id" not in {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    conn.close()

    db = Database(path)
    db.init()
    db.init()  # a restart changes nothing
    with db.conn() as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == 6
        assert c.execute("SELECT status, target_deck_id FROM jobs WHERE id = 'j'").fetchone()["status"] == "completed"
        assert c.execute("SELECT target_deck_id FROM jobs WHERE id = 'j'").fetchone()[0] is None


def test_the_shared_card_fixture_still_has_the_fronts_this_file_relies_on():
    # The tests above lean on CARDS[0] and CARDS[2] having these exact fronts; fail loudly here if the shared fixture changes.
    assert [c.front for c in CARDS] == ["What organelle produces ATP?", "Which organelle produces ATP?", "Define osmosis"]
