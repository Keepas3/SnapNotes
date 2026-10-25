"""Multiple choice: the wrong answers of a card, how they are saved and synced, and the AI's suggestions for them."""
import uuid

import pytest

from app.core.db import Database
from app.llm.choices import usable_wrong_answers
from app.llm.schemas import ChoiceItem, ChoiceSet
from app.store.library import MAX_CHOICES, clean_choices, now
from tests.conftest import FakeSDK
from tests.test_api import H, make_client
from tests.test_sync import Env


def add_card(client, deck_id, front, back, type="basic", choices=()):
    res = client.post(f"/decks/{deck_id}/cards", headers=H, json={"type": type, "front": front, "back": back, "choices": list(choices)})
    assert res.status_code == 201, res.text
    return res.json()


def add_definition_card(client, deck_id, front, back) -> dict:
    """Definition cards only come from the AI, so a test makes one straight in the database."""
    card_id, stamp = uuid.uuid4().hex, now()  # a current timestamp, so it sorts after the cards made earlier in the test
    with client.app.state.library.db.tx() as c:
        c.execute(
            "INSERT INTO cards(id, deck_id, job_id, type, front, back, origin, source_ref, source_snippet, tags_json, difficulty, status, atomic, grounded, issue, duplicate_of, created_at, updated_at)"
            " VALUES (?,?,NULL,'definition',?,?,'notes','p.1','','[]','easy','approved',1,1,'',NULL,?,?)",
            (card_id, deck_id, front, back, stamp, stamp),
        )
    return {"id": card_id}


def new_deck(client, name="Bio") -> str:
    return client.post("/decks", headers=H, json={"name": name}).json()["id"]


def fake_ai(*items: ChoiceItem) -> FakeSDK:
    return FakeSDK(parse_results={ChoiceSet: ChoiceSet(items=list(items))})


def suggest(client, deck_id, **body):
    return client.post(f"/decks/{deck_id}/suggest-choices", headers=H, json=body)


# --- the cleaning rules --------------------------------------------------------------------


def test_wrong_answers_are_trimmed_and_blanks_and_repeats_dropped():
    assert clean_choices(["  Golgi  apparatus ", "", "   ", "golgi apparatus.", "Nucleus"], "Mitochondria", "basic") == ["Golgi apparatus", "Nucleus"]
    assert clean_choices([], "x", "basic") == []


def test_a_wrong_answer_cannot_be_the_right_one():
    with pytest.raises(ValueError, match="right answer"):
        clean_choices(["Nucleus", " mitochondria. "], "Mitochondria", "basic")  # same answer ignoring case, spacing and punctuation


def test_limits_and_the_fill_in_the_blank_exception():
    with pytest.raises(ValueError, match="at most"):
        clean_choices([f"wrong {i}" for i in range(MAX_CHOICES + 1)], "right", "basic")
    assert len(clean_choices([f"wrong {i}" for i in range(MAX_CHOICES)], "right", "definition")) == MAX_CHOICES
    with pytest.raises(ValueError, match="characters"):
        clean_choices(["x" * 201], "right", "basic")
    with pytest.raises(ValueError, match="fill-in-the-blank"):
        clean_choices(["Nucleus"], "ATP", "cloze")
    assert clean_choices([], "ATP", "cloze") == []  # nothing to refuse


# --- saving and reading them ---------------------------------------------------------------


def test_wrong_answers_are_saved_with_a_typed_card_and_read_back(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        made = add_card(env.client, deck, "Where is ATP made?", "Mitochondria", choices=["Nucleus", "Golgi apparatus", "Nucleus"])
        assert made["choices"] == ["Nucleus", "Golgi apparatus"]
        assert env.client.get(f"/cards/{made['id']}", headers=H).json()["choices"] == ["Nucleus", "Golgi apparatus"]
        assert env.client.get(f"/decks/{deck}/cards", headers=H).json()[0]["choices"] == ["Nucleus", "Golgi apparatus"]


def test_wrong_answers_can_be_replaced_cleared_and_left_alone_by_an_edit(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        card = add_card(env.client, deck, "Where is ATP made?", "Mitochondria", choices=["Nucleus", "Golgi"])

        def patch(body):
            return env.client.patch(f"/cards/{card['id']}", headers=H, json=body)

        assert patch({"choices": ["Ribosome", "Vacuole", "Lysosome"]}).json()["choices"] == ["Ribosome", "Vacuole", "Lysosome"]
        assert patch({"difficulty": "hard"}).json()["choices"] == ["Ribosome", "Vacuole", "Lysosome"]  # an edit that omits them keeps them
        assert patch({"choices": []}).json()["choices"] == []
        assert patch({"choices": ["Nucleus"]}).status_code == 200


def test_an_edit_is_checked_against_the_right_answer_it_will_have(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        card = add_card(env.client, deck, "Where is ATP made?", "Mitochondria", choices=["Nucleus"])
        same = env.client.patch(f"/cards/{card['id']}", headers=H, json={"choices": ["mitochondria"]})
        assert same.status_code == 422 and "right answer" in same.json()["detail"]

        # changing the answer in the same edit: "Nucleus" is now the right answer, so it cannot stay as a wrong one
        both = env.client.patch(f"/cards/{card['id']}", headers=H, json={"back": "Nucleus", "choices": ["Nucleus", "Golgi"]})
        assert both.status_code == 422
        ok = env.client.patch(f"/cards/{card['id']}", headers=H, json={"back": "Nucleus", "choices": ["Mitochondria", "Golgi"]})
        assert ok.status_code == 200 and ok.json()["choices"] == ["Mitochondria", "Golgi"]


def test_fill_in_the_blank_cards_take_no_wrong_answers(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        res = env.client.post(f"/decks/{deck}/cards", headers=H, json={"type": "cloze", "front": "The {{c1::cell}} divides.", "back": "", "choices": ["x"]})
        assert res.status_code == 422 and "fill-in-the-blank" in res.json()["detail"]
        assert env.client.get(f"/decks/{deck}/cards", headers=H).json() == []  # nothing was left behind


def test_a_batch_is_saved_all_or_nothing(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        a = add_card(env.client, deck, "Q1?", "A1")
        b = add_card(env.client, deck, "Q2?", "A2")
        c = add_definition_card(env.client, deck, "Q3", "A3")

        ok = env.client.post("/cards/choices", headers=H, json={"items": [{"card_id": a["id"], "choices": ["x", "y"]}, {"card_id": b["id"], "choices": ["p", "q", "r"]}]})
        assert ok.status_code == 200 and ok.json() == {"saved": 2}
        assert [x["choices"] for x in env.client.get(f"/decks/{deck}/cards", headers=H).json()] == [["x", "y"], ["p", "q", "r"], []]

        bad = env.client.post("/cards/choices", headers=H, json={"items": [{"card_id": c["id"], "choices": ["fine", "also fine"]}, {"card_id": a["id"], "choices": ["A1"]}]})
        assert bad.status_code == 422  # the second item is the right answer
        after = {x["id"]: x["choices"] for x in env.client.get(f"/decks/{deck}/cards", headers=H).json()}
        assert after[c["id"]] == [] and after[a["id"]] == ["x", "y"]  # the first item was not kept either

        missing = env.client.post("/cards/choices", headers=H, json={"items": [{"card_id": "nope", "choices": ["x"]}]})
        assert missing.status_code == 404
        assert env.client.post("/cards/choices", headers=H, json={"items": []}).status_code == 422


# --- sync and the database upgrade ---------------------------------------------------------


def test_a_phone_is_sent_the_wrong_answers_and_hears_when_they_change(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        card = add_card(env.client, deck, "Q?", "A", choices=["x", "y"])
        full = env.sync()
        assert [c["choices"] for c in full["cards"]] == [["x", "y"]]

        env.client.patch(f"/cards/{card['id']}", headers=H, json={"choices": ["x", "y", "z"]})
        change = env.sync(cursor=full["cursor"])
        assert [(c["id"], c["choices"]) for c in change["cards"]] == [(card["id"], ["x", "y", "z"])]


def test_a_version_5_database_gains_the_column_and_keeps_its_cards(tmp_path):
    db = Database(tmp_path / "v5.db")
    db.init()
    with db.tx() as c:
        c.execute("INSERT INTO decks(id, name, topic, created_at, updated_at) VALUES ('d', 'Old', '', 't', 't')")
        c.execute(
            "INSERT INTO cards(id, deck_id, type, front, back, origin, source_ref, source_snippet, tags_json, difficulty, status, atomic, grounded, created_at, updated_at)"
            " VALUES ('c1','d','basic','f','b','notes','','','[]','easy','approved',1,1,'t','t')"
        )
        c.execute("ALTER TABLE cards DROP COLUMN choices_json")  # what a version 5 database looks like
        c.execute("PRAGMA user_version = 5")

    db.init()
    db.init()  # a restart changes nothing
    with db.conn() as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == 6
        assert c.execute("SELECT choices_json FROM cards WHERE id = 'c1'").fetchone()[0] == "[]"


# --- checking the AI's suggestions ---------------------------------------------------------


def test_suggestions_that_are_the_right_answer_reworded_are_thrown_out():
    kept = usable_wrong_answers(
        ["Mitochondria", "the mitochondria.", "Not the mitochondria", "Mitochondrion", "The nucleus", "the nucleus!", "Golgi apparatus", "x" * 300, "  "],
        "Mitochondria",
        wanted=3,
    )
    assert kept == ["The nucleus", "Golgi apparatus"]  # the copies, the one containing the answer, the near-copy, the repeat, the huge one and the blank are gone


def test_only_as_many_suggestions_as_wanted_are_kept():
    assert usable_wrong_answers(["a one", "b two", "c three", "d four", "e five"], "right answer", wanted=3) == ["a one", "b two", "c three"]


# --- the suggest endpoint ------------------------------------------------------------------


def test_suggestions_come_back_for_review_and_nothing_is_saved(tmp_path):
    sdk = fake_ai(
        ChoiceItem(index=0, wrong_answers=["The nucleus", "The Golgi apparatus", "Ribosomes"]),
        ChoiceItem(index=1, wrong_answers=["Lysosome", "Vacuole", "Chloroplast"]),
    )
    with make_client(tmp_path, sdk) as client:
        deck = new_deck(client)
        a = add_card(client, deck, "Where is ATP made?", "Mitochondria")
        b = add_definition_card(client, deck, "What does the Golgi do?", "Packages proteins")

        res = suggest(client, deck)
        assert res.status_code == 200, res.text
        body = res.json()
        assert [(s["card_id"], s["front"], s["answer"], s["wrong_answers"]) for s in body["suggestions"]] == [
            (a["id"], "Where is ATP made?", "Mitochondria", ["The nucleus", "The Golgi apparatus", "Ribosomes"]),
            (b["id"], "What does the Golgi do?", "Packages proteins", ["Lysosome", "Vacuole", "Chloroplast"]),
        ]
        assert body["skipped"] == []
        assert [c["choices"] for c in client.get(f"/decks/{deck}/cards", headers=H).json()] == [[], []]  # a suggestion is not a save


def test_a_card_the_ai_could_not_help_with_is_skipped_with_a_reason(tmp_path):
    sdk = fake_ai(
        ChoiceItem(index=0, wrong_answers=["Mitochondria", "the mitochondria.", "Mitochondrion"]),  # all the right answer reworded
        ChoiceItem(index=1, wrong_answers=[]),  # an open question
        ChoiceItem(index=2, wrong_answers=["only one usable"]),  # too few to make a question
        # index 3 is not answered at all
    )
    with make_client(tmp_path, sdk) as client:
        deck = new_deck(client)
        ids = [add_card(client, deck, f"Q{i}?", ["Mitochondria", "Because", "Yes", "Blue"][i])["id"] for i in range(4)]
        body = suggest(client, deck).json()
        assert body["suggestions"] == []
        reasons = {s["card_id"]: s["reason"] for s in body["skipped"]}
        assert "too close" in reasons[ids[0]]
        assert "no clear wrong answers" in reasons[ids[1]]
        assert "too close" in reasons[ids[2]]
        assert "no answer" in reasons[ids[3]]


def test_cards_that_already_have_enough_wrong_answers_and_fill_in_the_blank_cards_are_left_out(tmp_path):
    sdk = fake_ai(ChoiceItem(index=0, wrong_answers=["a one", "b two", "c three"]))
    with make_client(tmp_path, sdk) as client:
        deck = new_deck(client)
        add_card(client, deck, "Has them?", "Right", choices=["x1", "x2", "x3"])
        cloze = add_card(client, deck, "The {{c1::cell}} divides.", "", type="cloze")
        needs = add_card(client, deck, "Needs them?", "Right two")

        body = suggest(client, deck).json()  # asks about only the one that needs wrong answers
        assert [s["card_id"] for s in body["suggestions"]] == [needs["id"]]

        named = suggest(client, deck, card_ids=[cloze["id"], "nope"]).json()  # naming a card that does not qualify says why
        assert named["suggestions"] == [] and {s["card_id"] for s in named["skipped"]} == {cloze["id"], "nope"}


def test_one_request_asks_about_at_most_25_cards(tmp_path):
    sdk = fake_ai(*[ChoiceItem(index=i, wrong_answers=["a one", "b two", "c three"]) for i in range(30)])
    with make_client(tmp_path, sdk) as client:
        deck = new_deck(client)
        for i in range(30):
            add_card(client, deck, f"Q{i}?", f"Answer {i}")
        assert len(suggest(client, deck).json()["suggestions"]) == 25
        text = sdk.calls[0][2]["messages"][0]["content"][-1]["text"]
        assert text.count('"index"') == 25


def test_the_hourly_limit_and_the_spend_limit_hold(tmp_path):
    ok = ChoiceItem(index=0, wrong_answers=["a one", "b two", "c three"])
    with make_client(tmp_path / "a", fake_ai(ok), max_choice_requests_per_hour=1) as client:
        deck = new_deck(client)
        add_card(client, deck, "Q?", "A")
        assert suggest(client, deck).status_code == 200
        limited = suggest(client, deck)
        assert limited.status_code == 429 and "hour" in limited.json()["detail"]

    with make_client(tmp_path / "b", fake_ai(ok), max_job_cost_usd=0.0000001) as client:
        deck = new_deck(client)
        add_card(client, deck, "Q?", "A")
        over = suggest(client, deck)
        assert over.status_code == 409 and "cost limit" in over.json()["detail"]


def test_a_provider_failure_is_a_sentence_not_a_crash(tmp_path):
    with make_client(tmp_path, FakeSDK(parse_results={})) as client:  # the fake has no answer for this stage, so the call blows up
        deck = new_deck(client)
        add_card(client, deck, "Q?", "A")
        res = suggest(client, deck)
        assert res.status_code == 502 and isinstance(res.json()["detail"], str) and "Traceback" not in res.json()["detail"]


def test_suggest_needs_a_real_deck_and_valid_numbers(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        assert suggest(env.client, "nope").status_code == 404
        assert suggest(env.client, deck, wrong_per_card=1).status_code == 422
        assert suggest(env.client, deck, wrong_per_card=5).status_code == 422
        assert suggest(env.client, deck).json() == {"suggestions": [], "skipped": []}  # an empty deck asks the AI nothing


def test_the_demo_stand_in_answers_so_the_feature_can_be_tried_without_a_key(tmp_path):
    with Env(tmp_path) as env:  # demo mode
        deck = env.deck()
        card = add_card(env.client, deck, "Where is ATP made?", "The nucleus")  # even if a canned wrong answer equals the right one, it is not offered
        body = suggest(env.client, deck, wrong_per_card=4).json()
        assert len(body["suggestions"]) == 1 and len(body["suggestions"][0]["wrong_answers"]) == 4
        assert "The nucleus" not in body["suggestions"][0]["wrong_answers"]
        saved = env.client.post("/cards/choices", headers=H, json={"items": [{"card_id": card["id"], "choices": body["suggestions"][0]["wrong_answers"]}]})
        assert saved.status_code == 200
