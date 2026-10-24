import csv
import io
import json
import sqlite3
import uuid
import zipfile
from html.parser import HTMLParser
from urllib.parse import unquote

import pytest

from app.export.anki import build_apkg, deck_id_for
from app.store.library import now as ts_now
from tests.test_api import H
from tests.test_sync import Env

TS = 1_700_000_000.0  # fixed timestamp so exported files are reproducible


def add_card(env: Env, deck_id: str, front="Q?", back="A", *, type_="basic", origin="notes", source_ref="p.1",
             tags=None, status="approved", difficulty="medium") -> str:
    card_id, ts = uuid.uuid4().hex, ts_now()
    with env.db.tx() as c:
        c.execute(
            "INSERT INTO cards(id, deck_id, job_id, type, front, back, origin, source_ref, source_snippet, tags_json,"
            " difficulty, status, atomic, grounded, issue, duplicate_of, created_at, updated_at)"
            " VALUES (?,?,NULL,?,?,?,?,?,'', ?,?,?,1,1,'',NULL,?,?)",
            (card_id, deck_id, type_, front, back, origin, source_ref, json.dumps(tags or []), difficulty, status, ts, ts),
        )
    return card_id


def read_apkg(data: bytes):
    """Open an .apkg the way Anki does: unzip it and read the SQLite collection inside."""
    z = zipfile.ZipFile(io.BytesIO(data))
    assert sorted(z.namelist()) == ["collection.anki2", "media"]
    assert json.loads(z.read("media")) == {}
    conn = sqlite3.connect(":memory:")
    conn.deserialize(z.read("collection.anki2"))  # read the collection straight from memory, no temp files
    conn.row_factory = sqlite3.Row
    return conn


def notes(conn):
    return [{"guid": r["guid"], "fields": r["flds"].split("\x1f"), "tags": r["tags"].split(), "mid": r["mid"]}
            for r in conn.execute("SELECT guid, flds, tags, mid FROM notes ORDER BY id")]


def export(env: Env, deck_id: str, fmt: str):
    return env.client.get(f"/decks/{deck_id}/export", headers=H, params={"format": fmt})


# --- Anki -----------------------------------------------------------------------------------


def test_apkg_contains_the_deck_notes_and_note_types(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck("Cell biology")
        add_card(env, deck, "What makes ATP?", "Mitochondria", tags=["energy"])
        add_card(env, deck, "Water moves from {{c1::low}} to {{c2::high}} concentration.", "Osmosis", type_="cloze")
        add_card(env, deck, "Osmosis", "Movement of water", type_="definition")

        res = export(env, deck, "apkg")
        assert res.status_code == 200 and res.headers["content-type"] == "application/octet-stream"
        conn = read_apkg(res.content)

        found = notes(conn)
        assert len(found) == 3
        basic, cloze, definition = found
        assert basic["fields"][:2] == ["What makes ATP?", "Mitochondria"]
        assert cloze["fields"][0] == "Water moves from {{c1::low}} to {{c2::high}} concentration."  # Anki's own syntax, unchanged
        assert cloze["fields"][1] == "Osmosis"  # the explanation becomes the Extra field
        assert definition["fields"][:2] == ["Osmosis", "Movement of water"]

        models = json.loads(conn.execute("SELECT models FROM col").fetchone()[0])
        by_name = {m["name"]: m for m in models.values()}
        assert set(by_name) == {"SnapNotes Basic", "SnapNotes Cloze"}
        assert by_name["SnapNotes Cloze"]["type"] == 1  # cloze note type
        assert by_name["SnapNotes Basic"]["type"] == 0

        # one Anki card per note, except a two-blank cloze which makes two
        assert conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 4
        decks = json.loads(conn.execute("SELECT decks FROM col").fetchone()[0])
        assert "Cell biology" in {d["name"] for d in decks.values()}


def test_only_approved_cards_are_exported(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        add_card(env, deck, "approved one")
        add_card(env, deck, "still a draft", status="draft")
        add_card(env, deck, "rejected", status="rejected")
        conn = read_apkg(export(env, deck, "apkg").content)
        assert [n["fields"][0] for n in notes(conn)] == ["approved one"]


def test_note_guids_are_stable_so_reimporting_updates_instead_of_duplicating(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        a, b = add_card(env, deck, "A"), add_card(env, deck, "B")
        first = notes(read_apkg(export(env, deck, "apkg").content))
        env.client.patch(f"/cards/{a}", headers=H, json={"front": "A edited"})
        second = notes(read_apkg(export(env, deck, "apkg").content))

        assert [n["guid"] for n in first] == [n["guid"] for n in second]
        assert len({n["guid"] for n in first}) == 2 and first[0]["mid"] == second[0]["mid"]
        assert second[0]["fields"][0] == "A edited"
        assert b  # (silence: second card exists only to prove guids differ)


def test_the_same_inputs_and_timestamp_give_the_same_file():
    deck = {"id": "0123456789abcdef0123456789abcdef", "name": "Deck", "topic": ""}
    card = {"id": "c1", "type": "basic", "front": "Q", "back": "A", "origin": "notes", "source_ref": "p.1", "tags": [], "difficulty": "easy"}
    one = read_apkg(build_apkg(deck, [card], timestamp=TS))
    two = read_apkg(build_apkg(deck, [card], timestamp=TS))
    assert notes(one) == notes(two)
    assert 1 << 30 <= deck_id_for(deck["id"]) < 1 << 31


def test_text_is_html_escaped_and_line_breaks_kept(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        add_card(env, deck, "What does <b>bold</b> & <script>alert(1)</script> mean?\nSecond line", "5 < 6 && 7 > 3")
        fields = notes(read_apkg(export(env, deck, "apkg").content))[0]["fields"]
        assert fields[0] == "What does &lt;b&gt;bold&lt;/b&gt; &amp; &lt;script&gt;alert(1)&lt;/script&gt; mean?<br>Second line"
        assert fields[1] == "5 &lt; 6 &amp;&amp; 7 &gt; 3"


def test_a_cloze_card_without_blanks_becomes_a_basic_card(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        add_card(env, deck, "An edited cloze with no blanks left", "explanation", type_="cloze")
        conn = read_apkg(export(env, deck, "apkg").content)
        assert conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 1  # a note with zero cards would be unusable
        models = json.loads(conn.execute("SELECT models FROM col").fetchone()[0])
        mid = str(notes(conn)[0]["mid"])
        assert models[mid]["name"] == "SnapNotes Basic"


def test_tags_have_no_spaces_and_mark_the_source(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        add_card(env, deck, "q1", tags=["organic chemistry", "energy", "energy", " "])
        add_card(env, deck, "q2", origin="web", source_ref="https://en.wikipedia.org/wiki/Ribosome")
        first, second = notes(read_apkg(export(env, deck, "apkg").content))
        assert first["tags"] == ["organic_chemistry", "energy", "snapnotes", "source-notes"]
        assert second["tags"] == ["snapnotes", "source-web"]


def test_source_field_links_only_safe_web_urls(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        add_card(env, deck, "q1", origin="web", source_ref="https://www.example.org/a?b=1&c=2")
        add_card(env, deck, "q2", origin="web", source_ref="javascript:alert(1)")
        add_card(env, deck, "q3", origin="notes", source_ref="slide 4 <b>")
        one, two, three = (n["fields"][2] for n in notes(read_apkg(export(env, deck, "apkg").content)))
        assert one == 'From the web: <a href="https://www.example.org/a?b=1&amp;c=2">example.org</a>'
        assert "<a" not in two and "javascript:alert(1)" in two  # shown as text, never linked
        assert three == "From your notes: slide 4 &lt;b&gt;"


class _Tags(HTMLParser):
    """Records every tag and attribute in a piece of HTML, the way Anki's web view will see it."""

    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, list[tuple[str, str | None]]]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, attrs))


HOSTILE_URLS = [
    'https://example.com/x"onmouseover="alert(1)',  # ends the href attribute with a double quote
    "https://example.com/x' onmouseover='alert(1)",  # same, with single quotes and a space
    "https://example.com/x><script>alert(1)</script>",  # closes the tag and opens a script
    "https://example.com/x onclick=alert(1)",  # a space starts a new attribute
    "https://example.com/x\nonmouseover=alert(1)",  # so does a line break
    "https://example.com/x`onmouseover=alert(1)",  # a backtick ends an unquoted attribute in old browsers
    'https://example.com/x\\"onmouseover=\\"alert(1)',  # escaped quotes
    "https://example.com/</a><img src=x onerror=alert(1)>",  # leaves the link entirely
]


@pytest.mark.parametrize("url", HOSTILE_URLS)
def test_a_hostile_source_url_can_never_add_attributes_or_tags(tmp_path, url):
    with Env(tmp_path) as env:
        deck = env.deck()
        add_card(env, deck, "q", origin="web", source_ref=url)
        source = notes(read_apkg(export(env, deck, "apkg").content))[0]["fields"][2]

        parsed = _Tags()
        parsed.feed(source)
        assert [tag for tag, _ in parsed.tags] == ["a"], f"unexpected markup in {source!r}"
        attrs = parsed.tags[0][1]
        assert [name for name, _ in attrs] == ["href"], f"extra attributes in {source!r}"
        href = attrs[0][1]
        assert href.startswith("https://example.com/")
        assert not any(c in href for c in ' "<>`\n\\'), f"raw special character in href {href!r}"


def test_clean_urls_are_linked_unchanged_and_odd_schemes_are_not(tmp_path):
    from app.export import safe_http_url

    assert safe_http_url("https://example.org/a%20b?q=1&r=(2)#frag") == "https://example.org/a%20b?q=1&r=(2)#frag"  # no double encoding
    assert safe_http_url("  http://example.org/x  ") == "http://example.org/x"
    assert safe_http_url("https://example.org/café") == "https://example.org/caf%C3%A9"
    for bad in ("javascript:alert(1)", "data:text/html,<script>1</script>", "file:///etc/passwd", "https:/no-host", "ftp://example.org", "", "//example.org"):
        assert safe_http_url(bad) is None, bad


def test_deck_names_cannot_create_sub_decks(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck("Bio::Cells")
        add_card(env, deck)
        decks = json.loads(read_apkg(export(env, deck, "apkg").content).execute("SELECT decks FROM col").fetchone()[0])
        assert "Bio - Cells" in {d["name"] for d in decks.values()}


# --- CSV ------------------------------------------------------------------------------------


def parse_csv(data: bytes):
    assert data.startswith(b"\xef\xbb\xbf")  # BOM, so Excel reads UTF-8
    return list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def test_csv_has_a_header_and_one_row_per_approved_card(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck("Chem")
        add_card(env, deck, "Why?", "Because", tags=["a", "b c"], difficulty="hard")
        add_card(env, deck, "draft", status="draft")
        res = export(env, deck, "csv")
        assert res.status_code == 200 and res.headers["content-type"] == "text/csv; charset=utf-8"
        rows = parse_csv(res.content)
        assert rows[0] == ["deck", "type", "front", "back", "source", "origin", "tags", "difficulty"]
        assert rows[1] == ["Chem", "basic", "Why?", "Because", "p.1", "notes", "a b c", "hard"]
        assert len(rows) == 2


def test_csv_round_trips_commas_quotes_newlines_and_unicode(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        front, back = 'Say "héllo", then\nwait', "naïve, 日本語 ✓"
        add_card(env, deck, front, back)
        rows = parse_csv(export(env, deck, "csv").content)
        assert rows[1][2:4] == [front, back]


def test_csv_neutralises_spreadsheet_formulas(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        add_card(env, deck, '=HYPERLINK("http://evil.example","click")', "+1 charge", source_ref="@cmd")
        add_card(env, deck, "-5 is negative", "plain")
        rows = parse_csv(export(env, deck, "csv").content)
        assert rows[1][2] == "'=HYPERLINK(\"http://evil.example\",\"click\")"
        assert rows[1][3] == "'+1 charge" and rows[1][4] == "'@cmd"
        assert rows[2][2] == "'-5 is negative" and rows[2][3] == "plain"


# --- endpoint behaviour ---------------------------------------------------------------------


def test_filenames_survive_unicode_and_unsafe_characters(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck('Biología: "células"/week 1')
        add_card(env, deck)
        header = export(env, deck, "apkg").headers["content-disposition"]
        assert header.startswith('attachment; filename="')
        ascii_part = header.split('filename="')[1].split('"')[0]
        assert ascii_part.endswith(".apkg") and all(ord(c) < 128 and c not in '/\\:"' for c in ascii_part)
        # each run of unsafe characters (":" before the space, `"` and `"/` around the word) collapses to one "_"
        assert unquote(header.split("filename*=UTF-8''")[1]) == "Biología_ _células_week 1.apkg"


def test_export_errors(tmp_path):
    with Env(tmp_path) as env:
        empty = env.deck("Empty")
        add_card(env, empty, status="draft")
        assert export(env, empty, "apkg").status_code == 409
        assert export(env, empty, "csv").status_code == 409
        assert export(env, "nope", "apkg").status_code == 404
        assert env.client.get(f"/decks/{empty}/export", headers=H, params={"format": "pdf"}).status_code == 422
        assert env.client.get(f"/decks/{empty}/export").status_code == 401  # needs the token like everything else


def test_default_format_is_anki(tmp_path):
    with Env(tmp_path) as env:
        deck = env.deck()
        add_card(env, deck)
        res = env.client.get(f"/decks/{deck}/export", headers=H)
        assert res.status_code == 200 and zipfile.is_zipfile(io.BytesIO(res.content))


@pytest.mark.parametrize("fmt", ["apkg", "csv"])
def test_export_works_end_to_end_on_a_demo_generated_deck(tmp_path, fmt):
    from tests.test_api import upload, wait_for

    with Env(tmp_path) as env:
        job_id = upload(env.client, auto_approve="true").json()["job_id"]
        job = wait_for(env.client, job_id, "completed")
        cards = env.client.get(f"/jobs/{job_id}/cards", headers=H).json()
        env.client.post("/cards/review", headers=H, json={"approve_ids": [c["id"] for c in cards if c["status"] == "draft"]})
        res = export(env, job["deck_id"], fmt)
        assert res.status_code == 200
        if fmt == "csv":
            assert len(parse_csv(res.content)) == 1 + 6
        else:
            kinds = {json.loads(read_apkg(res.content).execute("SELECT models FROM col").fetchone()[0])[str(n["mid"])]["name"] for n in notes(read_apkg(res.content))}
            assert kinds == {"SnapNotes Basic", "SnapNotes Cloze"}
