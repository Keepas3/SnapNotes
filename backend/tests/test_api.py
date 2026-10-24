import dataclasses
import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.client import LLMClient
from app.llm.schemas import Card, CardSet, CardVerdict, UnderstandResult, VerifyResult
from app.main import create_app
from tests.conftest import FakeSDK, citation, response, text_block

TOKEN = "test-token-0123456789abcdef"  # long enough to pass the server's minimum-strength check
H = {"Authorization": f"Bearer {TOKEN}"}

UNDERSTANDING = UnderstandResult(
    main_topic="Cell biology", subject_area="biology", subtopics=["Mitochondria"],
    key_terms=["ATP"], gaps=["Notes say mitochondria only make DNA"], search_queries=["mitochondria function"],
)


def card(front: str, origin: str = "notes") -> Card:
    return Card(type="basic", front=front, back="answer", origin=origin, source_ref="p.1",
                source_snippet="snippet", tags=["bio"], difficulty="easy")


CARDS = [card("What organelle produces ATP?"), card("Which organelle produces ATP?"), card("Define osmosis")]  # [1] duplicates [0]
VERDICTS = VerifyResult(verdicts=[
    CardVerdict(index=i, atomic=True, grounded=True, issue="", duplicate_of=0 if i == 1 else -1) for i in range(3)
])


def enrich_response(n_citations: int = 2):
    blocks = [
        text_block(f"- Fact {i}.", [citation(f"https://example.org/{i}", f"Source {i}", f"passage {i}")])
        for i in range(n_citations)
    ]
    return response(blocks) if blocks else response([text_block("Nothing relevant found.")])


def make_sdk(n_jobs: int = 5, n_citations: int = 2) -> FakeSDK:
    return FakeSDK(
        parse_results={UnderstandResult: UNDERSTANDING, CardSet: CardSet(cards=CARDS), VerifyResult: VERDICTS},
        create_responses=[enrich_response(n_citations) for _ in range(n_jobs)],
    )


def make_client(tmp_path, sdk: FakeSDK | None = None, **overrides) -> TestClient:
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN, **overrides)
    sdk = sdk or make_sdk()
    return TestClient(create_app(cfg, llm_factory=lambda: LLMClient(sdk, cfg)))


def upload(client, name="notes.md", content=b"# Cells\nMitochondria make ATP.", **data):
    return client.post("/documents", headers=H, files=[("files", (name, content, "application/octet-stream"))], data=data)


def wait_for(client, job_id, statuses, timeout=10.0):
    statuses = {statuses} if isinstance(statuses, str) else set(statuses)
    deadline, job = time.time() + timeout, None
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}", headers=H).json()
        if job["status"] in statuses:
            return job
        time.sleep(0.03)
    raise AssertionError(f"job never reached {statuses}; last state: {job}")


# --- auth, limits, startup -----------------------------------------------------------------


def test_health_is_open_and_everything_else_needs_the_token(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/jobs").status_code == 401
        assert client.get("/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/jobs", headers={"Authorization": "Basic " + TOKEN}).status_code == 401
        assert client.get("/jobs", headers=H).status_code == 200


def test_server_refuses_to_start_without_a_token(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, api_token="", allow_no_auth=False))
    with pytest.raises(RuntimeError, match="SNAPNOTES_TOKEN"):
        with TestClient(app):
            pass


def test_no_auth_mode_for_local_dev(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, api_token="", allow_no_auth=True))
    with TestClient(app) as client:
        assert client.get("/jobs").status_code == 200


def test_request_rate_limit(tmp_path):
    with make_client(tmp_path, requests_per_minute=3) as client:
        codes = [client.get("/jobs", headers=H).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert client.get("/health").status_code == 200  # health is exempt


def test_hourly_job_quota(tmp_path):
    with make_client(tmp_path, max_jobs_per_hour=1) as client:
        assert upload(client).status_code == 202
        second = upload(client)
        assert second.status_code == 429 and "Retry-After" in second.headers


# --- upload validation -----------------------------------------------------------------------


def test_upload_rejections_leave_nothing_behind(tmp_path):
    with make_client(tmp_path, max_upload_bytes=1000, max_files_per_upload=2) as client:
        assert upload(client, "evil.exe", b"MZ").status_code == 415
        assert upload(client, "empty.txt", b"").status_code == 422
        assert upload(client, "big.txt", b"x" * 2000).status_code == 413
        assert upload(client, "fake.pdf", b"not a pdf").status_code == 415
        assert client.post("/documents", headers=H).status_code == 422  # no files at all
        three = [("files", (f"n{i}.txt", b"hi", "text/plain")) for i in range(3)]
        assert client.post("/documents", headers=H, files=three).status_code == 400
        assert list((tmp_path / "uploads").iterdir()) == []
        assert client.get("/jobs", headers=H).json() == []


def test_filename_cannot_escape_the_upload_directory(tmp_path):
    with make_client(tmp_path, delete_originals=False) as client:
        body = upload(client, "../../../etc/passwd.md", b"hello").json()
        files = list((tmp_path / "uploads" / body["document_id"]).iterdir())
        assert [f.name for f in files] == ["00_passwd.md"]


# --- the happy path ------------------------------------------------------------------------


def test_full_flow_with_source_approval(tmp_path):
    with make_client(tmp_path) as client:
        body = upload(client).json()
        job_id, doc_id = body["job_id"], body["document_id"]

        job = wait_for(client, job_id, "awaiting_approval")
        assert job["understanding"]["main_topic"] == "Cell biology"
        ev = client.get(f"/jobs/{job_id}/evidence", headers=H).json()
        assert len(ev["evidence"]) == 2 and ev["understanding"]["subject_area"] == "biology"

        # wrong-state and bad-id handling
        assert client.post(f"/jobs/{job_id}/approve", headers=H, json={"rejected_ids": [999999]}).status_code == 400
        approve = client.post(f"/jobs/{job_id}/approve", headers=H, json={"rejected_ids": [ev["evidence"][0]["id"]]})
        assert approve.json() == {"approved": 1, "rejected": 1}
        assert client.post(f"/jobs/{job_id}/approve", headers=H, json={}).status_code in (404, 409)  # already approved

        job = wait_for(client, job_id, "completed")
        assert job["deck_id"] and job["cost_usd"] > 0
        assert not (tmp_path / "uploads" / doc_id).exists()  # originals removed once cards exist

        cards = client.get(f"/jobs/{job_id}/cards", headers=H).json()
        assert [c["status"] for c in cards] == ["draft", "rejected", "draft"]  # duplicate starts out rejected
        assert cards[1]["duplicate_of"] == cards[0]["id"]

        deck = client.get(f"/decks/{job['deck_id']}", headers=H).json()
        assert deck["name"] == "Cell biology" and deck["counts"] == {"draft": 2, "approved": 0, "rejected": 1}

        # review: approve both drafts in one call
        res = client.post("/cards/review", headers=H, json={"approve_ids": [cards[0]["id"], cards[2]["id"]]})
        assert res.json() == {"changed": 2}
        deck = client.get(f"/decks/{job['deck_id']}", headers=H).json()
        assert deck["counts"] == {"draft": 0, "approved": 2, "rejected": 1}
        approved = client.get(f"/decks/{job['deck_id']}/cards", headers=H, params={"status": "approved"}).json()
        assert len(approved) == 2


def test_auto_approve_skips_the_approval_step(tmp_path):
    with make_client(tmp_path) as client:
        job_id = upload(client, auto_approve="true").json()["job_id"]
        assert wait_for(client, job_id, "completed")["deck_id"]


def test_no_web_evidence_goes_straight_to_generation(tmp_path):
    with make_client(tmp_path, sdk=make_sdk(n_citations=0)) as client:
        job_id = upload(client).json()["job_id"]
        assert wait_for(client, job_id, "completed")["deck_id"]
        assert client.get(f"/jobs/{job_id}/evidence", headers=H).json()["evidence"] == []


# --- failures --------------------------------------------------------------------------------


def test_failed_job_reports_a_friendly_error_and_can_be_retried(tmp_path):
    sdk = make_sdk()
    sdk.parse_stop[UnderstandResult] = "refusal"
    with make_client(tmp_path, sdk=sdk, delete_originals=False) as client:
        job_id = upload(client).json()["job_id"]
        job = wait_for(client, job_id, "failed")
        assert job["error"] == "The AI declined to process this content."
        assert job["stage"] == "understand"  # a failed job still says which step it was on

        sdk.parse_stop.clear()  # the cause goes away; the worker is idle, so this cannot race the retry
        assert client.post(f"/jobs/{job_id}/retry", headers=H).status_code == 200
        job = wait_for(client, job_id, "awaiting_approval")
        assert job["error"] is None and job["cost_usd"] > 0  # spend from the failed attempt is kept


def test_only_failed_jobs_can_be_retried(tmp_path):
    with make_client(tmp_path) as client:
        job_id = upload(client, auto_approve="true").json()["job_id"]
        wait_for(client, job_id, "completed")
        assert client.post(f"/jobs/{job_id}/retry", headers=H).status_code == 409
        assert client.post("/jobs/nope/retry", headers=H).status_code == 404


def test_cost_ceiling_stops_a_job_between_stages(tmp_path):
    with make_client(tmp_path, max_job_cost_usd=0.0) as client:
        job_id = upload(client).json()["job_id"]
        job = wait_for(client, job_id, "failed")
        assert "cost limit" in job["error"]
        assert job["cost_usd"] > 0  # spend up to the stop is still recorded


# --- library CRUD ------------------------------------------------------------------------


def test_decks_and_cards_crud(tmp_path):
    with make_client(tmp_path) as client:
        deck = client.post("/decks", headers=H, json={"name": "Chemistry"}).json()
        assert deck["counts"] == {"draft": 0, "approved": 0, "rejected": 0}
        assert client.post("/decks", headers=H, json={"name": ""}).status_code == 422
        renamed = client.patch(f"/decks/{deck['id']}", headers=H, json={"name": "Organic chem"}).json()
        assert renamed["name"] == "Organic chem"
        assert client.get("/decks/missing", headers=H).status_code == 404

        job_id = upload(client, auto_approve="true").json()["job_id"]
        job = wait_for(client, job_id, "completed")
        first = client.get(f"/jobs/{job_id}/cards", headers=H).json()[0]

        edited = client.patch(f"/cards/{first['id']}", headers=H, json={"front": "Edited?", "tags": ["x", "y"], "status": "approved"}).json()
        assert (edited["front"], edited["tags"], edited["status"]) == ("Edited?", ["x", "y"], "approved")
        assert client.patch(f"/cards/{first['id']}", headers=H, json={"status": "bogus"}).status_code == 422
        assert client.patch("/cards/missing", headers=H, json={"front": "x"}).status_code == 404

        assert client.delete(f"/cards/{first['id']}", headers=H).status_code == 204
        assert client.delete(f"/decks/{job['deck_id']}", headers=H).status_code == 204
        assert client.get(f"/decks/{job['deck_id']}/cards", headers=H).status_code == 404


def test_deleting_a_document_keeps_approved_cards_only(tmp_path):
    with make_client(tmp_path, delete_originals=False) as client:
        body = upload(client, auto_approve="true").json()
        job = wait_for(client, body["job_id"], "completed")
        cards = client.get(f"/jobs/{body['job_id']}/cards", headers=H).json()
        client.post("/cards/review", headers=H, json={"approve_ids": [cards[0]["id"]]})

        assert client.delete(f"/documents/{body['document_id']}", headers=H).status_code == 204
        assert client.delete(f"/documents/{body['document_id']}", headers=H).status_code == 404
        assert client.get(f"/jobs/{body['job_id']}", headers=H).status_code == 404
        assert not (tmp_path / "uploads" / body["document_id"]).exists()
        left = client.get(f"/decks/{job['deck_id']}/cards", headers=H).json()
        assert [c["id"] for c in left] == [cards[0]["id"]] and left[0]["job_id"] is None


# --- restart recovery ------------------------------------------------------------------------


def test_interrupted_jobs_are_requeued_on_startup(tmp_path):
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN)
    app = create_app(cfg, llm_factory=lambda: LLMClient(make_sdk(), cfg))
    store = app.state.store
    app.state.store.db.init()
    research = store.create(doc_id="d1", filename="a.md", kind="md", stored_names=["00_a.md"], auto_approve=False)
    claimed = store.claim_next()
    assert claimed["id"] == research and claimed["status"] == "researching"

    generating = store.create(doc_id="d2", filename="b.md", kind="md", stored_names=["00_b.md"], auto_approve=False)
    with store.db.tx() as c:
        c.execute("UPDATE jobs SET status = 'generating' WHERE id = ?", (generating,))

    assert store.recover_interrupted() == 2
    assert store.get(research)["status"] == "queued"
    assert store.get(generating)["status"] == "approved"
