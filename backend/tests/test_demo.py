from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.test_api import H, TOKEN, upload, wait_for

ORIGIN = "http://localhost:8081"


def demo_client(tmp_path, **overrides) -> TestClient:
    return TestClient(create_app(Settings(data_dir=tmp_path, api_token=TOKEN, demo_mode=True, demo_delay=0, **overrides)))


def test_demo_mode_runs_the_real_pipeline_without_claude(tmp_path):
    with demo_client(tmp_path) as client:
        job_id = upload(client, "n.md", b"# Cell biology - week 3\nMitochondria make ATP.").json()["job_id"]

        job = wait_for(client, job_id, "awaiting_approval")
        assert job["understanding"]["main_topic"] == "Cell biology - week 3"
        evidence = client.get(f"/jobs/{job_id}/evidence", headers=H).json()["evidence"]
        assert len(evidence) == 3 and all(e["url"].startswith("https://") for e in evidence)

        client.post(f"/jobs/{job_id}/approve", headers=H, json={"rejected_ids": [evidence[2]["id"]]})
        job = wait_for(client, job_id, "completed")
        assert job["cost_usd"] > 0

        cards = client.get(f"/jobs/{job_id}/cards", headers=H).json()
        by_status = {s: [c for c in cards if c["status"] == s] for s in ("draft", "rejected")}
        assert len(cards) == 8 and len(by_status["draft"]) == 6
        # one near-duplicate and one ungrounded card start out rejected, so the review UI has something to show
        assert {c["issue"] or "duplicate" for c in by_status["rejected"]} == {"duplicate", "Not supported by the notes or the evidence"}
        assert {c["type"] for c in cards} == {"basic", "cloze", "definition"}
        assert {c["origin"] for c in cards} == {"notes", "web"}


def test_cors_is_off_by_default_and_exact_when_enabled(tmp_path):
    with demo_client(tmp_path) as client:
        res = client.get("/health", headers={"Origin": ORIGIN})
        assert "access-control-allow-origin" not in res.headers

    with demo_client(tmp_path, cors_origins=(ORIGIN,)) as client:
        pre = client.options("/jobs", headers={
            "Origin": ORIGIN, "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization",
        })
        assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == ORIGIN

        rejected = client.get("/jobs", headers={"Origin": ORIGIN})  # no token
        assert rejected.status_code == 401 and rejected.headers["access-control-allow-origin"] == ORIGIN

        other = client.get("/jobs", headers={**H, "Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in other.headers
