"""Hardening from the security review: the token is checked before any body is read, Office files cannot unpack into a memory bomb,
the upload allowance cannot be reused by deleting jobs, and old originals are cleaned up."""
from __future__ import annotations

import asyncio
import io
import os
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from docx import Document

from app.core.authgate import AuthGateMiddleware
from app.core.config import Settings
from app.core.db import Database
from app.core.ratelimit import SlidingWindowLimiter
from app.ingest import loader
from app.ingest.loader import UnsupportedFileError, load_source, validate_file
from app.pipeline.cleanup import sweep_old_uploads
from app.store.jobs import JobStore
from tests.test_api import H, TOKEN, make_client, upload

# --- the token is checked before the body is read ----------------------------------------------


def run_asgi(app, headers: list[tuple[bytes, bytes]], path: str = "/documents"):
    """Drive an ASGI app by hand and report what it answered and whether it ever asked for the request body."""
    sent: list[dict] = []
    asked_for_body = False

    async def receive():
        nonlocal asked_for_body
        asked_for_body = True
        return {"type": "http.request", "body": b"x" * 10, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": path, "headers": headers}
    asyncio.run(app(scope, receive, send))
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    return status, asked_for_body, sent


async def body_reading_app(scope, receive, send):
    await receive()  # a route that reads its body
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def gate(token="secret-token-value", public=frozenset({"/health"})):
    return AuthGateMiddleware(body_reading_app, token=token, public_paths=public)


def test_an_anonymous_request_is_refused_without_reading_its_body():
    status, asked, sent = run_asgi(gate(), headers=[])
    assert status == 401 and not asked
    headers = dict(next(m for m in sent if m["type"] == "http.response.start")["headers"])
    assert headers[b"connection"] == b"close" and headers[b"www-authenticate"] == b"Bearer"


@pytest.mark.parametrize("header", [b"Bearer wrong", b"Bearer secret-token-valu", b"Basic secret-token-value", b"secret-token-value", b"Bearer "])
def test_a_wrong_token_is_refused_without_reading_its_body(header):
    status, asked, _ = run_asgi(gate(), headers=[(b"authorization", header)])
    assert status == 401 and not asked


def test_the_right_token_goes_through_to_the_route():
    status, asked, _ = run_asgi(gate(), headers=[(b"authorization", b"Bearer secret-token-value")])
    assert status == 200 and asked


def test_the_health_check_stays_open_and_a_server_with_no_token_leaves_the_decision_to_the_routes():
    assert run_asgi(gate(), headers=[], path="/health")[0] == 200
    assert run_asgi(gate(token=""), headers=[])[0] == 200


def test_end_to_end_an_anonymous_upload_gets_401_and_nothing_is_stored(tmp_path):
    with make_client(tmp_path) as client:
        r = client.post("/documents", files=[("files", ("n.md", b"# x\nhello", "text/markdown"))])
        assert r.status_code == 401
        assert not any(client.app.state.cfg.upload_dir.iterdir())
        assert upload(client).status_code == 202  # the same request with the token works


def test_wrong_tokens_still_count_against_the_rate_limit(tmp_path):
    with make_client(tmp_path, requests_per_minute=5) as client:
        codes = [client.get("/decks", headers={"Authorization": "Bearer wrong"}).status_code for _ in range(8)]
    assert codes[:5] == [401] * 5 and codes[5:] == [429] * 3


def test_docs_are_open_only_when_enabled(tmp_path):
    with make_client(tmp_path, docs_enabled=True) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
    with make_client(tmp_path, docs_enabled=False) as client:
        assert client.get("/docs").status_code == 401  # not a public path any more, so not even its absence is revealed


# --- Office files that unpack into far more than they weigh ------------------------------------


def zip_with(members: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return out.getvalue()


def real_docx(path: Path) -> Path:
    doc = Document()
    doc.add_heading("Cells", 1)
    doc.add_paragraph("Mitochondria make ATP.")
    doc.save(path)
    return path


def test_an_ordinary_docx_is_read_as_before(tmp_path):
    f = real_docx(tmp_path / "n.docx")
    validate_file(f)
    assert "Mitochondria make ATP." in load_source([f]).blocks[0]["text"]


def test_a_docx_whose_xml_unpacks_to_far_too_much_is_refused_before_anything_parses_it(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "MAX_XML_UNPACKED", 50_000)
    bomb = tmp_path / "bomb.docx"
    bomb.write_bytes(zip_with({"[Content_Types].xml": b"<Types/>", "word/document.xml": b"<w:document>" + b"<w:p/>" * 100_000 + b"</w:document>"}))
    assert bomb.stat().st_size < 10_000  # it is tiny on disk
    with pytest.raises(UnsupportedFileError, match="unpacks to far more"):
        validate_file(bomb)
    with pytest.raises(UnsupportedFileError):
        load_source([bomb])


def test_the_real_limits_stop_a_file_that_would_unpack_to_hundreds_of_megabytes(tmp_path):
    bomb = tmp_path / "bomb.pptx"
    bomb.write_bytes(zip_with({"[Content_Types].xml": b"<Types/>", "ppt/slides/slide1.xml": b"<p/>" * (30 * 1024 * 1024 // 4)}))
    assert bomb.stat().st_size < 1024 * 1024
    with pytest.raises(UnsupportedFileError, match="unpacks to far more"):
        validate_file(bomb)


def test_a_file_that_is_not_a_zip_and_a_zip_with_too_many_parts_are_refused(tmp_path, monkeypatch):
    junk = tmp_path / "junk.docx"
    junk.write_bytes(b"not a zip at all")
    with pytest.raises(UnsupportedFileError, match="not a valid Office file"):
        validate_file(junk)
    monkeypatch.setattr(loader, "MAX_ZIP_MEMBERS", 5)
    many = tmp_path / "many.docx"
    many.write_bytes(zip_with({f"part{i}.xml": b"<a/>" for i in range(10)}))
    with pytest.raises(UnsupportedFileError, match="too many parts"):
        validate_file(many)


def test_uploading_a_bomb_gets_a_clear_415_and_stores_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "MAX_XML_UNPACKED", 50_000)
    data = zip_with({"[Content_Types].xml": b"<Types/>", "word/document.xml": b"<w:document>" + b"<w:p/>" * 100_000 + b"</w:document>"})
    with make_client(tmp_path) as client:
        r = client.post("/documents", headers=H, files=[("files", ("bomb.docx", data, "application/octet-stream"))])
        assert r.status_code == 415 and "unpacks to far more" in r.json()["detail"]
        assert not any(client.app.state.cfg.upload_dir.iterdir())


# --- the hourly allowance, disk space and old originals ------------------------------------------


def test_deleting_a_job_does_not_give_its_allowance_back(tmp_path):
    with make_client(tmp_path, max_jobs_per_hour=2) as client:
        for _ in range(2):
            r = upload(client)
            assert r.status_code == 202
            assert client.delete(f"/documents/{r.json()['document_id']}", headers=H).status_code == 204
        assert client.app.state.store.count_since("2000-01-01") == 0  # the database has forgotten them
        assert upload(client).status_code == 429  # but the allowance has not


def test_a_failed_upload_does_not_use_up_the_allowance(tmp_path):
    with make_client(tmp_path, max_jobs_per_hour=2) as client:
        for _ in range(5):
            assert upload(client, name="notes.exe").status_code == 415
        assert upload(client).status_code == 202 and upload(client).status_code == 202


def test_an_almost_full_disk_refuses_an_upload_with_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.uploads.shutil.disk_usage", lambda _p: SimpleNamespace(total=100, used=99, free=1))
    with make_client(tmp_path) as client:
        r = upload(client)
        assert r.status_code == 507 and "out of storage" in r.json()["detail"]
        assert not any(client.app.state.cfg.upload_dir.iterdir())


def test_the_limiter_can_be_asked_without_being_counted():
    limiter = SlidingWindowLimiter(2, window_seconds=60, clock=lambda: 0.0)
    assert limiter.would_allow("k") and limiter.would_allow("k")  # asking does not use the allowance
    limiter.record("k")
    limiter.record("k")
    assert not limiter.would_allow("k") and not limiter.allow("k")


def test_old_originals_nobody_needs_are_removed_and_the_rest_kept(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init()
    store = JobStore(db)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    long_ago = time.time() - 40 * 86_400

    def folder(name: str, age: float) -> Path:
        d = uploads / name
        d.mkdir()
        (d / "00_notes.md").write_text("x")
        os.utime(d, (age, age))
        return d

    store.create(doc_id="busy", filename="b", kind="md", stored_names=["00_notes.md"], auto_approve=False)  # queued: in use
    store.create(doc_id="waiting", filename="w", kind="md", stored_names=["00_notes.md"], auto_approve=False)
    with db.tx() as c:
        c.execute("UPDATE jobs SET status = 'awaiting_approval' WHERE document_id = 'waiting'")
    folders = {"busy": folder("busy", long_ago), "waiting": folder("waiting", long_ago), "orphan": folder("orphan", long_ago), "fresh": folder("fresh", time.time())}

    assert sweep_old_uploads(uploads, store, ttl_days=14) == 2
    assert folders["busy"].exists() and folders["fresh"].exists()
    assert not folders["waiting"].exists() and not folders["orphan"].exists()
    assert sweep_old_uploads(uploads, store, ttl_days=0) == 0  # zero switches the sweep off


def test_a_job_whose_originals_were_removed_fails_with_a_clear_reason(tmp_path):
    with pytest.raises(UnsupportedFileError, match="no longer on the server"):
        load_source([tmp_path / "00_gone.md"])


def test_the_gate_is_wired_in_between_the_rate_limit_and_the_body_limit(tmp_path):
    from app.core.bodylimit import BodyLimitMiddleware
    from app.main import create_app
    from starlette.middleware.base import BaseHTTPMiddleware

    app = create_app(Settings(data_dir=tmp_path, api_token=TOKEN))
    order = [m.cls for m in app.user_middleware]  # outermost first
    assert order.index(BaseHTTPMiddleware) < order.index(AuthGateMiddleware) < order.index(BodyLimitMiddleware)
