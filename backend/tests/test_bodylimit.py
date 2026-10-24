"""The request size cutoff: oversized bodies are refused at the edge, before the application reads them."""
import json

import pytest
from fastapi.testclient import TestClient

from app.core.bodylimit import BodyLimitMiddleware
from app.core.config import Settings
from app.main import create_app
from tests.test_api import H, TOKEN, upload
from tests.test_sync import DEVICE, Env, log

LIMIT = 1000
UPLOAD_LIMIT = 5000


class Recorder:
    """A stand-in application that reads the whole request body and reports what happened."""

    def __init__(self):
        self.called = False
        self.body = b""
        self.saw_disconnect = False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        self.called = True
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                self.saw_disconnect = True
                raise RuntimeError("client disconnected")  # what real frameworks do; must not become a 500
            self.body += message.get("body", b"")
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})


async def call(app, *, method="POST", path="/x", headers=(), chunks=(b"",)):
    """Drive an ASGI app by hand; returns (sent messages, application's received body)."""
    sent = []
    queue = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": method, "path": path, "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}
    await app(scope, receive, send)
    return sent


def status_of(sent):
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def middleware(inner):
    return BodyLimitMiddleware(inner, upload_path="/documents", upload_limit=UPLOAD_LIMIT, default_limit=LIMIT)


@pytest.mark.anyio
async def test_a_body_within_the_limit_passes_through_untouched():
    inner = Recorder()
    sent = await call(middleware(inner), headers=[("content-length", "600")], chunks=(b"a" * 300, b"b" * 300))
    assert status_of(sent) == 200 and inner.body == b"a" * 300 + b"b" * 300


@pytest.mark.anyio
async def test_a_declared_length_over_the_limit_is_refused_without_reading_anything():
    inner = Recorder()
    sent = await call(middleware(inner), headers=[("content-length", str(LIMIT + 1))], chunks=(b"x" * 10,))
    assert status_of(sent) == 413
    assert inner.called is False  # the application was never even started
    body = json.loads(next(m["body"] for m in sent if m["type"] == "http.response.body"))
    assert "too large" in body["detail"]
    headers = dict(next(m["headers"] for m in sent if m["type"] == "http.response.start"))
    assert headers[b"connection"] == b"close"


@pytest.mark.anyio
async def test_a_chunked_body_is_cut_off_as_soon_as_it_passes_the_limit():
    inner = Recorder()
    chunks = tuple(b"x" * 400 for _ in range(50))  # 20 KB arriving with no Content-Length at all
    sent = await call(middleware(inner), chunks=chunks)
    assert status_of(sent) == 413
    assert len(inner.body) <= LIMIT + 400  # it never got to read more than a chunk beyond the limit
    assert inner.saw_disconnect  # the application was told the client went away...
    assert sum(1 for m in sent if m["type"] == "http.response.start") == 1  # ...and its own failure did not add a second response


@pytest.mark.anyio
async def test_a_content_length_that_understates_the_body_is_caught_too():
    inner = Recorder()
    sent = await call(middleware(inner), headers=[("content-length", "10")], chunks=tuple(b"x" * 400 for _ in range(10)))
    assert status_of(sent) == 413


@pytest.mark.anyio
async def test_uploads_get_the_larger_limit_and_nothing_else_does():
    declared = str(LIMIT * 3)  # over the normal limit, under the upload limit
    sent = await call(middleware(Recorder()), path="/documents", headers=[("content-length", declared)], chunks=(b"x",))
    assert status_of(sent) == 200
    sent = await call(middleware(Recorder()), path="/sync", headers=[("content-length", declared)], chunks=(b"x",))
    assert status_of(sent) == 413
    sent = await call(middleware(Recorder()), path="/documents", method="GET", headers=[("content-length", declared)], chunks=(b"x",))
    assert status_of(sent) == 413  # only POST uploads are special
    sent = await call(middleware(Recorder()), path="/documents", headers=[("content-length", str(UPLOAD_LIMIT + 1))], chunks=(b"x",))
    assert status_of(sent) == 413


@pytest.mark.anyio
async def test_requests_without_a_body_and_odd_headers_are_left_alone():
    assert status_of(await call(middleware(Recorder()), method="GET", path="/health", chunks=(b"",))) == 200
    assert status_of(await call(middleware(Recorder()), headers=[("content-length", "not-a-number")], chunks=(b"hi",))) == 200


@pytest.mark.anyio
async def test_non_http_traffic_such_as_lifespan_passes_straight_through():
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    await middleware(inner)({"type": "lifespan"}, None, None)
    assert seen == ["lifespan"]


# --- wired into the real app ---------------------------------------------------------------------


def app_client(tmp_path, **overrides) -> TestClient:
    return TestClient(create_app(Settings(data_dir=tmp_path, api_token=TOKEN, demo_mode=True, demo_delay=0, **overrides)))


def test_an_oversized_upload_is_refused_before_the_upload_code_runs(tmp_path):
    with app_client(tmp_path, max_upload_bytes=50_000) as client:
        res = client.post("/documents", headers=H, files=[("files", ("big.txt", b"x" * 2_000_000, "text/plain"))])
        assert res.status_code == 413 and "too large" in res.json()["detail"]
        # the handler never ran: no upload folder was even started, and no job exists
        assert list((tmp_path / "uploads").iterdir()) == []
        assert client.get("/jobs", headers=H).json() == []
        # and a normal upload still works afterwards
        assert upload(client, "ok.txt", b"hello").status_code == 202


def test_the_handlers_own_check_still_guards_the_total_of_many_files(tmp_path):
    """Each file is small and the whole request is under the edge limit, but the files together exceed the cap."""
    with app_client(tmp_path, max_upload_bytes=1000, max_files_per_upload=5) as client:
        files = [("files", (f"n{i}.txt", b"x" * 400, "text/plain")) for i in range(3)]  # 1200 bytes of files
        assert client.post("/documents", headers=H, files=files).status_code == 413


def test_json_endpoints_have_a_small_body_limit_but_a_full_sync_fits(tmp_path):
    with Env(tmp_path) as env:
        card = env.card(env.deck())
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        batch = [log(card, now, log_id=f"batch-log-{i:04d}") for i in range(500)]  # the most one request may carry
        res = env.client.post("/sync", headers=H, json={"device_id": DEVICE, "cursor": 0, "logs": batch})
        assert res.status_code == 200, res.status_code  # a maximum-size sync is nowhere near the limit

    with app_client(tmp_path / "small", max_body_kb=4) as client:
        res = client.post("/sync", headers=H, json={"device_id": DEVICE, "cursor": 0, "logs": [], "padding": "x" * 10_000})
        assert res.status_code == 413


def test_a_refused_request_still_carries_cors_headers_for_the_browser_preview(tmp_path):
    origin = "http://localhost:8081"
    with app_client(tmp_path, max_body_kb=1, cors_origins=(origin,)) as client:
        res = client.post("/sync", headers={**H, "Origin": origin}, json={"device_id": DEVICE, "padding": "x" * 5000})
        assert res.status_code == 413 and res.headers["access-control-allow-origin"] == origin
