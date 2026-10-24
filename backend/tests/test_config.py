import re
import tomllib
from pathlib import Path

import pytest

from app.core.config import Settings

BACKEND = Path(__file__).resolve().parent.parent


def test_defaults_keep_memory_use_modest(monkeypatch):
    for name in ("SNAPNOTES_JOB_CONCURRENCY", "SNAPNOTES_MAX_UPLOAD_MB"):
        monkeypatch.delenv(name, raising=False)
    cfg = Settings()
    assert cfg.job_concurrency == 1  # documents queue behind each other instead of competing for memory
    assert cfg.max_upload_bytes == 30 * 1024 * 1024


def test_both_limits_can_be_changed_from_the_environment(monkeypatch):
    monkeypatch.setenv("SNAPNOTES_JOB_CONCURRENCY", "3")
    monkeypatch.setenv("SNAPNOTES_MAX_UPLOAD_MB", "12.5")
    cfg = Settings()
    assert cfg.job_concurrency == 3
    assert cfg.max_upload_bytes == int(12.5 * 1024 * 1024)


def test_the_worker_really_runs_one_job_at_a_time(tmp_path):
    """Two uploads queued together: with concurrency 1 the second only starts after the first has finished."""
    import threading
    import time

    from fastapi.testclient import TestClient

    from app.llm.client import LLMClient
    from app.main import create_app
    from tests.test_api import H, TOKEN, make_sdk, upload

    running, peak, lock = 0, 0, threading.Lock()
    sdk = make_sdk(n_jobs=4)
    real_parse = sdk.beta.messages.parse

    def slow_parse(**kwargs):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.15)
        try:
            return real_parse(**kwargs)
        finally:
            with lock:
                running -= 1

    sdk.beta.messages.parse = slow_parse
    cfg = Settings(data_dir=tmp_path, api_token=TOKEN)
    assert cfg.job_concurrency == 1
    with TestClient(create_app(cfg, llm_factory=lambda: LLMClient(sdk, cfg))) as client:
        ids = [upload(client, auto_approve="true").json()["job_id"] for _ in range(2)]
        deadline = time.time() + 20
        while time.time() < deadline:
            states = [client.get(f"/jobs/{i}", headers=H).json()["status"] for i in ids]
            if all(s == "completed" for s in states):
                break
            time.sleep(0.05)
        assert states == ["completed", "completed"]
    assert peak == 1


@pytest.mark.skipif(
    not (BACKEND / "fly.toml").exists() or not (BACKEND / "DEPLOY.md").exists(),
    reason="deployment files are not in the Docker build context (docker build --target test)",
)
def test_fly_config_matches_what_the_docs_and_defaults_assume():
    fly = tomllib.loads((BACKEND / "fly.toml").read_text())
    memory = fly["vm"][0]["memory"]
    assert re.fullmatch(r"\d+gb", memory) and int(memory[:-2]) >= 1, "the machine needs at least 1 GB for large PDFs"
    assert "1 GB" in (BACKEND / "DEPLOY.md").read_text()
