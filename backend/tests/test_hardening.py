"""Minimum token strength and a rate limiter whose memory cannot grow without bound."""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from app.core.auth import MIN_TOKEN_LENGTH, token_problem
from app.core.config import Settings
from app.core.ratelimit import SlidingWindowLimiter
from app.main import create_app

STRONG = "x" * MIN_TOKEN_LENGTH


def cfg(**kw) -> Settings:
    base = dict(api_token=STRONG, allow_no_auth=False, allow_weak_token=False, demo_mode=False)
    return dataclasses.replace(Settings(), **{**base, **kw})


# --- token strength ----------------------------------------------------------------------------


def test_a_long_enough_token_is_accepted():
    assert token_problem(cfg()) is None
    assert token_problem(cfg(api_token="y" * 80)) is None


@pytest.mark.parametrize("weak", ["dev-token", "password", "1234", "x" * (MIN_TOKEN_LENGTH - 1)])
def test_a_short_token_is_refused_with_a_way_to_make_a_good_one(weak):
    problem = token_problem(cfg(api_token=weak))
    assert problem and f"only {len(weak)} characters" in problem
    assert "token_urlsafe(32)" in problem and "SNAPNOTES_ALLOW_WEAK_TOKEN" in problem


def test_demo_mode_and_the_explicit_opt_out_allow_a_short_token():
    assert token_problem(cfg(api_token="dev-token", demo_mode=True)) is None  # canned data, no Claude spend
    assert token_problem(cfg(api_token="dev-token", allow_weak_token=True)) is None


def test_no_token_at_all_is_still_its_own_error_unless_auth_is_switched_off():
    assert "SNAPNOTES_TOKEN is not set" in token_problem(cfg(api_token=""))
    assert token_problem(cfg(api_token="", allow_no_auth=True)) is None


def test_the_server_will_not_start_with_a_guessable_token(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, api_token="letmein", demo_mode=False, allow_weak_token=False))
    with pytest.raises(RuntimeError, match="only 7 characters"):
        with TestClient(app):
            pass


def test_the_server_starts_with_a_generated_token(tmp_path):
    import secrets

    token = secrets.token_urlsafe(32)
    with TestClient(create_app(Settings(data_dir=tmp_path, api_token=token))) as client:
        assert client.get("/jobs", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_whitespace_around_a_token_from_the_environment_is_ignored(monkeypatch):
    monkeypatch.setenv("SNAPNOTES_TOKEN", "  " + STRONG + "\r\n")  # a secret pasted with a trailing newline
    assert Settings().api_token == STRONG


# --- rate limiter -------------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_the_limiter_still_limits_per_key_and_recovers_after_the_window():
    clock = Clock()
    limiter = SlidingWindowLimiter(limit=3, window_seconds=60, clock=clock)
    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]
    assert limiter.allow("b") is True  # another client is unaffected
    clock.t = 61
    assert limiter.allow("a") is True


def test_a_flood_of_made_up_clients_cannot_grow_memory_without_limit():
    limiter = SlidingWindowLimiter(limit=5, window_seconds=60, clock=Clock(), max_keys=100)
    for i in range(50_000):
        assert limiter.allow(f"spoofed-{i}") is True
    assert len(limiter) <= 100


def test_when_full_the_least_recently_seen_client_is_the_one_forgotten():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=60, clock=Clock(), max_keys=3)
    for key in ("a", "b", "c"):
        limiter.allow(key)
    limiter.allow("a")  # touching "a" makes "b" the stalest (and "a" is over its limit, so it stays tracked)
    limiter.allow("d")  # needs room: drops "b"
    assert limiter.allow("a") is False  # still remembered and still limited
    assert limiter.allow("b") is True  # forgotten, so it starts fresh


def test_expired_clients_are_swept_without_anyone_asking():
    clock = Clock()
    limiter = SlidingWindowLimiter(limit=5, window_seconds=60, clock=clock, max_keys=10_000)
    for i in range(500):
        limiter.allow(f"old-{i}")
    clock.t = 120  # all of them expired
    for i in range(SlidingWindowLimiter.SWEEP_EVERY):
        limiter.allow("busy")  # ordinary traffic from one client triggers the periodic sweep
    assert len(limiter) < 5


def test_the_client_address_header_is_configurable_and_can_be_ignored(tmp_path):
    """Behind Fly the proxy's header names the client; anywhere else it must be ignorable so it cannot be faked."""
    def run(header):
        c = Settings(data_dir=tmp_path / (header or "none"), api_token=STRONG, requests_per_minute=2, client_ip_header=header)
        with TestClient(create_app(c)) as client:
            auth = {"Authorization": f"Bearer {STRONG}"}
            # three requests that all *claim* to come from different clients
            return [client.get("/jobs", headers={**auth, "Fly-Client-IP": f"10.0.0.{i}"}).status_code for i in range(3)]

    assert run("fly-client-ip") == [200, 200, 200]  # trusted: three apparent clients, none over its limit
    assert run("") == [200, 200, 429]  # ignored: they are all the one real client, so the limit applies
