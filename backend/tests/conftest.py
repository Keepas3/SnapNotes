"""Test doubles for the Anthropic SDK, so the whole pipeline is testable offline."""
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"  # the app runs on asyncio; do not also run the async tests under another event loop


def usage(inp=100, out=50, cache_read=0, cache_write=0, searches=0):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        server_tool_use=SimpleNamespace(web_search_requests=searches),
    )


def text_block(text: str, citations: list | None = None):
    return SimpleNamespace(type="text", text=text, citations=citations or [])


def citation(url: str, title: str, cited_text: str):
    return SimpleNamespace(type="web_search_result_location", url=url, title=title, cited_text=cited_text)


def search_result_block():
    return SimpleNamespace(type="web_search_tool_result", content=[SimpleNamespace(type="web_search_result")])


def response(content: list, stop_reason: str = "end_turn", **kw):
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=kw.pop("usage", usage()), stop_details=None, **kw)


class FakeNamespace:
    def __init__(self, tag: str, sdk: "FakeSDK"):
        self.tag = tag
        self.sdk = sdk

    def parse(self, **kwargs: Any):
        self.sdk.calls.append((self.tag, "parse", kwargs))
        value = self.sdk.parse_results[kwargs["output_format"]]
        stop = self.sdk.parse_stop.get(kwargs["output_format"], "end_turn")
        return SimpleNamespace(parsed_output=value, usage=usage(), stop_reason=stop, content=[], stop_details=None)

    def create(self, **kwargs: Any):
        self.sdk.calls.append((self.tag, "create", kwargs))
        return self.sdk.create_responses.pop(0)


class FakeSDK:
    """`messages` and `beta.messages` are separate so tests can assert which route was used."""

    def __init__(self, parse_results: dict | None = None, create_responses: list | None = None):
        self.parse_results = parse_results or {}
        self.parse_stop: dict = {}
        self.create_responses = create_responses or []
        self.calls: list[tuple[str, str, dict]] = []
        self.messages = FakeNamespace("plain", self)
        self.beta = SimpleNamespace(messages=FakeNamespace("beta", self))
