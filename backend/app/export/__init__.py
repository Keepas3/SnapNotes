"""Deck exports: Anki packages (.apkg) and CSV."""
from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import quote, urlparse


# Characters that may appear in a URL as they are (RFC 3986 unreserved + reserved, plus "%" so existing escapes survive).
# Everything else (quotes, angle brackets, spaces, backticks, control characters, non-ASCII...) gets percent-encoded.
_URL_SAFE = ":/?#[]@!$&'()*+,;=%-._~"


def safe_http_url(url: str) -> str | None:
    """A clickable form of a web evidence URL, or None if it must not be linked.

    URLs come from web search results, i.e. from pages we do not control. Only plain http(s) URLs qualify, and the
    result is percent-encoded so that no character in it can end an HTML attribute or start a tag, whatever the
    context it is placed in."""
    url = url.strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return quote(url, safe=_URL_SAFE)


def domain_of(url: str) -> str:
    return (urlparse(url).hostname or url).removeprefix("www.")


def html_text(text: str) -> str:
    """Plain text as an HTML field value (Anki fields are HTML): escape markup and keep line breaks."""
    return html.escape(text, quote=False).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def html_attr(text: str) -> str:
    """Plain text as the value of a double-quoted HTML attribute (unlike `html_text`, this escapes quotes too)."""
    return html.escape(text, quote=True)


def source_label(card: dict[str, Any]) -> str:
    """Where a card came from, as plain text (used by the CSV): a page or heading in the notes, or the web page URL."""
    return card["source_ref"]


def safe_filename_stem(name: str, fallback: str = "deck") -> str:
    stem = re.sub(r"[^\w .-]+", "_", name, flags=re.UNICODE).strip(" ._") or fallback
    return stem[:80]


def content_disposition(stem: str, extension: str) -> str:
    """`attachment` header that survives non-ASCII deck names (RFC 6266 / 5987)."""
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "deck"
    return f"attachment; filename=\"{ascii_name}.{extension}\"; filename*=UTF-8''{quote(stem + '.' + extension)}"
