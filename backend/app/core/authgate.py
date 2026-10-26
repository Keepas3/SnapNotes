"""Check the access token before the application reads any request body.

FastAPI reads and parses a request body (a 30 MB multipart upload, say) before it runs the route's dependencies, which is where
`require_auth` lives. So an anonymous client could make the server receive and spool whole uploads, and only then be told 401. This
pure ASGI middleware looks at the Authorization header first and answers 401 straight away, without reading a byte of the body.
`require_auth` stays on every route as a second check.
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from app.core.auth import bearer_ok

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class AuthGateMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]], *, token: str, public_paths: frozenset[str]):
        self.app = app
        self.token = token
        self.public_paths = public_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # With no token configured the routes decide (503, or open when no-auth was asked for), as before.
        if scope["type"] != "http" or not self.token or scope["path"] in self.public_paths:
            await self.app(scope, receive, send)
            return
        header = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
        if bearer_ok(header, self.token):
            await self.app(scope, receive, send)
            return
        body = json.dumps({"detail": "Invalid or missing token"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                    (b"connection", b"close"),  # the body we did not read makes this connection unusable for another request
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
