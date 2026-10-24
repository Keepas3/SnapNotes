"""Reject oversized request bodies before the application sees them.

Without this, FastAPI receives and parses a whole multipart upload before the handler can say "too big", so a 5 GB
request would be fully read (and spooled to disk) first. Here the limit applies at the edge:

- a request that *declares* a Content-Length over the limit is answered with 413 immediately, without reading the body;
- a body that arrives without a Content-Length (chunked), or with a Content-Length that understates it, is counted as
  it streams in and cut off the moment it passes the limit.

This is a pure ASGI middleware (not Starlette's BaseHTTPMiddleware) because it has to sit in front of the body stream.
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class BodyLimitMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]], *, upload_path: str, upload_limit: int, default_limit: int):
        self.app = app
        self.upload_path = upload_path
        self.upload_limit = upload_limit
        self.default_limit = default_limit

    def limit_for(self, scope: Scope) -> int:
        is_upload = scope["method"] == "POST" and scope["path"].rstrip("/") == self.upload_path
        return self.upload_limit if is_upload else self.default_limit

    @staticmethod
    async def _reject(send: Send, limit: int) -> None:
        body = json.dumps({"detail": f"Request body too large (the limit here is {limit // 1024} KB)."}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),  # the unread body makes this connection unusable for another request
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan, websocket
            await self.app(scope, receive, send)
            return

        limit = self.limit_for(scope)
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break  # a malformed header is the server's problem to reject, not ours
                if declared > limit:
                    await self._reject(send, limit)
                    return
                break

        received = 0
        rejected = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    rejected = True
                    if not response_started:
                        await self._reject(send, limit)
                    return {"type": "http.disconnect"}  # the application sees a client that went away
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if rejected:
                return  # we have already answered; anything the application still tries to send is dropped
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not rejected:
                raise
            # The application tripped over the "disconnect" we gave it after rejecting. The client already has its 413.
