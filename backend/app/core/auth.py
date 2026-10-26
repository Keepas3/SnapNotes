import hmac

from fastapi import HTTPException, Request

from app.core.config import Settings

MIN_TOKEN_LENGTH = 24  # token_urlsafe(32), the generator suggested in the docs, gives 43


def token_problem(cfg: Settings) -> str | None:
    """Why the server must not start with this configuration, or None.

    The access token is the only thing between the internet and the library and the Claude bill, so a guessable one
    is refused. Demo mode (canned responses, no spend) and an explicit opt-out are for local use."""
    if not cfg.api_token:
        if cfg.allow_no_auth:
            return None
        return "SNAPNOTES_TOKEN is not set. Set it to a long random string (or SNAPNOTES_ALLOW_NO_AUTH=1 for local dev)."
    if len(cfg.api_token) >= MIN_TOKEN_LENGTH or cfg.demo_mode or cfg.allow_weak_token:
        return None
    return (
        f"SNAPNOTES_TOKEN is only {len(cfg.api_token)} characters; use at least {MIN_TOKEN_LENGTH}. "
        'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))" '
        "(for purely local use you can set SNAPNOTES_ALLOW_WEAK_TOKEN=1)."
    )


def bearer_ok(authorization: str, token: str) -> bool:
    """True when an Authorization header carries exactly this bearer token. Constant-time comparison."""
    scheme, _, supplied = authorization.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(supplied.encode(), token.encode())


def require_auth(request: Request) -> None:
    """Single personal bearer token. Constant-time comparison; no token configured means refuse, not allow."""
    cfg = request.app.state.cfg
    if not cfg.api_token:
        if cfg.allow_no_auth:  # explicit local-development escape hatch
            return
        raise HTTPException(status_code=503, detail="Server token is not configured")

    if not bearer_ok(request.headers.get("authorization", ""), cfg.api_token):
        raise HTTPException(status_code=401, detail="Invalid or missing token", headers={"WWW-Authenticate": "Bearer"})
