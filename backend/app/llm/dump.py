"""Saving raw model responses for debugging, when SNAPNOTES_DEBUG_DUMP names a folder (off by default)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any


def jsonable(obj: Any) -> Any:
    """An SDK response as plain JSON."""
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return json.loads(json.dumps(obj, default=lambda o: dict(vars(o)) if hasattr(o, "__dict__") else repr(o)))


def write_dump(folder: Path | None, name: str, number: int, resp: Any, log: logging.Logger) -> None:
    """Save one response as `<time>-<name>-<number>.json`. It must never break a job, so any failure is only logged."""
    if folder is None:
        return
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}-{number}.json"
        path.write_text(json.dumps(jsonable(resp), indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        log.exception("could not write the debug dump to %s", folder)
