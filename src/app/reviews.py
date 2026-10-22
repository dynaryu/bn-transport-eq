"""Optional per-query read-back approval. Tokens refer to server-held query snapshots."""
import secrets
import threading

from src.inference.engine import QueryError, readback

_LOCK = threading.Lock()
_PENDING = {}


def wait_for_review(query, send):
    token = secrets.token_urlsafe(24)
    item = {"event": threading.Event(), "approved": False}
    with _LOCK:
        _PENDING[token] = item
    try:
        send({"type": "review", "token": token, "readback": readback(query)})
        return item["event"].wait(timeout=300) and item["approved"]
    finally:
        with _LOCK:
            _PENDING.pop(token, None)


def decide(token, approved):
    if not isinstance(approved, bool):
        raise QueryError("Review decision must be true or false")
    with _LOCK:
        item = _PENDING.get(token)
        if item is None or item["event"].is_set():
            raise QueryError("Review expired or already answered")
        item["approved"] = approved
        item["event"].set()
