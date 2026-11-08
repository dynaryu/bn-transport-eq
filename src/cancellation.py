"""Cooperative cancellation of a computation; ordinary CLI runs have no cancellation token."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class InferenceCancelled(Exception):
    """The user stopped this computation before its results were committed."""


_token: ContextVar[Event | None] = ContextVar("inference_cancellation", default=None)


def check_cancelled() -> None:
    event = _token.get()
    if event is not None and event.is_set():
        raise InferenceCancelled("Inference stopped. The previous session and results were kept.")


@contextmanager
def cancellable(event: Event):
    token = _token.set(event)
    try:
        check_cancelled()
        yield
    finally:
        _token.reset(token)
