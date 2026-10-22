"""Every test writes its query records to a scratch folder, never to the repository's records/."""

import pytest


@pytest.fixture(autouse=True)
def _records_in_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAKENET_RECORDS", str(tmp_path / "records"))


@pytest.fixture(autouse=True)
def _no_live_model_calls(monkeypatch):
    """Model tests must inject a client; never spend API credit from a pytest run."""
    from src.llm import chat

    def forbidden(*args, **kwargs):
        pytest.fail("Live model API calls are disabled in tests; inject a fake client")

    original_anthropic = chat._ask_anthropic
    def require_anthropic_client(client, *args, **kwargs):
        if client is None:
            forbidden()
        return original_anthropic(client, *args, **kwargs)

    monkeypatch.setattr(chat, "_ollama_request", forbidden)
    monkeypatch.setattr(chat, "_gemini_request", forbidden)
    monkeypatch.setattr(chat, "_ollama_available", lambda: (False, "Live model calls disabled in tests"))
    monkeypatch.setattr(chat, "_ask_anthropic", require_anthropic_client)
