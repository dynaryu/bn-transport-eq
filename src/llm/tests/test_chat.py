"""The model chooses the query, the engine computes it, and the order of events shows that."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.inference import engine
from src.llm import chat

EMA = "ema-highway/gmm"
QUERY = {"events": [{"node": "EdgeOpen::e0001", "op": "==", "value": "0"}], "samples": "500"}


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setenv("BN_LLM", "ollama")


def ollama_script(*replies):
    """A stand-in for Ollama's /api/chat that plays back replies and records requests."""
    sent = []

    def post(path, payload):
        sent.append(json.loads(json.dumps(payload)))
        return replies[len(sent) - 1]

    return post, sent


def tool_call(arguments, name=chat.TOOL_NAME):
    return {"message": {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}}


def final(text):
    return {"message": {"content": text}}


def test_the_query_is_stated_before_it_is_computed_and_the_answer_comes_last():
    post, sent = ollama_script(tool_call(QUERY), final("About a quarter."))
    events = []
    reply = chat.ask(EMA, "How likely is edge e01 to close?", client=post, model="m", on_event=events.append)
    assert [e["type"] for e in events] == ["query", "computed"]
    assert events[0]["statement"].startswith("P(EdgeOpen[e0001] = 0)")
    assert reply["answer"] == "About a quarter."
    assert reply["result"]["events"][0]["node"] == "EdgeOpen::e0001"
    tool_message = sent[1]["messages"][-1]
    computed = reply["result"]["events"][0]["p"]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"])["events"][0]["p"] == float(f"{computed:.4g}")   # the engine's number


def test_arguments_from_a_small_model_are_coerced():
    q = chat.coerce_arguments(json.dumps({"targets": "SA(1.0)::*", "samples": "300.0",
                                          "events": {"node": "TravelTime::n1>n22", "op": "<", "value": "60"}}), EMA)
    assert q == {"assembly": EMA, "targets": ["SA(1.0)::*"], "samples": 300,
                 "events": [{"node": "TravelTime::n1>n22", "op": "<", "value": 60}]}
    with pytest.raises(ValueError, match="unknown argument"):
        chat.coerce_arguments({"magnitude": 7}, EMA)


def test_page_settings_apply_unless_the_model_sets_its_own():
    defaults = {"rows": {"ema-reference": {"mw": 7.0}}}
    assert chat.coerce_arguments({"targets": ["SA(1.0)::*"]}, EMA, defaults)["rows"] == defaults["rows"]
    own = {"ema-reference": {"mw": 6.0}}
    assert chat.coerce_arguments({"targets": ["SA(1.0)::*"], "rows": own}, EMA, defaults)["rows"] == own


def test_evidence_is_refused_by_the_engine_and_the_refusal_reaches_the_model():
    post, sent = ollama_script(tool_call({"targets": ["Usable::*"], "evidence": {"EdgeOpen::e0001": 0}}),
                               final("That needs backward inference, which is not available."))
    events = []
    reply = chat.ask(EMA, "Which bridge failed, given e0001 is closed?", client=post, model="m", on_event=events.append)
    assert [e["type"] for e in events] == ["query", "refused"]
    assert json.loads(sent[1]["messages"][-1]["content"])["refused"] is True
    assert reply["result"] is None and "forward inference only" in reply["runs"][0]["error"]


def test_a_tool_that_does_not_exist_is_reported_to_the_model():
    post, sent = ollama_script(tool_call({}, name="guess"), final("ok"))
    chat.ask(EMA, "q", client=post, model="m")
    assert "no tool named 'guess'" in sent[1]["messages"][-1]["content"]


def test_a_model_that_never_stops_calling_tools_is_cut_off(monkeypatch):
    monkeypatch.setattr(chat, "MAX_TOOL_ROUNDS", 2)
    post, _ = ollama_script(tool_call(QUERY), tool_call(QUERY))
    assert "Stopped after 2 rounds" in chat.ask(EMA, "q", client=post, model="m")["answer"]


def test_the_prompt_describes_the_actual_bn():
    prompt = chat.system_prompt(engine.describe("ema-highway/gmm"))
    assert "EdgeOpen on Edge (129 nodes" in prompt
    assert '"ema-reference": {"mw"' in prompt
    assert "forward-only" in prompt and "Never estimate" in prompt


def test_the_anthropic_loop_runs_the_tool_and_returns_the_text(monkeypatch):
    monkeypatch.setenv("BN_LLM", "anthropic")
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append({**kwargs, "messages": list(kwargs["messages"])})   # the loop keeps appending
            if len(calls) == 1:
                return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="t1", name=chat.TOOL_NAME,
                                                                input={"targets": ["SA(1.0)::bridge-e0001"]})])
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="About 0.07 g.")])

    reply = chat.ask(EMA, "What is SA(1.0) at bridge e0001?", client=SimpleNamespace(messages=Messages()), model="m")
    assert reply["answer"] == "About 0.07 g."
    assert calls[0]["tools"][0]["name"] == chat.TOOL_NAME and "input_schema" in calls[0]["tools"][0]
    result_block = calls[1]["messages"][-1]["content"][0]
    assert result_block["type"] == "tool_result" and result_block["tool_use_id"] == "t1"


def test_an_empty_question_is_refused():
    with pytest.raises(chat.ChatUnavailable, match="Ask a question"):
        chat.ask(EMA, "  ", client=lambda *a: {})
