"""Translation fixtures are executable contracts, and evaluation examples stay out of prompts."""
import json
import pytest

from src.llm import cases, chat
from src.inference.engine import describe


@pytest.fixture(autouse=True)
def no_live_backend_status(monkeypatch):
    monkeypatch.setattr(chat, "available", lambda: (True, ""))


def test_cases_validate_and_holdout_is_not_fewshot():
    report = cases.check_contracts()
    assert report["contracts"] >= 8
    prompt = chat.system_prompt(describe("toy-ca/demo"))
    assert "What are n4's failure probabilities before and after changing b2 to HWB11" in prompt
    assert "Compare n4 failure before and after retrofitting b2, b4 and b6 together" not in prompt


def test_defaults_merge_without_losing_the_hazard_when_retrofitting():
    q = chat.coerce_arguments({"events": [{"node": "SystemState::n4", "op": "==", "value": 0}],
                              "rows": {"b2": {"hazus_class": "HWB11"}}}, "toy-ca/demo",
                             {"rows": {"demo-san-andreas": {"mw": 7}}})
    assert q["rows"] == {"demo-san-andreas": {"mw": 7}, "b2": {"hazus_class": "HWB11"}}


def test_declined_readback_never_runs_engine(monkeypatch):
    monkeypatch.setattr(chat.engine, "answer", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not sample")))
    events, runs = [], []
    reply = json.loads(chat.run_tool({"targets": ["Usable::b2"]}, assembly="toy-ca/demo", defaults=None,
                                    runs=runs, on_event=events.append, review=lambda q: False))
    assert reply["refused"] and "cancelled" in reply["error"]
    assert any(e.get("readback") for e in events)


def test_translate_only_does_not_write_proposals_or_run_engine(monkeypatch):
    from src import proposals
    monkeypatch.setattr(proposals, "draft", lambda *a: (_ for _ in ()).throw(AssertionError("must not draft")))
    reply = json.loads(chat.run_tool({"proposal": {}}, assembly="toy-ca/demo", defaults=None, runs=[], translate_only=True))
    assert reply["translation_only"]


def test_heldout_runner_scores_plans_without_running_engine(monkeypatch):
    monkeypatch.setenv("BN_LLM", "ollama")
    expected = {question: case["expected"]["tool_arguments"] for case in cases.load_cases()
                if case["split"] == "held_out" for question in case["questions"]}
    monkeypatch.setattr(chat.engine, "answer", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not sample")))
    def client(path, payload):
        if payload["messages"][-1]["role"] == "tool":
            return {"message": {"content": "Translation recorded."}}
        question = next(m["content"] for m in reversed(payload["messages"]) if m["role"] == "user")
        return {"message": {"tool_calls": [{"function": {"name": "run_query", "arguments": args}}
                                            for args in expected[question]]}}
    result = cases.evaluate(client=client, model="fixture")
    assert result["passed"] == result["total"] == len(expected)
    from src.inference import records
    followup = next(c for c in cases.load_cases() if c["id"] == "prep-importance-selected-probabilities")
    saved = next(r for r in records.read("ask") if r["question"] == followup["questions"][0])
    assert saved["history"] == followup["context"]["history"]


def test_analysis_tool_records_calculation_and_its_query_provenance():
    from src.inference import records
    case = next(c for c in cases.load_cases() if c["id"] == "emergency-inspection-voi")
    args = case["expected"]["tool_arguments"][0]
    args["analysis"]["query"]["samples"] = 1000
    events, runs = [], []
    result = json.loads(chat.run_tool(args, assembly="toy-ca/demo", defaults=None, runs=runs,
                                     on_event=events.append, record={"source": "test"}))
    assert result["kind"] == "voi" and result["loss_units"] == "demo loss units"
    assert any(e["type"] == "analysis_plan" for e in events)
    record = records.read("analysis")[0]
    assert record["queries"] == [r["result"]["record"]["id"] for r in runs]


@pytest.mark.parametrize("provider", ["ollama", "gemini", "anthropic"])
def test_clarification_stops_the_turn_before_other_batched_queries(monkeypatch, provider):
    from types import SimpleNamespace
    from src.inference import records
    monkeypatch.setenv("BN_LLM", provider)
    monkeypatch.setattr(chat.engine, "answer", lambda *a, **k: pytest.fail("must wait for the user's definition"))
    args = next(c for c in cases.load_cases() if c["id"] == "clarify-bridge-importance")["expected"]["tool_arguments"][0]
    # The model even puts a numerical query first: clarification must take priority.
    arguments = [{"targets": ["Usable::b2"]}, args]
    requests = []
    def client(path, payload):
        requests.append(payload)
        assert len(requests) == 1
        tools = [{"id": str(i), "type": "function", "function": {"name": "run_query", "arguments": a}}
                 for i, a in enumerate(arguments)]
        if provider == "ollama":
            return {"message": {"tool_calls": tools}}
        return {"choices": [{"message": {"tool_calls": tools}}]}
    if provider == "anthropic":
        def create(**kw):
            requests.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id=str(i), name="run_query", input=a)
                                            for i, a in enumerate(arguments)])
        backend = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    else:
        backend = client
    reply = chat.ask("toy-ca/demo", "Which bridge is most important?", client=backend, model="test")
    assert len(requests) == 1 and not reply["runs"] and reply["result"] is None
    assert reply["clarification"] == args["clarification"]
    assert "Forced outage" in reply["answer"] and "Diagnostic association" in reply["answer"]
    assert records.find(reply["record"])["clarification"] == reply["clarification"]
