"""Every answer is recorded with its provenance; a scenario's samples are banked once and read again; all replay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.inference import engine, records

TOY = "toy-ca/demo"
QUERY = {"assembly": TOY, "events": [{"node": "SystemState::*", "op": "==", "value": 0}], "samples": 2000, "seed": 5}


def _replays(record_id):
    from src.__main__ import main

    return main(["replay", record_id]) == 0


def test_a_recorded_query_keeps_its_numbers_and_provenance_and_banks_the_whole_bn():
    result = engine.answer(QUERY, record={"source": "test"})
    (record,) = records.read("query")
    assert record["id"] == result["record"]["id"] and record["source"] == "test"
    assert record["query"]["seed"] == 5 and record["samples"] == 2000 and record["elapsed_s"] > 0
    assert len(record["events"]) == 6 and record["rules"]["target_add_h"] == 0.5
    provenance = record["provenance"]
    assert len(provenance["model_sha256"]) == 64 and "tbnpy" in provenance["versions"]
    assert provenance["parameter_sets"]["fragility.bridge"] == {"hazus61.HWB11": 7, "hazus61.HWB5": 3}
    assert record["samples_from"] == "the whole BN, sampled now and stored" and not record["samples_file"]["reused"]
    bank = records.load_samples(record)
    assert len(bank) == 46 and bank["SA(1.0)::b4"].dtype == np.float32 and bank["Usable::b4"].dtype == np.uint8
    for event in record["events"]:                               # the numbers are the bank's
        assert np.mean(bank[event["node"]] == 0) == event["p"]
    assert _replays(record["id"])


def test_a_follow_up_on_the_same_scenario_reads_the_bank_and_its_scope_only():
    first = engine.answer(QUERY, record={"source": "test"})
    follow = engine.answer({**QUERY, "events": [{"node": "Usable::b4", "op": "==", "value": 0}]}, record={"source": "test"})
    assert follow["samples_from"] == "the scenario's stored samples" and follow["record"]["samples_file"]["reused"]
    assert follow["record"]["samples_file"]["bank"] == first["record"]["samples_file"]["bank"]
    assert follow["scope"] == {"relevant": 3, "of": 46, "computed": 0}      # read from the bank, nothing sampled
    bank = records.load_samples(records.find(follow["record"]["id"]))
    assert follow["events"][0]["p"] == np.mean(bank["Usable::b4"] == 0)
    assert _replays(follow["record"]["id"])                       # sampled afresh, it gives the bank's numbers


def test_another_scenario_gets_its_own_bank():
    base = engine.answer(QUERY, record={"source": "test"})
    retrofit = engine.answer({**QUERY, "rows": {"b2": {"hazus_class": "HWB11"}}}, record={"source": "test"})
    assert retrofit["record"]["samples_file"]["bank"] != base["record"]["samples_file"]["bank"]
    assert not retrofit["record"]["samples_file"]["reused"]


def test_a_bn_too_large_to_bank_is_sampled_on_the_fly_for_its_scope_only(monkeypatch):
    monkeypatch.setattr(records, "SAMPLE_LIMIT_BYTES", 1000)
    result = engine.answer({**QUERY, "events": [{"node": "Usable::b4", "op": "==", "value": 0}]}, record={"source": "test"})
    stored = result["record"]["samples_file"]
    assert stored["bank"] is None and "scope only" in stored["reason"]
    assert result["samples_from"] == "the query's scope, sampled now" and result["scope"]["computed"] == 12
    assert records.load_samples(records.find(result["record"]["id"])) is None
    assert _replays(result["record"]["id"])


def test_scope_names_the_ancestors_and_prunes_the_rest():
    s = engine.scope({"assembly": TOY, "events": [{"node": "SystemState::n4", "op": "==", "value": 0}]})
    assert s["relevant"]["count"] == 41 and s["relevant"]["by_quantity"]["SystemState"] == 1
    assert s["pruned"] == [f"SystemState::{n}" for n in ("n3", "n5", "n6", "n7", "n8")]
    bridge = engine.scope({"assembly": TOY, "targets": ["Usable::b4"]})
    assert bridge["relevant"]["by_quantity"] == {"SA(1.0)": 1, "DamageState.HazusBridge": 1, "Usable": 1}
    assert len(bridge["sampled"]["beyond_relevant"]) == 9 and bridge["joint"] == ["hazard.gmm"]


def test_a_question_is_recorded_with_its_translation_and_the_query_behind_it(monkeypatch):
    from src.llm import chat

    monkeypatch.setenv("BN_LLM", "anthropic")
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(stop_reason="tool_use", content=[SimpleNamespace(
                    type="tool_use", id="t1", name=chat.TOOL_NAME,
                    input={"events": [{"node": "SystemState::n4", "op": "==", "value": 0}], "samples": 500})])
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="About 14%.")])

    reply = chat.ask(TOY, "How likely is n4 to be cut off?", client=SimpleNamespace(beta=SimpleNamespace(messages=Messages())),
                     model="m")
    ask = records.find(reply["record"])
    assert ask["kind"] == "ask" and ask["question"] == "How likely is n4 to be cut off?" and ask["answer"] == "About 14%."
    assert ask["llm"] == {"provider": "anthropic", "model": "m"}
    (translation,) = ask["translations"]
    query = records.find(translation["record"])
    assert query["source"] == "ask" and query["question"] == ask["question"] and query["events"][0]["node"] == "SystemState::n4"
