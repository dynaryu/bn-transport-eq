"""Causal semantics, joint dependence, evidence denominators and decision value."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.inference import engine
from src.inference.comparisons import execute
from src.inference.decision import from_distribution

TOY = "toy-ca/demo"


def test_forced_states_propagate_and_have_distinct_sample_banks():
    base = {"assembly": TOY, "samples": 200, "targets": ["EdgeOpen::e2"]}
    a = engine.answer({**base, "do": {"EdgeOpen::e2": 0}}, record={"source": "test"})         # by index
    b = engine.answer({**base, "do": {"EdgeOpen::e2": "True"}}, record={"source": "test"})    # by label
    assert a["targets"][0]["summary"]["P"]["False"] == 1
    assert b["targets"][0]["summary"]["P"]["True"] == 1
    assert a["record"]["samples_file"]["bank"] != b["record"]["samples_file"]["bank"]
    again = engine.answer({**base, "do": {"EdgeOpen::e2": 0}}, record={"source": "test"})
    assert again["record"]["samples_file"]["reused"]


def test_joint_probability_uses_the_same_samples_and_evidence_denominator():
    leaf = {"node": "EdgeOpen::e2", "op": "==", "value": 0}
    r = engine.answer({"assembly": TOY, "samples": 1000, "events": [leaf], "joint_events": [
        {"name": "same twice", "expression": {"all": [leaf, leaf]}},
        {"name": "tautology", "expression": {"any": [leaf, {"not": leaf}]}}
    ]})
    assert r["events"][0]["p"] == r["events"][1]["p"]
    assert r["events"][2]["p"] == 1
    observed = engine.answer({"assembly": TOY, "samples": 5000, "evidence": {"EdgeOpen::e2": 0}, "events": [leaf]})
    assert observed["events"][0]["p"] == 1
    assert observed["inference"]["accepted"] == 5000       # the clique tree conditions on it: every row counts


def test_observation_and_intervention_differ_with_a_common_cause():
    """SA drives both b4's damage and its station's reading: observing complete damage raises the reading,
    forcing it does not."""
    event = [{"node": "SAObs(1.0)::b4", "op": ">", "value": 0.5}]
    base = {"assembly": TOY, "samples": 20000, "events": event}
    prior = engine.answer(base)["events"][0]["p"]
    observed = engine.answer({**base, "evidence": {"DamageState.HazusBridge::b4": 4}})["events"][0]["p"]
    forced = engine.answer({**base, "do": {"DamageState.HazusBridge::b4": 4}})["events"][0]["p"]
    assert observed > prior + 0.1
    assert forced == pytest.approx(prior, abs=0.03)


def test_impossible_evidence_is_not_reported_as_zero_probability():
    with pytest.raises(engine.QueryError, match="contradicts"):
        engine.answer({"assembly": TOY, "samples": 100, "targets": ["EdgeOpen::e2"],
                       "do": {"EdgeOpen::e2": 0}, "evidence": {"EdgeOpen::e2": 1}})
    with pytest.raises(engine.QueryError, match="continuous observations"):
        engine.answer({"assembly": TOY, "targets": ["EdgeOpen::e2"], "evidence": {"SA(1.0)::b2": .5}})
    with pytest.raises(engine.QueryError, match="not a state"):
        engine.answer({"assembly": TOY, "targets": ["EdgeOpen::e2"], "do": {"EdgeOpen::e2": "unknown"}})


def test_forced_system_state_does_not_invoke_rule_reports():
    result = engine.answer({"assembly": TOY, "samples": 100, "do": {"SystemState::n4": 0},
                            "events": [{"node": "SystemState::n4", "op": "==", "value": 0}]})
    assert result["events"][0]["p"] == 1
    assert "n4" not in [o["origin"] for o in result["rules"]["origins"]]      # the forced origin is not evaluated


def test_voi_optimises_after_each_observation_and_accounts_for_cost():
    task = {"state_nodes": ["EdgeOpen::e2"], "loss_units": "test units", "actions": [
        {"name": "repair", "losses": {"0": 20, "1": 20}}, {"name": "defer", "losses": {"0": 100, "1": 0}}],
        "inspections": [{"node": "EdgeOpen::e2", "cost": 2, "accuracy": [[1, 0], [0, 1]]}]}
    result = from_distribution([(0,), (1,)], [.2, .8], task)
    assert result["current_expected_loss"] == 20
    assert result["rows"][0]["evsi"] == pytest.approx(16)
    assert result["rows"][0]["net_value"] == pytest.approx(14)
    assert [o["action"] for o in result["rows"][0]["outcomes"]] == ["repair", "defer"]
    task["inspections"][0]["accuracy"] = [[.5, .5], [.5, .5]]
    assert from_distribution([(0,), (1,)], [.2, .8], task)["recommendation"] == "do not inspect"


def test_path_candidates_end_at_requested_destination_and_have_joint_events():
    result = execute({"kind": "paths", "query": {"assembly": TOY, "samples": 300},
                      "origin": "n4", "destination": "n1", "limit": 3})
    assert len(result["rows"]) == 3
    assert all(r["nodes"][0] == "n4" and r["nodes"][-1] == "n1" for r in result["rows"])
    assert all(r["travel_time_h"] > 0 and 0 <= r["p"] <= 1 for r in result["rows"])
