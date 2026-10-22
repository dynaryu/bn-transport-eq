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
    a = engine.answer({**base, "do": {"Usable::b2": 0}}, record={"source": "test"})
    b = engine.answer({**base, "do": {"Usable::b2": 1}}, record={"source": "test"})
    assert a["targets"][0]["summary"]["P"]["0"] == 1
    assert b["targets"][0]["summary"]["P"]["1"] == 1
    assert a["record"]["samples_file"]["bank"] != b["record"]["samples_file"]["bank"]
    again = engine.answer({**base, "do": {"Usable::b2": 0}}, record={"source": "test"})
    assert again["record"]["samples_file"]["reused"]


def test_joint_probability_uses_the_same_samples_and_evidence_denominator():
    leaf = {"node": "Usable::b2", "op": "==", "value": 0}
    r = engine.answer({"assembly": TOY, "samples": 1000, "events": [leaf], "joint_events": [
        {"name": "same twice", "expression": {"all": [leaf, leaf]}},
        {"name": "tautology", "expression": {"any": [leaf, {"not": leaf}]}}
    ]})
    assert r["events"][0]["p"] == r["events"][1]["p"]
    assert r["events"][2]["p"] == 1
    observed = engine.answer({"assembly": TOY, "samples": 5000, "evidence": {"Usable::b2": 0}, "events": [leaf]})
    assert observed["events"][0]["p"] == 1
    assert 30 <= observed["inference"]["accepted"] < 5000


def test_observation_and_intervention_differ_with_a_common_cause(monkeypatch):
    import torch
    from tbnpy.variable import Variable
    from src.builder.kernels import Probability

    names = ["Root::r", "A::a", "B::b"]
    variables = {n: Variable(n, [0, 1]) for n in names}
    class Root(Probability):
        def evaluate(self, pars):
            return (torch.rand((len(pars), 1)) < .5).double()
    class A(Probability):
        def evaluate(self, pars):
            return torch.where(torch.rand((len(pars), 1)) < .8, pars, 1 - pars)
    class B(Probability):
        def evaluate(self, pars):
            return pars
    probs = {n: cls([variables[n]], [] if i == 0 else [variables[names[0]]],
                    {"role": "test", "function": n}) for i, (n, cls) in enumerate(zip(names, [Root, A, B]))}
    quantities = {n.split("::")[0]: {"id": n.split("::")[0], "kind": "discrete", "states": [0, 1]} for n in names}
    nodes = [{"id": n, "quantity": n.split("::")[0], "row": n.split("::")[1],
              "parents": [] if i == 0 else [names[0]]} for i, n in enumerate(names)]
    compiled = SimpleNamespace(library=SimpleNamespace(quantities=quantities), nodes=lambda: nodes,
        producer={(n["quantity"], n["row"]): i for i, n in enumerate(nodes)}, ancestors=lambda _: {0, 1, 2},
        bindings=[SimpleNamespace(function={"id": n}) for n in names], order=[0, 1, 2], assembly={"label": "confounded"})
    monkeypatch.setattr(engine, "compile_query", lambda q: compiled)
    monkeypatch.setattr(engine, "build_tbn", lambda *args: ({}, copy.deepcopy(probs)))
    monkeypatch.setattr(engine, "_computed", lambda *args: 3)
    base = {"assembly": "test/demo", "samples": 12000, "events": [{"node": "B::b", "op": "==", "value": 1}]}
    observed = engine.answer({**base, "evidence": {"A::a": 1}})
    forced = engine.answer({**base, "do": {"A::a": 1}})
    assert observed["events"][0]["p"] == pytest.approx(.8, abs=.025)
    assert forced["events"][0]["p"] == pytest.approx(.5, abs=.025)


def test_impossible_evidence_is_not_reported_as_zero_probability():
    with pytest.raises(engine.QueryError, match="contradicts"):
        engine.answer({"assembly": TOY, "samples": 100, "targets": ["Usable::b2"],
                       "do": {"Usable::b2": 0}, "evidence": {"Usable::b2": 1}})
    with pytest.raises(engine.QueryError, match="continuous observations"):
        engine.answer({"assembly": TOY, "targets": ["Usable::b2"], "evidence": {"SA(1.0)::b2": .5}})
    with pytest.raises(engine.QueryError, match="not a state"):
        engine.answer({"assembly": TOY, "targets": ["Usable::b2"], "do": {"Usable::b2": "unknown"}})


def test_forced_system_state_does_not_invoke_rule_reports():
    result = engine.answer({"assembly": TOY, "samples": 100, "do": {"SystemState::n4": 0},
                            "events": [{"node": "SystemState::n4", "op": "==", "value": 0}]})
    assert result["events"][0]["p"] == 1 and result["rules"] is None


def test_voi_optimises_after_each_observation_and_accounts_for_cost():
    task = {"state_nodes": ["Usable::b2"], "loss_units": "test units", "actions": [
        {"name": "repair", "losses": {"0": 20, "1": 20}}, {"name": "defer", "losses": {"0": 100, "1": 0}}],
        "inspections": [{"node": "Usable::b2", "cost": 2, "accuracy": [[1, 0], [0, 1]]}]}
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
