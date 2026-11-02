"""Checks of the generated toy-ca model against independent calculations."""

import copy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest
import torch
from tbnpy import inference

from src.builder.bn_builder import (
    build_tbn, compile_path, define_tbn_probabilities, define_tbn_variables, generate_tbn, tbn_data,
)
from src.builder.load import assembly_path
from src.inference.engine import compile_query

MODEL = Path(__file__).resolve().parents[3] / "examples/toy-ca/tbnpy/demo"


@pytest.fixture
def model():
    data = json.loads((MODEL / "model.json").read_text(encoding="utf-8"))
    variables = define_tbn_variables(data)
    return data, variables, define_tbn_probabilities(data, variables, directory=MODEL)


def test_generated_snapshot_matches_sources_and_owns_joint_hazard(model):
    data, variables, probs = model
    expected = tbn_data(compile_path(assembly_path("toy-ca/demo")))
    actual = copy.deepcopy(data)
    for key in ("source_sha256", "versions"):
        actual.pop(key)
    for spec in actual["probabilities"].values():
        spec.pop("references", None)
    assert actual == expected
    owned = [v.name for p in probs.values() for v in p.childs]
    assert len(owned) == len(set(owned)) == len(variables) == 46
    hazard = [p for p in probs.values() if p.spec["function"] == "hazard.gmm"]
    assert len(hazard) == 1
    assert len(hazard[0].childs) == 10
    assert len(hazard[0].parents) == 0                 # the event is fixed data, not a parent
    assert set(inference.get_ancestor_order(probs, list(probs))) == set(probs)


def test_every_object_is_a_tbnpy_probability_that_reads_as_its_distribution(model):
    import tbnpy

    _, _, probs = model
    assert all(isinstance(p, tbnpy.Probability) for p in probs.values())
    assert repr(probs["DamageState.HazusBridge::b4"]).startswith(
        "Fragility(P(DamageState.HazusBridge::b4 | SA(1.0)::b4): lognormal, P(DS >= k)")
    assert repr(probs["EdgeOpen::e4"]) == "EdgeOpen(P(EdgeOpen::e4 | Usable::b4): 1 if Usable::b4 is 1, else 0)"
    assert "object at 0x" not in repr(probs)
    assert "rules for n4:" in str(probs["SystemState::n4"])
    assert "note: hazard.gmm draws each Bridge's residual independently" in str(probs["hazard.gmm"])
    assert tbnpy.check_model(probs, n_sample=64).ok


def test_samples_and_log_probabilities_agree(model):
    _, _, probs = model
    torch.manual_seed(19)
    results = inference.sample(probs, list(probs), n_sample=100, batch_size=37)
    values = {v.name: results[key].Cs[:, j] for key, p in probs.items() for j, v in enumerate(p.childs)}
    for key, prob in probs.items():
        full = torch.stack([values[v.name] for v in prob.childs + prob.parents], 1)
        logp = prob.log_prob(full)
        assert logp.shape == (100,)
        assert torch.isfinite(logp).all()
        torch.testing.assert_close(logp, results[key].ps, check_dtype=False)
        if prob.spec["function"] not in ("hazard.gmm", "fragility.bridge"):
            invalid = full.clone()
            invalid[:, 0] = -1
            assert torch.isneginf(prob.log_prob(invalid)).all()


def test_system_rules_match_ndtools_for_every_edge_configuration(model):
    _, _, probs = model
    states = [p for p in probs.values() if p.spec["role"] == "system"]
    assert len(states) == 6 and all(len(s.childs) == 1 and s.refs[0] is not None for s in states)
    for state in states:                  # one object per origin, each with its own rules
        edges = list(itertools.product((0, 1), repeat=len(state.parents)))
        pars = torch.tensor(edges, dtype=torch.float64)
        classified = state.evaluate(pars)
        evaluate = state.system_function(0)
        names = [v.name.split("::", 1)[1] for v in state.parents]
        expected = [evaluate(dict(zip(names, row))) for row in edges]
        torch.testing.assert_close(classified[:, 0], torch.tensor([e[1] for e in expected], dtype=torch.float64))
        # Force every state to be unclassified: the ndtools fallback must give identical results.
        state.refs = [(torch.empty((0, len(state.parents), 2), dtype=torch.int32),) * 2]
        torch.testing.assert_close(state.evaluate(pars), classified)


def test_every_child_outside_the_hazard_field_is_its_own_object(model):
    _, _, probs = model
    for key, prob in probs.items():
        if prob.spec["function"] == "hazard.gmm":
            assert len(prob.childs) == 10                     # the field: every bridge's intensity together
        else:
            assert [v.name for v in prob.childs] == [key]     # one child, keyed by it
    assert [v.name for v in probs["DamageState.HazusBridge::b4"].parents] == ["SA(1.0)::b4"]
    assert [v.name for v in probs["EdgeOpen::e4"].parents] == ["Usable::b4"]


@pytest.fixture
def stored_at_network_threshold(tmp_path, monkeypatch):
    """The toy-ca rules as the repository ships them, at the network's threshold only, whatever else
    the app may have extracted since: a private copy the model directory points at."""
    import shutil

    from src.builder import bn_builder

    model = Path(__file__).resolve().parents[3] / "examples/toy-ca/tbnpy/demo"
    copy = tmp_path / "model"
    for folder in (model / "rsr").glob("*/target_add_h=0.5"):
        shutil.copytree(folder, copy / "rsr" / folder.parent.name / folder.name)
    monkeypatch.setattr(bn_builder, "model_dir", lambda assembly_file: copy)
    return copy


def _sources(query):
    from src.builder.bn_builder import rules_source, rules_threshold

    _, probs = build_tbn(compile_query(query))
    systems = [p for p in probs.values() if p.spec["role"] == "system"]
    return {p.spec["rows"][0]["id"]: rules_source(p.spec["references"][0] if p.refs[0] is not None else None,
                                                  rules_threshold(p.spec)) for p in systems}


def test_the_engine_reuses_stored_rules_under_another_scenario(stored_at_network_threshold):
    """Rules depend on the network and the system event, not the hazard: a changed Event keeps them."""
    for rows in ({}, {"demo-san-andreas": {"mw": 7.0}}):
        assert set(_sources({"assembly": "toy-ca/demo", "rows": rows}).values()) == {"exact"}


def test_rules_from_another_threshold_are_reused_where_they_still_hold(stored_at_network_threshold):
    # Stricter (0.25 h): failing within 0.5 h means failing within 0.25 h, so only the failure rules hold.
    stricter = _sources({"assembly": "toy-ca/demo", "rows": {"reach-key-destination": {"target_add_h": 0.25}}})
    assert set(stricter.values()) == {"failure rules from target_add_h=0.5"}
    # Looser (1 h): surviving within 0.5 h means surviving within 1 h, so only the survival rules hold.
    looser = _sources({"assembly": "toy-ca/demo", "rows": {"reach-key-destination": {"target_add_h": 1.0}}})
    assert set(looser.values()) == {"survival rules from target_add_h=0.5"}


def test_choose_rules_takes_the_nearest_threshold_on_each_side():
    from src.builder.bn_builder import choose_rules

    stored = [(0.25, Path("a")), (0.5, Path("b")), (1.0, Path("c"))]
    ref = choose_rules(stored, 0.75)
    assert (ref["upper_at"], ref["lower_at"]) == (0.5, 1.0)              # survival from below, failure from above
    assert choose_rules(stored, 0.5)["upper_at"] == choose_rules(stored, 0.5)["lower_at"] == 0.5
    assert choose_rules(stored, 2.0)["lower"] is None and choose_rules([], 0.5) is None


def test_borrowed_rules_give_the_answer_ndtools_gives_and_report_what_is_left(stored_at_network_threshold, monkeypatch):
    from src.builder import bn_builder
    from src.inference import engine

    query = {"assembly": "toy-ca/demo", "events": [{"node": "SystemState::*", "op": "==", "value": 0}],
             "samples": 4000, "seed": 3, "rows": {"reach-key-destination": {"target_add_h": 0.25}}}
    borrowed = engine.answer(query)
    assert borrowed["rules"]["target_add_h"] == 0.25 and borrowed["rules"]["extract"]       # mostly unclassified
    monkeypatch.setattr(bn_builder, "attach_rules", lambda data, directory: 0)
    assert [e["p"] for e in engine.answer(query)["events"]] == [e["p"] for e in borrowed["events"]]


def test_extracted_rules_are_stored_by_threshold_and_then_used(tmp_path, monkeypatch):
    from src.builder import bn_builder

    compiled = compile_query({"assembly": "toy-ca/demo", "rows": {"reach-key-destination": {"target_add_h": 0.25}}})
    assert bn_builder.extract_system_rules(compiled, tmp_path, {"n4"}, log=lambda _: None) == ["n4"]
    assert (tmp_path / "rsr" / "n4" / "target_add_h=0.25" / "refs_up_1.pt").is_file()
    data = bn_builder.tbn_data(compiled)
    assert bn_builder.attach_rules(data, tmp_path) == 1                  # n4 only: the others have none stored here
    n4 = next(s for s in data["probabilities"].values() if s["role"] == "system" and s["rows"][0]["id"] == "n4")
    assert bn_builder.rules_source(n4["references"][0], 0.25) == "exact"


def test_generation_without_rules_is_importable_and_rejects_stale_rules(tmp_path):
    assembly = assembly_path("toy-ca/demo")
    (tmp_path / "toy-ca").mkdir()
    local = tmp_path / "toy-ca" / assembly.name
    local.write_text(assembly.read_text(encoding="utf-8"), encoding="utf-8")
    # Generated files normally sit inside the repo. For this import check,
    # retain their documented directory structure beneath a temporary root.
    (tmp_path / "src/builder").mkdir(parents=True)
    (tmp_path / "src/builder/bn_builder.py").touch()
    out = generate_tbn(local, extract_rules=False)
    spec = importlib.util.spec_from_file_location("generated_toy", out / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    variables = module.define_variables()
    probs = module.define_probs(variables)
    assert len(inference.sample(probs, list(probs), 10)) == 37      # 1 hazard field + 10 + 10 + 10 + 6

    data = json.loads((MODEL / "model.json").read_text(encoding="utf-8"))
    data["source_sha256"] = "different model"
    with pytest.raises(ValueError, match="different generated model"):
        define_tbn_probabilities(data, define_tbn_variables(data), directory=MODEL)


def test_every_threshold_without_its_own_rules_gets_them(stored_at_network_threshold):
    """A threshold is part of the system function: borrowed rules carry the answer meanwhile, but every
    origin without rules of its own is listed for extraction, however little it leaves to ndtools."""
    from src.inference import engine

    result = engine.answer({"assembly": "toy-ca/demo", "events": [{"node": "SystemState::*", "op": "==", "value": 0}],
                            "samples": 2000, "rows": {"reach-key-destination": {"target_add_h": 1.0}}})
    assert sorted(result["rules"]["extract"]) == ["n3", "n4", "n5", "n6", "n7", "n8"]
    assert min(o["unclassified"] for o in result["rules"]["origins"]) < 0.05       # even the nearly covered ones
