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
    assert len(owned) == len(set(owned)) == len(variables) == 52
    hazard = [p for p in probs.values() if p.spec["function"] == "hazard.gmm"]
    assert len(hazard) == 1
    assert len(hazard[0].childs) == 10
    assert len(hazard[0].parents) == 0                 # the event is fixed data, not a parent
    assert set(inference.get_ancestor_order(probs, list(probs))) == set(probs)


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
    state = next(p for p in probs.values() if p.spec.get("output") == "state")
    time = next(p for p in probs.values() if p.spec.get("output") == "travel_time")
    edges = list(itertools.product((0, 1), repeat=len(state.parents)))
    pars = torch.tensor(edges, dtype=torch.float64)
    classified = state.evaluate(pars)
    direct_time = time.evaluate(pars)
    for j in range(len(state.childs)):
        evaluate = state.system_function(j)
        names = [v.name.split("::", 1)[1] for v in state.parents]
        expected = [evaluate(dict(zip(names, row))) for row in edges]
        torch.testing.assert_close(classified[:, j], torch.tensor([e[1] for e in expected], dtype=torch.float64))
        torch.testing.assert_close(direct_time[:, j], torch.tensor([e[0] for e in expected], dtype=torch.float64))
    # Force all states to be unclassified: fallback must give identical results.
    state.refs = [(torch.empty((0, len(state.parents), 2), dtype=torch.int32),) * 2 for _ in state.childs]
    torch.testing.assert_close(state.evaluate(pars), classified)


def test_the_engine_reuses_stored_rules_under_another_scenario():
    """Rules depend on the network and the threshold, not the hazard: a changed Event keeps them."""
    for mw in (None, 7.0):
        query = {"assembly": "toy-ca/demo"} if mw is None else {
            "assembly": "toy-ca/demo", "rows": {"demo-san-andreas": {"mw": mw}}}
        _, probs = build_tbn(compile_query(query))
        state = next(p for p in probs.values() if p.spec.get("output") == "state")
        assert all(ref is not None for ref in state.refs)
    # Another threshold is another system definition: no stored rule applies, ndtools answers.
    _, probs = build_tbn(compile_query({"assembly": "toy-ca/demo",
                                        "parameters": {"system.travel-time-to-nearest": {"target_max": 0.25}}}))
    state = next(p for p in probs.values() if p.spec.get("output") == "state")
    assert all(ref is None for ref in state.refs)


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
    assert len(inference.sample(probs, list(probs), 10)) == 6

    data = json.loads((MODEL / "model.json").read_text(encoding="utf-8"))
    data["source_sha256"] = "different model"
    with pytest.raises(ValueError, match="different generated model"):
        define_tbn_probabilities(data, define_tbn_variables(data), directory=MODEL)
