"""BN comparisons retain actual prior samples and weights across updates and restarts."""

import pytest
import torch

from src.builder.bn_builder import build_tbn
from src.inference import cliques, engine
from src.inference.session import Session


@pytest.fixture
def session():
    variables, probs = build_tbn(engine.compile_query({"assembly": "toy-ca/demo"}))
    return Session(cliques.tree(probs), probs, variables, threshold=0,
                   sampler={"n_chain": 20, "n_keep": 10, "n_burn": 5})


def test_discrete_comparison_uses_previous_weights_and_advances_one_round(session):
    node = "EdgeOpen::e6"
    initial = session.distribution(node)
    assert initial["previous"] is None
    session.observe("SAObs(1.0)::b6", 0.8, at="12:00:01")
    updated = session.distribution(node)
    assert updated["previous"]["p"] == initial["p"]
    assert updated["previous"]["se"] == initial["se"]
    assert updated["previous"]["at"] == initial["at"]
    assert updated["p"] != initial["p"]
    session.observe(node, "closed", at="12:00:02")
    final = session.distribution(node)
    assert final["p"] == pytest.approx([1, 0])
    assert final["previous"]["p"] == updated["p"]
    assert final["previous"]["se"] == updated["se"]
    assert final["previous"]["at"] == "12:00:01"
    assert not final["previous"]["observed"]
    assert final["observed"]


@pytest.mark.parametrize("threshold", [0, 1.1])
def test_continuous_comparison_shares_bins_and_keeps_prior_after_resampling(session, threshold):
    node = "SAObs(1.0)::b6"
    initial = session.distribution(node)
    prior_rows = session.rows
    session.threshold = threshold
    session.observe(node, 0.8)
    d = session.distribution(node)
    previous = d["previous"]
    assert session.previous["rows"] is prior_rows
    assert previous["median_g"] == initial["median_g"]
    assert d["median_g"] == pytest.approx(0.8)
    assert d["ln_edges"] == previous["ln_edges"]
    for round_ in (d, previous):
        edges = round_["ln_edges"]
        assert sum(p * (b - a) for p, a, b in zip(round_["density"], edges, edges[1:])) == pytest.approx(1)
    assert session.timeline[-1]["update"] == ("reweighted" if threshold == 0 else "sampled again")


def test_comparison_is_saved_and_old_session_files_still_load(session, tmp_path):
    node = "EdgeOpen::e6"
    session.observe("SAObs(1.0)::b6", 0.8)
    expected = session.distribution(node)
    path = tmp_path / "session.pt"
    session.save(path)
    loaded, _ = Session.load(path, session.t, session.probs, session.variables)
    assert loaded.distribution(node) == expected
    # Unchanged sample columns stay shared rather than doubling the saved sample set.
    assert loaded.previous["rows"][node] is loaded.rows[node]
    data = torch.load(path, weights_only=False)
    data.pop("previous")
    torch.save(data, path)
    legacy, _ = Session.load(path, session.t, session.probs, session.variables)
    assert legacy.distribution(node)["previous"] is None
    assert legacy.distribution(node)["p"] == expected["p"]


def test_query_round_retains_same_distribution(session):
    node = "EdgeOpen::e6"
    before = session.distribution(node)
    session.add_query({"target": {"kind": "cut_off"}, "decision": "track"})
    after = session.distribution(node)
    assert after["p"] == after["previous"]["p"] == before["p"]
