"""Tests for the tbnpy backend.

The load-bearing one is `test_backends_agree_*`: two independent implementations
of the same model should give the same answer, and that is the cheapest way to
notice if a change to either quietly moves the numbers.
"""

import math

import pytest

torch = pytest.importorskip("torch")
tbnpy_inference = pytest.importorskip("tbnpy.inference")

from tbnpy import variable  # noqa: E402

from src.forward_pilot.catalog import load_case  # noqa: E402
from src.forward_pilot.engine import ForwardQueryError, run_forward_query, tbn_available  # noqa: E402
from src.forward_pilot.hazard import component_probabilities  # noqa: E402
from src.forward_pilot.tbn import build, query_nodes_for  # noqa: E402
from src.forward_pilot.tbn.sa1 import SA1  # noqa: E402
from src.forward_pilot.tbn.x import UNUSABLE, USABLE, X  # noqa: E402


@pytest.fixture(scope="module")
def ema():
    return load_case("ema-highway")


@pytest.fixture(scope="module")
def hebei():
    return load_case("hebei-transport")


# ---- the model is a well-formed tbnpy network ----------------------------


def test_model_is_a_valid_dag_with_one_node_per_variable(hebei):
    varis, probs, meta = build(hebei, {"model": "uniform_sa1", "sa1_g": 0.7})
    order = tbnpy_inference.get_ancestor_order(probs, ["S"])
    assert order, "ancestor ordering failed, so the graph is cyclic or disconnected"

    # tbnpy requires every parent to be the child of exactly one node.
    defined = [v.name for node in probs.values() for v in node.childs]
    assert len(defined) == len(set(defined)), "a variable is defined by two nodes"
    for node in probs.values():
        for parent in node.parents:
            assert parent.name in set(defined), f"{parent.name} has no defining node"


def test_model_shape_matches_the_package(ema):
    _, _, meta = build(ema, dict(ema["default_hazard"]))
    n_edges, n_origins = len(ema["edges"]), len(ema["default_origins"])
    # SA1 + X per bridge, one S node, one population node.
    assert meta["n_nodes"] == 2 * n_edges + 2
    assert meta["n_variables"] == 2 * n_edges + n_origins + 1
    assert meta["has_population"] is True


def test_hebei_has_no_population_node(hebei):
    varis, probs, meta = build(hebei, {"model": "uniform_sa1", "sa1_g": 0.7})
    assert meta["has_population"] is False
    assert "PopFraction" not in probs
    assert "PopFraction" not in query_nodes_for(meta)


# ---- node-level contracts ------------------------------------------------


def test_sa1_sample_and_log_prob_describe_the_same_distribution():
    """tbnpy's MCMC calls log_prob far more than sample; they must agree."""
    var = variable.Variable(name="SA1::e", values=(0.0, float("inf")))
    node = SA1(childs=[var], parents=[], ln_median=math.log(0.4), sigma=0.5)
    torch.manual_seed(0)
    Cs, logp = node.sample(20000)
    assert torch.allclose(node.log_prob(Cs.unsqueeze(1)), logp, atol=1e-5)
    # Median of a lognormal is exp(ln_median).
    assert Cs.median().item() == pytest.approx(0.4, rel=0.05)


def test_degenerate_sa1_is_deterministic():
    var = variable.Variable(name="SA1::e", values=(0.0, float("inf")))
    node = SA1(childs=[var], parents=[], ln_median=math.log(0.7), sigma=0.0)
    Cs, logp = node.sample(16)
    assert node.deterministic
    assert torch.allclose(Cs, torch.full((16,), 0.7))
    assert torch.all(logp == 0)
    off = node.log_prob(torch.full((3, 1), 0.9))
    assert torch.all(torch.isinf(off)) and torch.all(off < 0)


def test_x_matches_the_hazus_curve_and_orders_states_worst_first():
    var = variable.Variable(name="X::e", values=[0, 1])
    parent = variable.Variable(name="SA1::e", values=(0.0, float("inf")))
    node = X(childs=[var], parents=[parent], median=0.6, beta=0.6)

    # At the median intensity the Extensive limit state is reached half the time.
    assert node.p_unusable(torch.tensor([0.6])).item() == pytest.approx(0.5, abs=1e-6)
    # No shaking, no damage.
    assert node.p_unusable(torch.tensor([0.0])).item() == 0.0
    # Monotone in intensity.
    p = node.p_unusable(torch.tensor([0.1, 0.3, 0.6, 1.2, 3.0]))
    assert torch.all(p[1:] >= p[:-1])
    assert (UNUSABLE, USABLE) == (0, 1)


def test_x_log_prob_is_minus_inf_outside_the_support():
    var = variable.Variable(name="X::e", values=[0, 1])
    parent = variable.Variable(name="SA1::e", values=(0.0, float("inf")))
    node = X(childs=[var], parents=[parent], median=0.6, beta=0.6)
    Cs = torch.tensor([[0.0, 0.6], [1.0, 0.6], [7.0, 0.6]])
    lp = node.log_prob(Cs)
    assert torch.isfinite(lp[:2]).all()
    assert torch.isinf(lp[2]) and lp[2] < 0


def test_service_node_is_a_degenerate_kernel(hebei):
    varis, probs, meta = build(hebei, {"model": "uniform_sa1", "sa1_g": 0.7})
    s = probs["S"]
    all_up = torch.ones((4, len(meta["edge_order"])), dtype=torch.long)
    Cs, logp = s.sample(all_up)
    assert torch.all(logp == 0), "a deterministic node contributes no log-probability"
    assert torch.all(Cs == 1), "with every bridge usable, every origin is connected"

    consistent = torch.cat([Cs.float(), all_up.float()], dim=1)
    assert torch.all(s.log_prob(consistent) == 0)
    wrong = consistent.clone()
    wrong[:, 0] = 0
    assert torch.all(torch.isinf(s.log_prob(wrong)))


# ---- the claim that justifies the split ----------------------------------


def test_split_dispersions_reproduce_the_combined_sigma(ema):
    """Ground motion and fragility sit in separate nodes here and in one closed
    form in the analytic path. The marginals must still agree, or the split
    changed the model rather than just its representation."""
    hazard = dict(ema["default_hazard"])
    _, probs, meta = build(ema, hazard)
    edges = meta["edge_order"][:25]
    torch.manual_seed(11)
    out = tbnpy_inference.sample(probs, [f"X::{e}" for e in edges], 20000)

    analytic, _, _ = component_probabilities(ema, hazard)
    standard_error = (0.25 / 20000) ** 0.5
    for edge_id in edges:
        empirical = 1.0 - out[f"X::{edge_id}"].Cs.float().mean().item()
        assert abs(empirical - analytic[edge_id]) < 5 * standard_error


# ---- the two backends answer the same question ---------------------------


def test_backends_agree_on_hebei(hebei):
    query = {"query_type": "connectivity", "samples": 8000, "seed": 7,
             "hazard": {"model": "uniform_sa1", "sa1_g": 0.7}}
    a = run_forward_query(hebei, {**query, "backend": "analytic"})
    t = run_forward_query(hebei, {**query, "backend": "tbn"})
    assert a["backend"] == "analytic" and t["backend"] == "tbn"
    assert abs(a["origin_service"][0]["probability"] - t["origin_service"][0]["probability"]) < 0.02


def test_backends_agree_on_ema_population(ema):
    query = {"query_type": "population_accessibility", "samples": 4000, "seed": 3}
    a = run_forward_query(ema, {**query, "backend": "analytic"})
    t = run_forward_query(ema, {**query, "backend": "tbn"})
    assert (abs(a["population_accessibility"]["expected_fraction"]
                - t["population_accessibility"]["expected_fraction"]) < 0.01)
    # The per-bridge marginal has a closed form, so both report it exactly.
    assert [c["edge_id"] for c in a["components"][:10]] == [c["edge_id"] for c in t["components"][:10]]
    assert a["components"][0]["p_unusable"] == pytest.approx(t["components"][0]["p_unusable"])


def test_tbn_is_the_default_when_installed(ema):
    result = run_forward_query(ema, {"query_type": "connectivity", "samples": 200})
    assert tbn_available()
    assert result["backend"] == "tbn"
    assert result["inference"] == "forward_tbnpy_sampling"
    assert result["model_size"]["nodes"] == 2 * len(ema["edges"]) + 2


def test_tbn_backend_is_reproducible_given_a_seed(hebei):
    query = {"query_type": "connectivity", "samples": 1500, "seed": 99, "backend": "tbn"}
    a = run_forward_query(hebei, query)
    b = run_forward_query(hebei, query)
    assert a["origin_service"][0]["probability"] == b["origin_service"][0]["probability"]


def test_unknown_backend_is_rejected(hebei):
    with pytest.raises(ForwardQueryError, match="unknown backend"):
        run_forward_query(hebei, {"samples": 100, "backend": "pgmpy"})


def test_forward_boundary_still_holds_on_the_tbn_backend(ema):
    """The validator runs before the backend is chosen, so neither path can
    reach a diagnostic query."""
    with pytest.raises(ForwardQueryError, match="forward inference only"):
        run_forward_query(ema, {"query_type": "critical_components",
                                "samples": 100, "backend": "tbn"})
    with pytest.raises(ForwardQueryError, match="forward inference only"):
        run_forward_query(ema, {"evidence": {"S::n1": 0}, "samples": 100, "backend": "tbn"})


def test_zero_intensity_is_a_legal_query(hebei):
    """The slider reaches zero; the tbn path must not take log(0) and fall over."""
    result = run_forward_query(hebei, {
        "samples": 200, "backend": "tbn",
        "hazard": {"model": "uniform_sa1", "sa1_g": 0.0},
    })
    assert result["origin_service"][0]["probability"] == 1.0
    assert all(c["p_unusable"] == 0.0 for c in result["components"])
