"""Assemble the pilot's Bayesian network for one package and one scenario.

The network the UI diagram draws, built for real:

    SA1::<edge>   root, one per bridge     ground motion in g
        |
    X::<edge>     one per bridge           usable / unusable
        |
    S             one node, one child per origin
        |
    PopFraction   the decision metric      (omitted without population weights)

The scenario is not a variable. It is an input the user sets, so it parameterises
the SA1 roots -- which is exactly what the `EarthquakeScenario` box in the UI
means by `type: input`.

Ground-motion and fragility dispersions are deliberately kept in separate nodes.
The analytic engine combines them into one sigma, which gives the same marginals
but leaves nowhere for a correlated residual to attach; keeping them apart means
the spatial-correlation extension is a new parent on SA1 rather than a rewrite.
"""

from __future__ import annotations

from math import exp, hypot, log, sqrt
from typing import Any, Mapping, Sequence

from tbnpy import variable

from ..hazard import _aleatory_sigma, _edge_midpoint, _gmpe_ln_sa1, _median_extensive
from .pop import PopulationAccess
from .s import S
from .sa1 import SA1
from .x import X

SERVICE_STATES = [0, 1]          # worst -> best, as tbnpy expects
USABILITY_STATES = [0, 1]
_EPISTEMIC_SIGMA = 0.3           # the dataset's placeholder, same value the analytic path uses


def _site_terms(
    network: Mapping[str, Any], hazard: Mapping[str, Any]
) -> dict[str, tuple[float, float]]:
    """Per-edge (ln median SA1, ground-motion sigma) for the chosen scenario.

    Mirrors `hazard.component_probabilities`, but stops before folding the
    fragility dispersion in -- that belongs to the X node here.
    """
    model = hazard.get("model")
    if model not in network["hazard_models"]:
        raise ValueError(
            f"hazard model {model!r} is not available; choose from {network['hazard_models']}"
        )

    if model == "uniform_sa1":
        sa1 = float(hazard.get("sa1_g", 0.0))
        if sa1 < 0:
            raise ValueError("sa1_g must be nonnegative")
        # One fixed intensity everywhere, so the ground-motion node is degenerate.
        # Zero intensity is a legal query (the UI slider reaches it): -inf log
        # median gives SA1 = 0 and, through the fragility, no chance of damage.
        ln_median = log(sa1) if sa1 > 0 else float("-inf")
        return {edge_id: (ln_median, 0.0) for edge_id in network["edges"]}

    if model == "shakemap":
        # ShakeMap's own median and log standard deviation at each bridge site.
        from ..shakemap import site_terms

        return site_terms(network, hazard)[0]

    magnitude = float(hazard.get("magnitude", 8.0))
    if not 4.0 <= magnitude <= 9.5:
        raise ValueError("magnitude must lie between 4.0 and 9.5 for this pilot")
    epicentre_x = float(hazard.get("epicentre_x", 1.0))
    epicentre_y = float(hazard.get("epicentre_y", 10.0))
    sigma = sqrt(_aleatory_sigma(magnitude) ** 2 + _EPISTEMIC_SIGMA**2)

    terms = {}
    for edge_id, edge in network["edges"].items():
        x, y = _edge_midpoint(edge, network["nodes"])
        distance = hypot(x - epicentre_x, y - epicentre_y)
        terms[edge_id] = (_gmpe_ln_sa1(magnitude, distance), sigma)
    return terms


def build(
    network: Mapping[str, Any],
    hazard: Mapping[str, Any],
    origins: Sequence[str] | None = None,
    destinations: Sequence[str] | None = None,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (varis, probs, meta) for tbnpy's inference functions."""
    origins = list(origins or network["default_origins"])
    destinations = list(destinations or network["destinations"])
    edge_order = list(network["edges"])
    terms = _site_terms(network, hazard)

    varis: dict[str, Any] = {}
    probs: dict[str, Any] = {}

    for edge_id in edge_order:
        ln_median, sigma = terms[edge_id]
        sa_name, x_name = f"SA1::{edge_id}", f"X::{edge_id}"

        varis[sa_name] = variable.Variable(name=sa_name, values=(0.0, float("inf")))
        probs[sa_name] = SA1(
            childs=[varis[sa_name]], parents=[],
            ln_median=ln_median, sigma=sigma, device=device,
        )

        median, beta = _median_extensive(network["edges"][edge_id]["bridge_class"])
        varis[x_name] = variable.Variable(name=x_name, values=list(USABILITY_STATES))
        probs[x_name] = X(
            childs=[varis[x_name]], parents=[varis[sa_name]],
            median=median, beta=beta, device=device,
        )

    service_vars = []
    for origin in origins:
        name = f"S::{origin}"
        varis[name] = variable.Variable(name=name, values=list(SERVICE_STATES))
        service_vars.append(varis[name])

    probs["S"] = S(
        childs=service_vars,
        parents=[varis[f"X::{e}"] for e in edge_order],
        network=network, edge_order=edge_order,
        origins=origins, destinations=destinations, device=device,
    )

    weights = [float(network["nodes"][o].get("population", 0.0)) for o in origins]
    has_population = sum(weights) > 0
    if has_population:
        varis["PopFraction"] = variable.Variable(name="PopFraction", values=(0.0, 1.0))
        probs["PopFraction"] = PopulationAccess(
            childs=[varis["PopFraction"]], parents=service_vars,
            weights=weights, device=device,
        )

    meta = {
        "edge_order": edge_order,
        "origins": origins,
        "destinations": destinations,
        "has_population": has_population,
        "n_variables": len(varis),
        "n_nodes": len(probs),
    }
    return varis, probs, meta


def query_nodes_for(meta: Mapping[str, Any]) -> list[str]:
    """The nodes a forward query needs sampled."""
    nodes = ["S"] + [f"X::{edge_id}" for edge_id in meta["edge_order"]]
    if meta["has_population"]:
        nodes.append("PopFraction")
    return nodes
