"""Run a forward query by sampling the tbnpy network.

Produces exactly the result shape `engine.run_forward_query` produces, so the
UI, the packages and the natural-language layer are unaffected by which backend
answered.

What differs from the analytic backend, and what does not:

* **Does not differ:** the per-bridge failure probabilities. That marginal has a
  closed form, so both backends report the closed form rather than a sampled
  estimate of it -- a number you can compute exactly should not arrive with Monte
  Carlo noise on it.
* **Does differ:** how the system level is reached. The analytic backend samples
  bridge states straight from those marginals. This one samples ground motion,
  then usability given ground motion, then service given usability, through the
  real network. Today the two give the same distribution, which is the point of
  the equivalence test. Tomorrow they do not: a correlated residual, evidence on
  a bridge, or an intervention node all attach to this one and to nothing in the
  analytic path.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from tbnpy import inference

from ..hazard import component_probabilities
from .model import build, query_nodes_for


def run(
    network: Mapping[str, Any],
    hazard: Mapping[str, Any],
    origins: Sequence[str],
    destinations: Sequence[str],
    samples: int,
    seed: int,
    wilson,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)          # reproducible given the seed, like the other backend
    varis, probs, meta = build(network, hazard, origins, destinations)
    out = inference.sample(probs, query_nodes_for(meta), int(samples))

    service = out["S"].Cs.detach().cpu().numpy()      # (n_sample, n_origins)
    counts = service.sum(axis=0)

    origin_results = [
        {
            "origin": origin,
            "label": network["nodes"][origin].get("city")
            or network["nodes"][origin].get("label")
            or origin,
            "probability": float(counts[i]) / samples,
            "ci95": wilson(int(counts[i]), samples),
            "population": float(network["nodes"][origin].get("population", 0.0)),
        }
        for i, origin in enumerate(meta["origins"])
    ]
    origin_results.sort(key=lambda item: (item["probability"], item["origin"]))

    failure, intensity, warnings = component_probabilities(network, hazard)
    components = [
        {
            "edge_id": edge_id,
            "from": edge["from"],
            "to": edge["to"],
            "bridge_class": edge["bridge_class"],
            "p_unusable": failure[edge_id],
            "sa1_g": intensity[edge_id],
        }
        for edge_id, edge in network["edges"].items()
    ]
    components.sort(key=lambda item: item["p_unusable"], reverse=True)

    result: dict[str, Any] = {
        "case_id": network["case_id"],
        "inference": "forward_tbnpy_sampling",
        "backend": "tbn",
        "model_size": {"variables": meta["n_variables"], "nodes": meta["n_nodes"]},
        "hazard": dict(hazard),
        "samples": int(samples),
        "seed": seed,
        "destinations": list(meta["destinations"]),
        "origin_service": origin_results,
        "components": components,
        "warnings": warnings,
    }

    if meta["has_population"]:
        fraction = out["PopFraction"].Cs.detach().cpu().numpy().astype(float)
        mean = float(fraction.mean())
        result["population_accessibility"] = {
            "expected_fraction": mean,
            "monte_carlo_standard_error": float(fraction.std(ddof=0) / np.sqrt(samples)),
            "population_in_scope": float(
                sum(network["nodes"][o].get("population", 0.0) for o in meta["origins"])
            ),
        }
    return result
