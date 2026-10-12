"""Hazard-to-component probability functions used by the pilot."""

from __future__ import annotations

from math import exp, hypot, log, sqrt
from typing import Any

import numpy as np
from scipy.special import ndtr

from src.fragility.hazus61.data import curves


_GMPE = {
    "c1": -0.6104,
    "c2": 0.451,
    "c3": -0.2090,
    "c4": -1.158,
    "c5": -0.00255,
    "c6": 0.000141,
    "c7": 0.299,
    "c8": 0.503,
}


def _median_extensive(bridge_class: str) -> tuple[float, float]:
    selected = [
        curve
        for curve in curves("highway_bridge", bridge_class, "ground_shaking")
        if curve["damage_state"] == "extensive"
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one Extensive curve for {bridge_class}")
    return float(selected[0]["median"]), float(selected[0]["beta_ln"])


def _gmpe_ln_sa1(magnitude: float, distance_km: float) -> float:
    """Reproduce the EMA dataset's Campbell (2003) T=1 s implementation."""
    c = _GMPE
    distance_km = max(float(distance_km), 1e-9)
    f1 = c["c2"] * magnitude + c["c3"] * (8.5 - magnitude) ** 2
    radius = sqrt(distance_km**2 + (c["c7"] * exp(c["c8"] * magnitude)) ** 2)
    f2 = c["c4"] * log(radius) + (c["c5"] + c["c6"] * magnitude) * distance_km
    if distance_km <= 70.0:
        f3 = 0.0
    elif distance_km <= 130.0:
        f3 = c["c7"] * (log(distance_km) - log(70.0))
    else:
        f3 = c["c7"] * (log(distance_km) - log(70.0)) + c["c8"] * (
            log(distance_km) - log(130.0)
        )
    return c["c1"] + f1 + f2 + f3


def _aleatory_sigma(magnitude: float) -> float:
    return 1.110 - 0.0793 * magnitude if magnitude < 7.16 else 0.543


def _edge_midpoint(edge: dict[str, Any], nodes: dict[str, Any]) -> tuple[float, float]:
    start, end = nodes[edge["from"]], nodes[edge["to"]]
    return (float(start["x"]) + float(end["x"])) / 2, (
        float(start["y"]) + float(end["y"])
    ) / 2


def component_probabilities(
    network: dict[str, Any], hazard: dict[str, Any]
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Return P(unusable), representative SA1, and traceable warnings per edge."""
    model = hazard.get("model")
    if model not in network["hazard_models"]:
        raise ValueError(
            f"hazard model {model!r} is not available; choose from {network['hazard_models']}"
        )
    failures: dict[str, float] = {}
    intensities: dict[str, float] = {}
    warnings: list[str] = []

    if model == "uniform_sa1":
        sa1 = float(hazard.get("sa1_g", 0.0))
        if not np.isfinite(sa1) or sa1 < 0:
            raise ValueError("sa1_g must be finite and nonnegative")
        for edge_id, edge in network["edges"].items():
            median, beta = _median_extensive(edge["bridge_class"])
            failures[edge_id] = float(ndtr((log(sa1) - log(median)) / beta)) if sa1 else 0.0
            intensities[edge_id] = sa1
        return failures, intensities, warnings

    if model == "shakemap":
        from .shakemap import site_terms

        terms, warnings, _ = site_terms(network, hazard)
        for edge_id, edge in network["edges"].items():
            ln_sa, sigma = terms[edge_id]
            median, beta = _median_extensive(edge["bridge_class"])
            failures[edge_id] = float(ndtr((ln_sa - log(median)) / sqrt(sigma**2 + beta**2)))
            intensities[edge_id] = float(exp(ln_sa))
        return failures, intensities, warnings

    magnitude = float(hazard.get("magnitude", 8.0))
    epicentre_x = float(hazard.get("epicentre_x", 1.0))
    epicentre_y = float(hazard.get("epicentre_y", 10.0))
    if not 4.0 <= magnitude <= 9.5:
        raise ValueError("magnitude must lie between 4.0 and 9.5 for this pilot")
    sigma_a = _aleatory_sigma(magnitude)
    sigma_e = 0.3
    for edge_id, edge in network["edges"].items():
        x, y = _edge_midpoint(edge, network["nodes"])
        distance = hypot(x - epicentre_x, y - epicentre_y)
        ln_sa = _gmpe_ln_sa1(magnitude, distance)
        median, beta = _median_extensive(edge["bridge_class"])
        denom = sqrt(sigma_a**2 + sigma_e**2 + beta**2)
        failures[edge_id] = float(ndtr((ln_sa - log(median)) / denom))
        intensities[edge_id] = float(exp(ln_sa))
    warnings.append(
        "EMA reference scenario reproduces the dataset script; its 0.3 epistemic sigma is explicitly a placeholder."
    )
    warnings.append(
        "The marginal GMPE and fragility variances are combined analytically, then component states are sampled independently."
    )
    return failures, intensities, warnings
