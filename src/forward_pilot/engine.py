"""Forward-only simulation engine for the assembled parametric BN."""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np

from .hazard import component_probabilities


class ForwardQueryError(ValueError):
    """A query lies outside the declared forward-inference scope."""


FORWARD_QUERY_TYPES = {"connectivity", "population_accessibility"}
BLOCKED_QUERY_TYPES = {"critical_components", "diagnosis", "backward_inference"}


def _validate_query(network: dict[str, Any], query: dict[str, Any]) -> tuple[list[str], list[str], int, int]:
    query_type = query.get("query_type", "connectivity")
    if query_type in BLOCKED_QUERY_TYPES or query.get("evidence"):
        raise ForwardQueryError(
            "This pilot supports forward inference only. Evidence-conditioned diagnosis and "
            "component criticality are outside the current query scope."
        )
    if query_type not in FORWARD_QUERY_TYPES:
        raise ForwardQueryError(f"unsupported query_type {query_type!r}; choose from {sorted(FORWARD_QUERY_TYPES)}")
    node_ids = set(network["nodes"])
    origins = list(query.get("origins") or network["default_origins"])
    destinations = list(query.get("destinations") or network["destinations"])
    unknown = (set(origins) | set(destinations)) - node_ids
    if unknown:
        raise ForwardQueryError(f"unknown network nodes: {sorted(unknown)}")
    if not origins or not destinations:
        raise ForwardQueryError("at least one origin and one destination are required")
    samples = int(query.get("samples", 5000))
    if not 100 <= samples <= 50000:
        raise ForwardQueryError("samples must be between 100 and 50000")
    seed = int(query.get("seed", 20261004))
    return origins, destinations, samples, seed


def _reachable_nodes(
    nodes: Iterable[str],
    edge_items: list[tuple[str, dict[str, Any]]],
    usable: np.ndarray,
    destinations: list[str],
    directed: bool,
) -> set[str]:
    """Nodes that can reach any destination in one sampled component state."""
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for (_, edge), live in zip(edge_items, usable):
        if not live:
            continue
        start, end = edge["from"], edge["to"]
        if directed:
            adjacency[end].append(start)  # reverse search: who can reach the destination?
        else:
            adjacency[start].append(end)
            adjacency[end].append(start)
    reached = set(destinations)
    queue = deque(destinations)
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in reached:
                reached.add(neighbour)
                queue.append(neighbour)
    return reached


def _wilson(successes: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def tbn_available() -> bool:
    """Whether the tbnpy backend can run here.

    tbnpy needs torch, which is a large dependency and may not be installed. The
    analytic backend is kept for exactly that case, so the pilot still runs.
    """
    from importlib.util import find_spec

    return find_spec("tbnpy") is not None and find_spec("torch") is not None


def run_forward_query(network: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    origins, destinations, samples, seed = _validate_query(network, query)
    hazard = dict(network["default_hazard"])
    hazard.update(query.get("hazard") or {})

    # The tbnpy network is the model; the analytic path is the fallback. Both are
    # kept because an equivalence test between them is the cheapest way to notice
    # if a change to one quietly alters the answers.
    backend = query.get("backend") or ("tbn" if tbn_available() else "analytic")
    if backend == "tbn":
        if not tbn_available():
            raise ForwardQueryError(
                "the tbnpy backend needs tbnpy and torch installed; "
                "pass backend='analytic' to use the closed-form path instead"
            )
        from .tbn.run import run as run_tbn

        try:
            result = run_tbn(network, hazard, origins, destinations, samples, seed, _wilson)
            result["query_type"] = query.get("query_type", "connectivity")
            return result
        except ValueError as exc:
            raise ForwardQueryError(str(exc)) from exc
    if backend != "analytic":
        raise ForwardQueryError(f"unknown backend {backend!r}; choose 'tbn' or 'analytic'")
    try:
        failure, intensity, warnings = component_probabilities(network, hazard)
    except ValueError as exc:
        raise ForwardQueryError(str(exc)) from exc

    edge_items = list(network["edges"].items())
    p_fail = np.asarray([failure[edge_id] for edge_id, _ in edge_items])
    rng = np.random.default_rng(seed)
    counts = {origin: 0 for origin in origins}
    population_sum = 0.0
    population_sq_sum = 0.0
    weights = {
        origin: float(network["nodes"][origin].get("population", 0.0)) for origin in origins
    }
    total_population = sum(weights.values())

    # Chunking bounds memory for the 50,000-sample upper limit.
    completed = 0
    while completed < samples:
        batch = min(2000, samples - completed)
        states = rng.random((batch, len(edge_items))) >= p_fail
        for row in states:
            reached = _reachable_nodes(
                network["nodes"], edge_items, row, destinations, network["directed"]
            )
            for origin in origins:
                counts[origin] += int(origin in reached)
            if total_population > 0:
                fraction = sum(weights[o] for o in origins if o in reached) / total_population
                population_sum += fraction
                population_sq_sum += fraction * fraction
        completed += batch

    origin_results = [
        {
            "origin": origin,
            "label": network["nodes"][origin].get("city")
            or network["nodes"][origin].get("label")
            or origin,
            "probability": counts[origin] / samples,
            "ci95": _wilson(counts[origin], samples),
            "population": weights[origin],
        }
        for origin in origins
    ]
    origin_results.sort(key=lambda item: (item["probability"], item["origin"]))
    components = [
        {
            "edge_id": edge_id,
            "from": edge["from"],
            "to": edge["to"],
            "bridge_class": edge["bridge_class"],
            "p_unusable": failure[edge_id],
            "sa1_g": intensity[edge_id],
        }
        for edge_id, edge in edge_items
    ]
    components.sort(key=lambda item: item["p_unusable"], reverse=True)

    result: dict[str, Any] = {
        "case_id": network["case_id"],
        "query_type": query.get("query_type", "connectivity"),
        "inference": "forward_monte_carlo",
        "backend": "analytic",
        "hazard": hazard,
        "samples": samples,
        "seed": seed,
        "destinations": destinations,
        "origin_service": origin_results,
        "components": components,
        "warnings": warnings,
    }
    if total_population > 0:
        mean = population_sum / samples
        variance = max(0.0, population_sq_sum / samples - mean * mean)
        result["population_accessibility"] = {
            "expected_fraction": mean,
            "monte_carlo_standard_error": float(np.sqrt(variance / samples)),
            "population_in_scope": total_population,
        }
    return result
