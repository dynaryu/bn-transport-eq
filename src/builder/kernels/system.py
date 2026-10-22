"""System kernels: edge states -> network performance, evaluated by ndtools.

The network's tables become ndtools' ``G_base``: nodes with their attributes,
and one graph edge per Edge row carrying its ``eid`` and a ``time_h`` weight
(``length_km / avg_speed_kph``). ndtools is then called with ``avg_speed=1``
and ``length_attr="time_h"``, so travel times come out in hours with each
edge's own speed. A sample's ``EdgeOpen`` values are its ``comps_state``.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from ndtools import fun_binary_graph as fbg

from . import Call


def base_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> nx.Graph:
    """ndtools' G_base from the Node and Edge tables: a DiGraph if any edge is directed."""
    graph = nx.DiGraph() if any(e["directed"] for e in edges) else nx.Graph()
    for node in nodes:
        graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in edges:
        if graph.is_directed() and not edge["directed"]:
            raise ValueError(f"link {edge['id']} is undirected in a directed network; give it two directed edges")
        graph.add_edge(edge["source"], edge["target"], eid=edge["id"],
                       time_h=edge["length_km"] / edge["avg_speed_kph"])
    return graph


class TravelTimeToNearest:
    """ndtools.fun_binary_graph.eval_travel_time_to_nearest, for each origin row, per sample."""

    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        graph = base_graph(call.tables["Node"], call.tables["Edge"])
        destinations = [n["id"] for n in call.tables["Node"] if n["key_destination"]]
        if not destinations:
            raise ValueError("no node is a key_destination, so there is nothing to travel to")
        links = call.parents["open"][0]                     # via *: the same edge list for every origin
        open_ = call.inputs["open"][0].astype(int)          # (n, links)
        origins = [row["id"] for row in call.rows]
        target_max = call.params["target_max"]

        def evaluate(states: np.ndarray) -> list[tuple[float, int]]:
            comps = dict(zip(links, states.tolist()))
            out = []
            for origin in origins:
                time_h, state, _ = fbg.eval_travel_time_to_nearest(
                    comps, graph, origin, destinations, avg_speed=1.0, target_max=target_max, length_attr="time_h")
                out.append((np.inf if time_h is None else float(time_h), int(state)))
            return out

        intact = evaluate(np.ones(len(links), int))
        never = [o for o, (_, s) in zip(origins, intact) if s == 0]
        if never:
            call.warnings.append(f"{len(never)} origin(s) fail even with every edge open (no route to a key "
                                 f"destination): {', '.join(never[:5])}")

        times = np.empty((call.n, len(origins)))
        states = np.empty((call.n, len(origins)), dtype=int)
        cache: dict[bytes, list[tuple[float, int]]] = {}    # identical edge states, identical answers
        for s in range(call.n):
            key = open_[s].tobytes()
            if key not in cache:
                cache[key] = evaluate(open_[s])
            times[s], states[s] = zip(*cache[key])
        return {"travel_time": times, "state": states}
