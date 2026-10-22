"""System kernels: edge states -> network performance, evaluated by ndtools.

The network's tables become ndtools' ``G_base``: nodes with their attributes,
and one graph edge per Edge row carrying its ``eid`` and a ``time_h`` weight
(``length_km / avg_speed_kph``). ndtools is then called with ``avg_speed=1``
and ``length_attr="time_h"``, so travel times come out in hours with each
edge's own speed. A sample's ``EdgeOpen`` values are its ``comps_state``.

The binding's two outputs are two probability objects (``spec["output"]``), so
a question about the state alone never computes times: the state object first
classifies each sample with the origin's stored rsr rules, and evaluates only
what they leave unclassified.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import torch

from . import Probability


def base_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> nx.Graph:
    """ndtools' G_base from the Node and Edge tables: a DiGraph if any edge is directed."""
    graph = nx.DiGraph() if any(e["directed"] for e in edges) else nx.Graph()
    for node in nodes:
        graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in edges:
        if graph.is_directed() and not edge["directed"]:
            raise ValueError(f"edge {edge['id']} is undirected in a directed network; give it two directed edges")
        graph.add_edge(edge["source"], edge["target"], eid=edge["id"],
                       time_h=edge["length_km"] / edge["avg_speed_kph"])
    return graph


class TravelTimeToNearest(Probability):
    """ndtools.fun_binary_graph.eval_travel_time_to_nearest, for each origin row."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        tables = self.spec["tables"]
        self.graph = base_graph(tables["Node"], tables["Edge"])
        self.destinations = [r["id"] for r in tables["Node"] if r["key_destination"]]
        if not self.destinations:
            raise ValueError("no node is a key_destination, so there is nothing to travel to")
        self.refs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.spec["rows"])
        if self.spec["output"] == "state" and self.directory:
            for j, ref in enumerate(self.spec.get("references", [])):
                if ref:
                    self.refs[j] = tuple(torch.load(self.directory / ref[k], map_location=self.device,
                                                    weights_only=True) for k in ("upper", "lower"))
        never = [row["id"] for j, row in enumerate(self.spec["rows"])
                 if self.system_function(j)(dict.fromkeys(self.edges(j), 1))[1] == 0]
        if never:
            self.notes.add(f"{len(never)} origin(s) fail even with every edge open (no route to a key "
                           f"destination): {', '.join(never[:5])}")

    def edges(self, j: int) -> list[str]:
        return self.spec["parent_rows"]["open"][j]

    def system_function(self, j: int):
        """(travel time in hours, state, None) for origin row j, from a {edge id: 0/1} dict."""
        from ndtools import fun_binary_graph as fbg

        origin = self.spec["rows"][j]["id"]
        target_max = self.spec["parameters"][j]["target_max"]

        def evaluate(comps: dict[str, int]) -> tuple[float, int, None]:
            time_h, state, _ = fbg.eval_travel_time_to_nearest(
                comps, self.graph, origin, self.destinations, avg_speed=1.0, target_max=target_max,
                length_attr="time_h")
            return (float("inf") if time_h is None else float(time_h)), int(state), None

        return evaluate

    def evaluate(self, pars: torch.Tensor) -> torch.Tensor:
        if ((pars != 0) & (pars != 1)).any():
            raise ValueError("edge states must be binary")
        unique, inverse = torch.unique(pars, dim=0, return_inverse=True)    # identical states, identical answers
        out = torch.empty((len(unique), len(self.childs)), dtype=torch.float64, device=self.device)
        want = 0 if self.spec["output"] == "travel_time" else 1
        for j, columns in enumerate(self.spec["input_columns"]):
            x = unique[:, columns["open"]].long()
            unknown = torch.ones(len(x), dtype=torch.bool, device=self.device)
            if self.refs[j] is not None:
                from rsr.rsr import classify_samples_with_indices

                result = classify_samples_with_indices(
                    torch.nn.functional.one_hot(x, 2).to(torch.int32), *self.refs[j], return_masks=True)
                out[result["mask_upper"], j] = 1
                out[result["mask_lower"], j] = 0
                unknown = result["mask_unknown"]
            evaluate, names = self.system_function(j), self.edges(j)
            for i in torch.where(unknown)[0].tolist():
                out[i, j] = evaluate(dict(zip(names, x[i].tolist())))[want]
        return out[inverse]
