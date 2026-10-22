"""System kernels: edge states -> network performance, evaluated by ndtools.

The network's tables become ndtools' ``G_base``: nodes with their attributes,
and one graph edge per Edge row carrying its ``eid`` and a ``time_h`` weight
(``length_km / avg_speed_kph``). ndtools is then called with ``avg_speed=1``
and ``length_attr="time_h"``, so travel times come out in hours with each
edge's own speed. A sample's ``EdgeOpen`` values are its ``comps_state``.

The output is the binary system state of each origin. Each sample is first
classified by the origin's stored rsr rules that hold at the system event's
threshold: exact ones, or survival rules from a stricter threshold and failure
rules from a looser one (``bn_builder.choose_rules``). ndtools evaluates only what
they leave unclassified, and ``tally`` counts how much that was, so the engine can
say when the rules need extending. The travel time is internal to the judgement.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import torch

from . import Probability


class Tally(dict):
    """Samples seen and left unclassified, per origin row. tbnpy deep-copies its probability
    objects before sampling; the copies keep counting into this same tally."""

    def __deepcopy__(self, memo: dict) -> "Tally":
        return self


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
        events = tables["SystemEvent"]
        if len(events) != 1:
            raise ValueError(f"a system function judges exactly one system event, got {len(events)}")
        self.target_add_h = events[0]["target_add_h"]
        rows = len(self.spec["rows"])
        self.refs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * rows
        if self.directory:
            for j, ref in enumerate(self.spec.get("references", [])):
                if ref:
                    # A side with no rules that hold at this threshold is an empty set: it classifies nothing.
                    empty = torch.empty((0, len(self.edges(j)), 2), dtype=torch.int32, device=self.device)
                    self.refs[j] = tuple(
                        torch.load(self.directory / ref[k], map_location=self.device, weights_only=True)
                        if ref.get(k) else empty for k in ("upper", "lower"))
        self.tally = Tally(seen=[0] * rows, unclassified=[0] * rows)
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

        origin, target_add_h = self.spec["rows"][j]["id"], self.target_add_h

        def evaluate(comps: dict[str, int]) -> tuple[float, int, None]:
            time_h, state, _ = fbg.eval_travel_time_to_nearest(
                comps, self.graph, origin, self.destinations, avg_speed=1.0, target_max=target_add_h,
                length_attr="time_h")
            return (float("inf") if time_h is None else float(time_h)), int(state), None

        return evaluate

    def evaluate(self, pars: torch.Tensor) -> torch.Tensor:
        if ((pars != 0) & (pars != 1)).any():
            raise ValueError("edge states must be binary")
        out = torch.empty((len(pars), len(self.childs)), dtype=torch.float64, device=self.device)
        for j, columns in enumerate(self.spec["input_columns"]):
            x = pars[:, columns["open"]].long()
            unknown = torch.ones(len(x), dtype=torch.bool, device=self.device)
            if self.refs[j] is not None:
                from rsr.rsr import classify_samples_with_indices

                result = classify_samples_with_indices(
                    torch.nn.functional.one_hot(x, 2).to(torch.int32), *self.refs[j], return_masks=True)
                out[result["mask_upper"], j] = 1
                out[result["mask_lower"], j] = 0
                unknown = result["mask_unknown"]
            rows = torch.where(unknown)[0]
            self.tally["seen"][j] += len(x)
            self.tally["unclassified"][j] += len(rows)
            if len(rows):
                # Only what the rules leave unclassified goes to ndtools, once per distinct edge state.
                distinct, inverse = torch.unique(x[rows], dim=0, return_inverse=True)
                evaluate, names = self.system_function(j), self.edges(j)
                states = self.tensor([evaluate(dict(zip(names, s)))[1] for s in distinct.tolist()])
                out[rows, j] = states[inverse]
        return out
