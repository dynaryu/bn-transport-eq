"""System kernels: edge states -> network performance, evaluated by ndtools.

The network's tables become ndtools' ``G_base``: nodes with their attributes,
and one graph edge per Edge row carrying its ``eid`` and a ``time_h`` weight
(``length_km / avg_speed_kph``). ndtools is then called with ``avg_speed=1``
and ``length_attr="time_h"``, so travel times come out in hours with each
edge's own speed. A sample's ``EdgeOpen`` values are its ``comps_state``.

Each sample is first classified by the origin's stored rsr rules that hold at
the system event's threshold: exact ones, or survival rules from a stricter
threshold and failure rules from a looser one (``bn_builder.choose_rules``).
ndtools evaluates only what they leave unclassified, and ``tally`` counts how
much that was, so the engine can say when the rules need extending.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import torch

from . import Probability


class Tally(dict):
    """Samples seen and left unclassified. tbnpy deep-copies its probability objects
    before sampling; the copies keep counting into this same tally."""

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
    """Whether an origin still reaches a key destination in time, given every edge's state (deterministic).

        S = 1 if t(origin -> nearest key destination | edge states) <= t(same, intact network) + target_add_h
        S = 0 otherwise

    The travel time t comes from ndtools.fun_binary_graph.eval_travel_time_to_nearest.
    """

    def __init__(self, childs, parents, origin, edge_ids, target_add_h, network, rules=None,
                 directory=None, device="cpu"):
        """
        childs       : [Variable S], values [0, 1]
        parents      : [Variable EdgeOpen] of every edge, values [0, 1]
        origin       : the origin node id
        edge_ids     : the network's edge id of each parent, in the same order
        target_add_h : hours allowed on top of the intact network's travel time
        network      : the Node and Edge tables, {"Node": [...], "Edge": [...]}, or the name of a
                       JSON file holding them, relative to directory. Nodes marked key_destination
                       are the destinations.
        rules        : the stored rsr rules that hold at target_add_h, or None:
                       {"upper": survival rules file, "lower": failure rules file} relative to
                       directory, each a (m, n_edges, 2) tensor or null for none, with the threshold
                       each was extracted at ("upper_at", "lower_at"). An edge state matching a
                       survival rule is S = 1, a failure rule S = 0.
        directory    : the folder file names are relative to (tbnpy passes the spec's folder)
        """
        super().__init__(childs, parents, device)
        assert len(childs) == 1, "TravelTimeToNearest: one S child"
        assert len(edge_ids) == len(parents), "one edge id per EdgeOpen parent"

        if isinstance(network, str):
            network = json.loads((Path(directory) / network).read_text(encoding="utf-8"))
        self.graph = base_graph(network["Node"], network["Edge"])
        self.destinations = [node["id"] for node in network["Node"] if node["key_destination"]]
        if not self.destinations:
            raise ValueError("no node is a key_destination, so there is nothing to travel to")

        self.origin = origin
        self.edge_ids = list(edge_ids)
        self.target_add_h = float(target_add_h)
        self.rules_reference = rules
        self.rules = None if rules is None else self._load_rules(rules, Path(directory))
        self.tally = Tally(seen=0, unclassified=0)

        intact = self.system_function()(dict.fromkeys(self.edge_ids, 1))
        if intact[1] == 0:
            self.notes.add(f"{origin} has no route to a key destination even with every edge open, "
                           f"so it always fails")

    def _load_rules(self, rules, directory):
        """(survival, failure) tensors; a side with no rules that hold here is an empty set: it classifies nothing."""
        empty = torch.empty((0, len(self.edge_ids), 2), dtype=torch.int32, device=self.device)
        return tuple(torch.load(directory / rules[side], map_location=self.device, weights_only=True)
                     if rules.get(side) else empty for side in ("upper", "lower"))

    # ------------------------------------------------------------------
    def form(self):
        return (f"1 if {self.origin} reaches its nearest key destination within {self.target_add_h:g} h "
                f"of the intact network's time, else 0 (ndtools, after the stored rules)")

    def form_details(self):
        rules = ("none, so ndtools evaluates every sample" if self.rules is None else
                 f"{len(self.rules[0])} survival and {len(self.rules[1])} failure reference sets")
        return [f"key destinations: {', '.join(self.destinations)}", f"rules: {rules}"] + super().form_details()

    # ------------------------------------------------------------------
    def system_function(self):
        """ndtools' judgement: {edge id: 0 or 1} -> (travel time in hours, S, None)."""
        from ndtools import fun_binary_graph as fbg

        def evaluate(comps: dict[str, int]) -> tuple[float, int, None]:
            time_h, state, _ = fbg.eval_travel_time_to_nearest(
                comps, self.graph, self.origin, self.destinations, avg_speed=1.0,
                target_max=self.target_add_h, length_attr="time_h")
            return (float("inf") if time_h is None else float(time_h)), int(state), None

        return evaluate

    def evaluate(self, Cs_pars):
        """
        Cs_pars : (N, n_edges) EdgeOpen of each edge, in the order of edge_ids
        Returns (N, 1) S
        """
        if ((Cs_pars != 0) & (Cs_pars != 1)).any():
            raise ValueError("edge states must be binary")
        x = Cs_pars.long()
        S = torch.empty(len(x), dtype=torch.float64, device=self.device)

        # 1. The stored rules classify what they can.
        unknown = torch.ones(len(x), dtype=torch.bool, device=self.device)
        if self.rules is not None:
            from rsr.rsr import classify_samples_with_indices

            result = classify_samples_with_indices(
                torch.nn.functional.one_hot(x, 2).to(torch.int32), *self.rules, return_masks=True)
            S[result["mask_upper"]] = 1
            S[result["mask_lower"]] = 0
            unknown = result["mask_unknown"]

        # 2. ndtools decides the rest, once per distinct edge state.
        rows = torch.where(unknown)[0]
        self.tally["seen"] += len(x)
        self.tally["unclassified"] += len(rows)
        if len(rows):
            distinct, inverse = torch.unique(x[rows], dim=0, return_inverse=True)
            evaluate = self.system_function()
            states = self.tensor([evaluate(dict(zip(self.edge_ids, s)))[1] for s in distinct.tolist()])
            S[rows] = states[inverse]
        return S[:, None]
