"""Export the assembled network's real structure, for drawing.

Read off the live ``probs`` dict rather than described by hand, so the picture
cannot drift from the model: if a node gains a parent, the drawing gains an arrow
without anyone remembering to update it.

**Why a slice and not the whole thing.** The EMA model has 331 variables. Drawing
all of them produces a hairball that communicates "it is big" and nothing else.
The repeated structure is the point -- every bridge has the identical three-node
chain -- so one readable slice of a few bridges, plus the true totals, says more
than the full graph does. The plate view in the UI is the complement: it shows the
repetition as repetition, this shows what a plate actually stands for.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .model import build

# What each node family means, for the drawing's legend and tooltips.
_KINDS = {
    "SA1": ("hazard", "Ground motion at this bridge's site, in g. "
                      "Lognormal about the scenario's median; a root, because the "
                      "scenario is an input rather than an uncertain quantity."),
    "X": ("asset", "Usable or not. Hazus Extensive-limit-state fragility evaluated "
                   "at this bridge's ground motion."),
    "S": ("system", "Service for every origin at once: does it still reach a "
                    "destination? Deterministic given the bridge states."),
    "PopFraction": ("impact", "Share of the population in scope that still reaches a "
                              "destination. The decision metric."),
}


def _kind(node_name: str) -> tuple[str, str]:
    head = node_name.split("::", 1)[0]
    return _KINDS.get(head, ("other", ""))


def structure(
    network: Mapping[str, Any],
    hazard: Mapping[str, Any] | None = None,
    edge_limit: int = 3,
    origin_limit: int = 3,
) -> dict[str, Any]:
    """A readable slice of the real DAG, plus the totals it was taken from."""
    hazard = dict(hazard or network["default_hazard"])
    varis, probs, meta = build(network, hazard)

    shown_edges = meta["edge_order"][:max(1, edge_limit)]
    shown_origins = meta["origins"][:max(1, origin_limit)]
    keep = {f"SA1::{e}" for e in shown_edges} | {f"X::{e}" for e in shown_edges}
    keep |= {"S"} | ({"PopFraction"} if meta["has_population"] else set())

    # Which variables survive the slice, so parent lists can be trimmed honestly.
    shown_vars = set()
    for name in keep:
        shown_vars.update(v.name for v in probs[name].childs)
    kept_service = {f"S::{o}" for o in shown_origins}

    nodes = []
    for name in sorted(keep, key=lambda n: (_order(n), n)):
        node = probs[name]
        kind, description = _kind(name)
        childs = [v.name for v in node.childs]
        parents = [v.name for v in node.parents]
        visible_parents = [p for p in parents if p in shown_vars]
        if name == "PopFraction":
            visible_parents = [p for p in parents if p in kept_service]
        nodes.append({
            "id": name,
            "kind": kind,
            "description": description,
            "childs": childs if name != "S" else sorted(kept_service),
            "n_childs": len(childs),
            "parents": visible_parents,
            "n_parents": len(parents),
            "hidden_parents": len(parents) - len(visible_parents),
            "deterministic": kind in {"system", "impact"},
            "root": not parents,
        })

    return {
        "case_id": network["case_id"],
        "hazard": hazard,
        "nodes": nodes,
        "shown": {"edges": shown_edges, "origins": shown_origins},
        "totals": {
            "variables": meta["n_variables"],
            "nodes": meta["n_nodes"],
            "bridges": len(meta["edge_order"]),
            "origins": len(meta["origins"]),
            "has_population": meta["has_population"],
        },
        "note": (
            f"Showing {len(shown_edges)} of {len(meta['edge_order'])} bridges. "
            "Every other bridge repeats the same two-node chain into the same "
            "service node."
        ),
    }


def _order(node_name: str) -> int:
    """Left-to-right position of each family, matching the flow of evidence."""
    return {"SA1": 0, "X": 1, "S": 2, "PopFraction": 3}.get(node_name.split("::", 1)[0], 9)
