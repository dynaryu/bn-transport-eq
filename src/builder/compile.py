"""Compile an assembly into a Bayesian network: replicate, then wire.

Each binding names a function and the rows it describes. Compiling it

1. replicates the function once per row of its `per` plate (filtered by `where`),
   giving one node ``<Quantity>::<row id>`` per output;
2. resolves each row's parameter set, so ``"hazus61.{hazus_class}"`` hands every
   bridge the curve for its own class, and ``parameter_set_by_location`` hands it the
   first listed set whose region contains it; then each "{imt}" port becomes the IM
   that set names;
3. resolves each input's parents by following its `via` through the network's
   foreign keys, then finds the one node producing that quantity on that row.

Wiring is by quantity, so an output reaches an input only when the types agree,
and every (quantity, row) must have exactly one producer: two hazard bindings
over the same sites is an error, as is a fragility curve whose sites nobody
shakes. The binding graph must be acyclic; that order is the sampling order.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .load import (Library, Network, RegistryError, load_assembly, load_library, load_network, resolve_quantities,
                   via_kind)

_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def node_id(quantity: str, row_id: str) -> str:
    return f"{quantity}::{row_id}"


@dataclass
class Binding:
    index: int
    spec: dict[str, Any]                     # the assembly entry
    function: dict[str, Any]
    rows: list[dict[str, Any]]
    parameter_sets: list[str | None]         # per row; None for inline or no parameters
    parameters: list[dict[str, Any] | None]  # per row, resolved values
    quantities: list[dict[str, str]]         # per row, port name -> quantity, "{imt}" resolved
    parents: dict[str, list[list[str]]]      # input name -> per row, the parent row ids
    upstream: set[int]

    @property
    def label(self) -> str:
        return f"#{self.index} {self.function['id']}"

    def port_quantity(self, name: str) -> str:
        """The port's quantity across this binding's rows, e.g. "SA(1.0)", or "PGA | SA(1.0)" if they differ."""
        return " | ".join(dict.fromkeys(q[name] for q in self.quantities))


@dataclass
class Compiled:
    assembly: dict[str, Any]
    library: Library
    network: Network
    bindings: list[Binding]
    producer: dict[tuple[str, str], int]     # (quantity, row id) -> binding index
    order: list[int]

    @property
    def n_nodes(self) -> int:
        return len(self.producer)

    def ancestors(self, indices: set[int]) -> set[int]:
        """These bindings and every binding they read from, directly or not."""
        out, todo = set(), list(indices)
        while todo:
            i = todo.pop()
            if i not in out:
                out.add(i)
                todo.extend(self.bindings[i].upstream)
        return out

    def nodes(self) -> list[dict[str, Any]]:
        """Every BN node, with its parents, in sampling order."""
        out = []
        for i in self.order:
            b = self.bindings[i]
            for r, row in enumerate(b.rows):
                q = b.quantities[r]
                parents = [
                    node_id(q[port["name"]], pid)
                    for port in b.function["inputs"]
                    for pid in b.parents[port["name"]][r]
                ]
                for port in b.function["outputs"]:
                    out.append({
                        "id": node_id(q[port["name"]], row["id"]),
                        "quantity": q[port["name"]],
                        "entity": b.function["per"],
                        "row": row["id"],
                        "function": b.function["id"],
                        "parameter_set": b.parameter_sets[r],
                        "parents": parents,
                    })
        return out

    def summary(self) -> list[dict[str, Any]]:
        out = []
        for i in self.order:
            b = self.bindings[i]
            inputs = []
            for port in b.function["inputs"]:
                counts = [len(p) for p in b.parents[port["name"]]]
                inputs.append({**port, "quantity": b.port_quantity(port["name"]),
                               "parents_min": min(counts), "parents_max": max(counts)})
            out.append({
                "binding": i,
                "function": b.function["id"],
                "per": b.function["per"],
                "rows": len(b.rows),
                "nodes": len(b.rows) * len(b.function["outputs"]),
                "parameter_sets": dict(Counter(s for s in b.parameter_sets if s)),
                "inline_parameters": b.spec.get("parameters"),
                "inputs": inputs,
                "upstream": sorted(b.upstream),
            })
        return out


def _matches(row: dict[str, Any], where: dict[str, Any]) -> bool:
    for attr, wanted in where.items():
        value = row.get(attr)
        if isinstance(wanted, list) and value not in wanted:
            return False
        if not isinstance(wanted, list) and value != wanted:
            return False
    return True


def _few(items: list[str], n: int = 3) -> str:
    return ", ".join(items[:n]) + (f" ... ({len(items)} in all)" if len(items) > n else "")


def contains(polygon: list[list[float]], lat: float, lon: float) -> bool:
    """Whether (lat, lon) lies inside a polygon of [lon, lat] vertices (ray casting; the ring closes itself)."""
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _by_location(choices: list[str], fn: dict[str, Any], rows: list[dict[str, Any]], lib: Library,
                 where_label: str, problems: list[str]) -> tuple[list[str | None], list[dict[str, Any] | None]]:
    """Each row's parameter set: the first of ``choices`` whose region contains the row's lat/lon."""
    sets: list[str | None] = [None] * len(rows)
    values: list[dict[str, Any] | None] = [None] * len(rows)
    before = len(problems)
    for set_id in choices:
        found = lib.parameter_sets.get(set_id)
        if found is None or found["function"] != fn["id"]:
            problems.append(f"{where_label}: no parameter set {set_id!r} for {fn['id']}")
        elif "region" not in found:
            problems.append(f"{where_label}: {set_id} has no region, so it cannot be chosen by location")
    attrs = lib.entities[fn["per"]]["attributes"]
    if not {"lat", "lon"} <= attrs.keys():
        problems.append(f"{where_label}: {fn['per']} has no lat/lon to choose a parameter set by")
    if len(problems) > before:
        return sets, values
    uncovered = []
    for r, row in enumerate(rows):
        chosen = next((s for s in choices
                       if contains(lib.parameter_sets[s]["region"]["polygon"], row["lat"], row["lon"])), None)
        if chosen is None:
            uncovered.append(row["id"])
        else:
            sets[r], values[r] = chosen, lib.parameter_sets[chosen]["values"]
    if uncovered:
        regions = "; ".join(f"{s}: {lib.parameter_sets[s]['region']['name']}" for s in choices)
        problems.append(f"{where_label}: no listed parameter set covers {fn['per']} {_few(uncovered)} ({regions})")
    return sets, values


def compile_assembly(assembly: dict[str, Any], lib: Library, network: Network) -> Compiled:
    problems: list[str] = []
    bindings: list[Binding] = []
    producer: dict[tuple[str, str], int] = {}

    # Pass 1: rows, parameters and outputs of every binding.
    for index, spec in enumerate(assembly["bindings"]):
        where_label = f"{assembly['id']} binding #{index} ({spec['function']})"
        fn = lib.functions.get(spec["function"])
        if fn is None:
            problems.append(f"{where_label}: function is not in the library")
            continue
        per = fn["per"]
        where = spec.get("where", {})
        for attr in where:
            if attr not in lib.entities[per]["attributes"]:
                problems.append(f"{where_label}: where {attr!r} is not an attribute of {per}")
        rows = [r for r in network.rows(per) if _matches(r, where)]
        if not rows:
            problems.append(f"{where_label}: matches no {per} rows in network {network.id}")
            continue

        for attr in fn.get("attributes", []):
            missing = [r["id"] for r in rows if r.get(attr) is None]
            if missing:
                problems.append(f"{where_label}: {fn['id']} reads {per}.{attr}, missing on {_few(missing)}")

        sets: list[str | None] = [None] * len(rows)
        values: list[dict[str, Any] | None] = [None] * len(rows)
        if "parameters" not in fn:
            if any(k in spec for k in ("parameter_set", "parameters", "parameter_set_by_location")):
                problems.append(f"{where_label}: {fn['id']} takes no parameters")
        elif "parameter_set_by_location" in spec:
            sets, values = _by_location(spec["parameter_set_by_location"], fn, rows, lib, where_label, problems)
        elif "parameters" in spec:
            errors = list(Draft202012Validator(fn["parameters"]).iter_errors(spec["parameters"]))
            for error in errors:
                at = "/".join(str(p) for p in error.absolute_path) or "(parameters)"
                problems.append(f"{where_label}: {at}: {error.message}")
            if not errors:                          # invalid values are reported, not used
                values = [spec["parameters"]] * len(rows)
        elif "parameter_set" in spec:
            template = spec["parameter_set"]
            unresolved: dict[str, list[str]] = defaultdict(list)
            for r, row in enumerate(rows):
                try:
                    set_id = _PLACEHOLDER.sub(lambda m: str(row[m.group(1)]), template)
                except KeyError as exc:
                    problems.append(f"{where_label}: {template!r} needs {per}.{exc.args[0]}, missing on {row['id']}")
                    continue
                found = lib.parameter_sets.get(set_id)
                if found is None or found["function"] != fn["id"]:
                    unresolved[set_id].append(row["id"])
                    continue
                sets[r], values[r] = set_id, found["values"]
            for set_id, ids in unresolved.items():
                problems.append(f"{where_label}: no parameter set {set_id!r} for {fn['id']}, needed by {_few(ids)}")
        else:
            problems.append(f"{where_label}: {fn['id']} needs parameter_set, parameter_set_by_location or parameters")

        quantities = [resolve_quantities(fn, v) for v in values]
        for port in fn["outputs"]:
            for row, q in zip(rows, quantities):
                key = (q[port["name"]], row["id"])
                if key in producer:
                    other = assembly["bindings"][producer[key]]["function"]
                    problems.append(
                        f"{where_label}: {node_id(*key)} is already produced by binding "
                        f"#{producer[key]} ({other}); each node needs exactly one producer"
                    )
                else:
                    producer[key] = index

        bindings.append(Binding(index, spec, fn, rows, sets, values, quantities, {}, set()))

    # Pass 2: parents, now that every producer is known.
    for b in bindings:
        per = b.function["per"]
        for port in b.function["inputs"]:
            over, via = port["over"], port["via"]
            kind = via_kind(via)
            label = f"{assembly['id']} binding #{b.index} ({b.function['id']}) input {port['name']!r}"
            if kind == "all" and lib.entities[over].get("singleton") and len(network.rows(over)) != 1:
                problems.append(f"{label}: needs exactly one {over} row, network {network.id} has {len(network.rows(over))}")
            per_row: list[list[str]] = []
            orphans: dict[str, list[str]] = defaultdict(list)       # quantity -> parent rows nobody produces it for
            for row, q in zip(b.rows, b.quantities):
                quantity = q[port["name"]]
                if kind == "self":
                    pids = [row["id"]]
                elif kind == "fk":
                    pids = [row[via]] if row.get(via) is not None else []
                    if not pids:
                        problems.append(f"{label}: {per} {row['id']!r} has no {via}")
                elif kind == "reverse":
                    attr = via.partition(".")[2]
                    pids = [r["id"] for r in network.rows(over) if r.get(attr) == row["id"]]
                else:
                    pids = [r["id"] for r in network.rows(over)]
                for pid in pids:
                    source = producer.get((quantity, pid))
                    if source is None:
                        orphans[quantity].append(pid)
                    else:
                        b.upstream.add(source)
                per_row.append(pids)
            for quantity, pids in orphans.items():
                unique = list(dict.fromkeys(pids))
                made = sorted({q for (q, pid) in producer if pid in unique[:1]})
                hint = f"; {over} {unique[0]} has {', '.join(made)}" if made else ""
                problems.append(f"{label}: no binding produces {quantity} for {over} {_few(unique)}{hint}")
            b.parents[port["name"]] = per_row

    # Pass 3: a sampling order over bindings.
    by_index = {b.index: b for b in bindings}
    remaining = {i: set(b.upstream) for i, b in by_index.items()}
    order: list[int] = []
    while remaining:
        ready = sorted(i for i, deps in remaining.items() if not deps)
        if not ready:
            cycle = ", ".join(by_index[i].label for i in sorted(remaining))
            problems.append(f"{assembly['id']}: bindings form a cycle among {cycle}")
            break
        for i in ready:
            order.append(i)
            del remaining[i]
        for deps in remaining.values():
            deps.difference_update(ready)

    if problems:
        raise RegistryError(problems)
    indexed = [by_index[i] for i in sorted(by_index)]
    # Binding.index is the position in the assembly; keep the list addressable by it.
    full: list[Binding] = [None] * len(assembly["bindings"])  # type: ignore[list-item]
    for b in indexed:
        full[b.index] = b
    return Compiled(assembly, lib, network, full, producer, order)


def compile_path(path: Path | str, lib: Library | None = None) -> Compiled:
    lib = lib or load_library()
    assembly = load_assembly(path)
    return compile_assembly(assembly, lib, load_network(assembly["network"], lib))
