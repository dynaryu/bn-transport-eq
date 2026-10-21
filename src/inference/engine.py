"""Answer a probabilistic query on a compiled BN.

    query (query.schema.json) --> interventions applied to a copy of the
    assembly and network --> compile --> forward Monte Carlo over the bindings
    the targets depend on --> per-node summaries and event probabilities

The engine is forward-only: a query with evidence is refused, here, whatever
asked for it, the page or a language model. Every result carries ``statement``,
the query written as a probability expression, so the question that was
actually computed can be read back against the one that was asked.
"""

from __future__ import annotations

import copy
import json
import time
from collections import Counter
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from src.builder.compile import Compiled, compile_assembly, node_id
from src.builder.load import (Library, RegistryError, assembly_id_of, assembly_path, assembly_paths,
                              check_tables, load_assembly, load_library, load_network)

from .sample import forward_sample, summarise

DEFAULT_SAMPLES = 2000
DEFAULT_SEED = 20261021
MAX_NODES = 500                    # per query; a pattern matching more is a sign to narrow it
_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
        "==": np.equal, "!=": np.not_equal}
_OP_TEXT = {"<": "<", "<=": "≤", ">": ">", ">=": "≥", "==": "=", "!=": "≠"}
_VALIDATOR = Draft202012Validator(json.loads(Path(__file__).with_name("query.schema.json").read_text(encoding="utf-8")))

FORWARD_ONLY = (
    "Evidence is not supported: this engine does forward inference only (scenario in, "
    "distribution out). A question conditioned on an observation, such as which bridge most "
    "likely failed given that a route is closed, needs backward inference, which is not implemented."
)


class QueryError(ValueError):
    """The query is malformed, outside the engine's scope, or names something the BN lacks."""


@cache
def library() -> Library:
    return load_library()


def node_label(node: str) -> str:
    quantity, _, row = node.partition("::")
    return f"{quantity}[{row}]" if row else node


def statement(query: Mapping[str, Any]) -> str:
    """The query as a probability expression, e.g. P(TravelTime[n1>n22] ≤ 60 | do(...))."""
    asked = [f"P({node_label(e['node'])} {_OP_TEXT.get(e['op'], e['op'])} {e['value']})"
             for e in query.get("events", [])]
    asked += [f"distribution of {node_label(t)}" for t in query.get("targets", [])]
    do = [f"{row}.{k} = {v}" for row, attrs in query.get("rows", {}).items() for k, v in attrs.items()]
    do += [f"{fn}.{k} = {v}" for fn, values in query.get("parameters", {}).items() for k, v in values.items()]
    do += [f"{fn} with {sid}" for fn, sid in query.get("parameter_sets", {}).items()]
    given = f" | do({', '.join(do)})" if do else ""
    return f"{'; '.join(asked)}{given}   [BN {query.get('assembly')}]"


def _validate(query: Any) -> dict[str, Any]:
    if not isinstance(query, Mapping):
        raise QueryError("a query is a JSON object")
    errors = sorted(_VALIDATOR.iter_errors(query), key=lambda e: list(e.absolute_path))
    if errors:
        raise QueryError("; ".join(f"{'/'.join(map(str, e.absolute_path)) or 'query'}: {e.message}"
                                   for e in errors[:5]))
    if query.get("evidence"):
        raise QueryError(FORWARD_ONLY)
    return dict(query)


def compile_query(query: Mapping[str, Any]) -> Compiled:
    """The assembly and network with the query's interventions applied, compiled."""
    lib = library()
    try:
        assembly = copy.deepcopy(load_assembly(assembly_path(query["assembly"])))
        network = copy.deepcopy(load_network(assembly["network"], lib))
    except RegistryError as exc:
        raise QueryError("; ".join(exc.problems)) from exc

    for row_id, attrs in query.get("rows", {}).items():
        entity = network.entity_of.get(row_id)
        if entity is None:
            raise QueryError(f"no row {row_id!r} in network {network.id}")
        defined = lib.entities[entity]["attributes"]
        for attr in attrs:
            if attr not in defined or "ref" in defined[attr]:
                settable = sorted(a for a, s in defined.items() if "ref" not in s)
                raise QueryError(f"{entity} has no settable attribute {attr!r}; settable: {', '.join(settable)}")
        network.row(row_id).update(attrs)
    if query.get("rows"):
        _, problems = check_tables(network.tables, lib, "rows")
        if problems:
            raise QueryError("; ".join(problems))

    bound = sorted({b["function"] for b in assembly["bindings"]})
    for fn_id, values in query.get("parameters", {}).items():
        targets = [b for b in assembly["bindings"] if b["function"] == fn_id]
        if not targets:
            raise QueryError(f"{fn_id} is not bound in {assembly['id']}; bound: {', '.join(bound)}")
        for b in targets:
            if "parameters" not in b:
                raise QueryError(f"{fn_id} takes its values from parameter set {b.get('parameter_set')!r}; "
                                 "choose another set with parameter_sets instead")
            b["parameters"] = {**b["parameters"], **values}
    for fn_id, set_id in query.get("parameter_sets", {}).items():
        targets = [b for b in assembly["bindings"] if b["function"] == fn_id]
        if not targets:
            raise QueryError(f"{fn_id} is not bound in {assembly['id']}; bound: {', '.join(bound)}")
        for b in targets:
            b.pop("parameters", None)
            b.pop("parameter_set_by_location", None)
            b["parameter_set"] = set_id

    try:
        return compile_assembly(assembly, lib, network)
    except RegistryError as exc:
        raise QueryError("; ".join(exc.problems)) from exc


def _match(pattern: str, ids: list[str], examples: list[str]) -> list[str]:
    found = [i for i in ids if fnmatchcase(i, pattern)]
    if not found:
        raise QueryError(f"no node matches {pattern!r}; node ids look like {', '.join(examples)}")
    return found


def _wilson(successes: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def _event_mask(quantity: dict[str, Any], x: np.ndarray, op: str, value: Any) -> np.ndarray:
    if quantity["kind"] == "discrete":
        states = quantity["states"]
        index = next((i for i, s in enumerate(states) if s == value or str(s) == str(value)), None)
        if index is None:
            raise QueryError(f"{value!r} is not a state of {quantity['id']}; states: {states}")
        return _OPS[op](np.asarray(x, int), index)
    if quantity["kind"] == "continuous":
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            raise QueryError(f"{quantity['id']} is continuous; compare it with a number, not {value!r}") from None
        return _OPS[op](x, threshold)
    raise QueryError(f"{quantity['id']} is a record; ask for a target summary instead of an event")


def answer(query: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    query = _validate(query)
    query.setdefault("samples", DEFAULT_SAMPLES)
    query.setdefault("seed", DEFAULT_SEED)
    compiled = compile_query(query)
    lib = compiled.library
    nodes = {n["id"]: n for n in compiled.nodes()}
    ids = list(nodes)
    first: dict[str, str] = {}
    for n in nodes.values():
        first.setdefault(n["quantity"], n["id"])
    examples = list(first.values())

    targets: list[str] = []
    for pattern in query.get("targets", []):
        targets += [i for i in _match(pattern, ids, examples) if i not in targets]
    events = [(e, _match(e["node"], ids, examples)) for e in query.get("events", [])]
    needed = set(targets) | {i for _, matched in events for i in matched}
    if len(needed) > MAX_NODES:
        raise QueryError(f"the query touches {len(needed)} nodes; narrow the patterns to {MAX_NODES} or fewer")

    bindings = compiled.ancestors({compiled.producer[(nodes[i]["quantity"], nodes[i]["row"])] for i in needed})
    kernel_warnings: list[str] = []
    try:
        samples = forward_sample(compiled, query["samples"], query["seed"], bindings, kernel_warnings)
    except ValueError as exc:            # a kernel refusing its inputs, e.g. a site outside a ShakeMap
        raise QueryError(f"the scenario could not be computed: {exc}") from exc

    out_targets = [{"node": i, "quantity": nodes[i]["quantity"], "row": nodes[i]["row"],
                    "summary": summarise(lib.quantities[nodes[i]["quantity"]], samples[i])} for i in targets]
    out_events = []
    for event, matched in events:
        for i in matched:
            mask = _event_mask(lib.quantities[nodes[i]["quantity"]], samples[i], event["op"], event["value"])
            hits = int(mask.sum())
            out_events.append({"node": i, "op": event["op"], "value": event["value"],
                               "p": hits / query["samples"], "ci95": _wilson(hits, query["samples"])})

    computed = [compiled.bindings[i] for i in compiled.order if i in bindings]
    warnings = [f"{b.function['id']} draws each {b.function['per']} independently: no spatial correlation, "
                "so joint outcomes across the network are likely understated."
                for b in computed if b.function["drawKind"] == "Marginal" and not b.function["deterministic"]
                and b.function["role"] == "hazard"] + kernel_warnings
    return {
        "assembly": query["assembly"],
        "label": compiled.assembly["label"],
        "statement": statement(query),
        "query": query,
        "samples": query["samples"],
        "seed": query["seed"],
        "targets": out_targets,
        "events": out_events,
        "computed": [b.function["id"] for b in computed],
        "warnings": warnings,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def describe(assembly_id: str) -> dict[str, Any]:
    """What a query on this BN can ask for and set: the menu the page and the LLM read."""
    compiled = compile_query({"assembly": assembly_id})
    lib, network = compiled.library, compiled.network

    quantities = []
    for i in compiled.order:
        b = compiled.bindings[i]
        for port in b.function["outputs"]:
            rows_by_quantity: dict[str, list[str]] = {}
            for row, resolved in zip(b.rows, b.quantities):
                rows_by_quantity.setdefault(resolved[port["name"]], []).append(row["id"])
            for qid, ids in rows_by_quantity.items():
                q = lib.quantities[qid]
                quantities.append({
                    "id": q["id"], "label": q["label"], "kind": q["kind"], "entity": b.function["per"],
                    "unit": q.get("unit"), "states": q.get("states"), "fields": list(q.get("fields", {})) or None,
                    "nodes": len(ids), "example": node_id(q["id"], ids[0]), "function": b.function["id"],
                })

    rows, parameters, choices = [], [], []
    for i in compiled.order:
        b = compiled.bindings[i]
        fn = b.function
        if lib.entities[fn["per"]].get("singleton"):
            for row in b.rows:
                for attr in fn.get("attributes", []):
                    spec = lib.entities[fn["per"]]["attributes"][attr]
                    rows.append({"row": row["id"], "entity": fn["per"], "attribute": attr,
                                 "value": row.get(attr), "unit": spec.get("unit"), "type": spec.get("type")})
        if "parameters" in b.spec:
            props = fn["parameters"].get("properties", {})
            for name, value in b.spec["parameters"].items():
                parameters.append({"function": fn["id"], "name": name, "value": value, "schema": props.get(name, {})})
        elif "parameter_set" in b.spec and "{" not in b.spec["parameter_set"]:
            sets = [{"id": sid, "label": s["label"]} for sid, s in lib.parameter_sets.items() if s["function"] == fn["id"]]
            if len(sets) > 1:
                choices.append({"function": fn["id"], "current": b.spec["parameter_set"], "choices": sets})
        elif "parameter_set_by_location" in b.spec:
            # Chosen per row by region; a query may still force one set everywhere.
            sets = [{"id": sid, "label": lib.parameter_sets[sid]["label"]} for sid in b.spec["parameter_set_by_location"]]
            choices.append({"function": fn["id"], "current": None, "by_location": dict(Counter(b.parameter_sets)),
                            "choices": sets})

    names = {r["id"]: r["name"] for r in network.rows("Node") if r.get("name")}
    siblings = [{"id": assembly_id_of(p), "label": load_assembly(p)["label"]}
                for p in assembly_paths() if p.parent.name == network.id]
    return {
        "assembly": assembly_id,
        "label": compiled.assembly["label"],
        "description": compiled.assembly.get("description", ""),
        "network": {"id": network.id, "title": network.title,
                    "rows": {k: len(v) for k, v in network.tables.items()}},
        "alternatives": siblings,
        "quantities": quantities,
        "settable": {"rows": rows, "parameters": parameters, "parameter_sets": choices},
        "names": names,
        "n_nodes": compiled.n_nodes,
    }
