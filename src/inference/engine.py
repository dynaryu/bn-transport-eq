"""Answer a probabilistic query on a compiled BN.

    query (query.schema.json) --> interventions applied to a copy of the
    assembly and network --> compile --> tbnpy variables and probabilities for
    the bindings the targets depend on --> tbnpy forward sampling (PyTorch) -->
    per-node summaries and event probabilities

Discrete evidence is handled by rejection conditioning, with acceptance diagnostics.
Every result carries ``statement``,
the query written as a probability expression, so the question that was
actually computed can be read back against the one that was asked.
"""

from __future__ import annotations

import copy
import json
import time
import threading
from collections import Counter
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from src.builder.bn_builder import Compiled, build_tbn, compile_assembly, node_id
from src.builder.load import (Library, RegistryError, assembly_id_of, assembly_path, assembly_paths,
                              check_tables, load_assembly, load_library, load_network)

DEFAULT_SAMPLES = 100_000
DEFAULT_SEED = 20261021
BATCH_SIZE = 50_000
MAX_NODES = 500                    # per query; a pattern matching more is a sign to narrow it
_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
        "==": np.equal, "!=": np.not_equal}
_OP_TEXT = {"<": "<", "<=": "≤", ">": ">", ">=": "≥", "==": "=", "!=": "≠"}
_VALIDATOR = Draft202012Validator(json.loads(Path(__file__).with_name("query.schema.json").read_text(encoding="utf-8")))

MIN_ACCEPTED = 30
_SAMPLING_LOCK = threading.RLock()  # torch's CPU RNG is process-global; protect seeded concurrent requests.


class QueryError(ValueError):
    """The query is malformed, outside the engine's scope, or names something the BN lacks."""


@cache
def library() -> Library:
    return load_library()


def node_label(node: str) -> str:
    quantity, _, row = node.partition("::")
    return f"{quantity}[{row}]" if row else node


def statement(query: Mapping[str, Any]) -> str:
    """The query as a probability expression, e.g. P(SystemState[n1>n22:time] = 0 | do(...))."""
    asked = [f"P({node_label(e['node'])} {_OP_TEXT.get(e['op'], e['op'])} {e['value']})"
             for e in query.get("events", [])]
    asked += [f"distribution of {node_label(t)}" for t in query.get("targets", [])]
    asked += [f"P({e['name']}: {expression_text(e['expression'])})" for e in query.get("joint_events", [])]
    do = [f"{row}.{k} = {v}" for row, attrs in query.get("rows", {}).items() for k, v in attrs.items()]
    do += [f"{fn}.{k} = {v}" for fn, values in query.get("parameters", {}).items() for k, v in values.items()]
    do += [f"{fn} with {sid}" for fn, sid in query.get("parameter_sets", {}).items()]
    do += [f"{node_label(n)} = {v}" for n, v in query.get("do", {}).items()]
    given = f" | do({', '.join(do)})" if do else ""
    if query.get("evidence"):
        given += (", observed " if given else " | observed ") + ", ".join(
            f"{node_label(n)} = {v}" for n, v in query["evidence"].items())
    return f"{'; '.join(asked)}{given}   [BN {query.get('assembly')}]"


def _validate(query: Any) -> dict[str, Any]:
    if not isinstance(query, Mapping):
        raise QueryError("a query is a JSON object")
    errors = sorted(_VALIDATOR.iter_errors(query), key=lambda e: list(e.absolute_path))
    if errors:
        raise QueryError("; ".join(f"{'/'.join(map(str, e.absolute_path)) or 'query'}: {e.message}"
                                   for e in errors[:5]))
    return dict(query)


def leaves(expr):
    if "node" in expr:
        return [expr]
    if "not" in expr:
        return leaves(expr["not"])
    return [leaf for child in expr.get("all", expr.get("any", [])) for leaf in leaves(child)]


def expression_text(expr):
    if "node" in expr:
        return f"{node_label(expr['node'])} {_OP_TEXT[expr['op']]} {expr['value']}"
    if "not" in expr:
        return f"not ({expression_text(expr['not'])})"
    key = "all" if "all" in expr else "any"
    return "(" + (" and " if key == "all" else " or ").join(map(expression_text, expr[key])) + ")"


def state_index(quantity, value):
    if quantity["kind"] != "discrete":
        raise QueryError("do/evidence currently require discrete nodes; continuous observations need a likelihood model")
    for i, state in enumerate(quantity["states"]):
        if state == value or str(state) == str(value):
            return i
    raise QueryError(f"{value!r} is not a state of {quantity['id']}; states: {quantity['states']}")


def assignments(query, compiled, key):
    nodes = {n["id"]: n for n in compiled.nodes()}
    out = {}
    for name, value in query.get(key, {}).items():
        if name not in nodes:
            raise QueryError(f"no node {name!r}; do/evidence require exact node names")
        out[name] = state_index(compiled.library.quantities[nodes[name]["quantity"]], value)
    return out


def readback(query):
    """Derived from the query, rather than a second independent LLM interpretation."""
    labels = {"SystemState": "meeting the system requirement", "Usable": "being usable", "EdgeOpen": "being open"}
    def describe_node(name):
        qty, _, row = name.partition("::")
        return f"{row} {labels[qty]}" if qty in labels else node_label(name)
    parts = [f"the probability of {describe_node(e['node'])} {e['op']} {e['value']}" for e in query.get("events", [])]
    parts += [f"the distribution of {describe_node(t)}" for t in query.get("targets", [])]
    joint = query.get("joint_events", [])
    parts += ([f"{len(joint)} joint state probabilities" ] if len(joint) > 4 else
              [f"the joint probability of {e['name']} ({expression_text(e['expression'])})" for e in joint])
    text = "I’ll estimate " + "; ".join(parts) + f", using {query['assembly']}"
    settings = [f"{row}.{attr} = {v}" for row, attrs in query.get("rows", {}).items() for attr, v in attrs.items()]
    settings += [f"{fn}.{attr} = {v}" for fn, attrs in query.get("parameters", {}).items() for attr, v in attrs.items()]
    settings += [f"{fn} uses {sid}" for fn, sid in query.get("parameter_sets", {}).items()]
    if settings:
        text += ". Scenario settings: " + "; ".join(settings)
    if query.get("do"):
        text += ". Force " + ", ".join(f"{node_label(n)} to {v}" for n, v in query["do"].items())
    if query.get("evidence"):
        text += ". Condition on the observation " + ", ".join(f"{node_label(n)} = {v}" for n, v in query["evidence"].items())
    return text + "."


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


def sample_nodes(compiled: Compiled, n: int, seed: int | None = None, bindings: set[int] | None = None,
                 notes: list[str] | None = None, nodes: set[str] | None = None,
                 rules: dict[str, Any] | None = None,
                 on_missing_rules: Callable[[float, list[str]], None] | None = None,
                 do: Mapping[str, int] | None = None) -> dict[str, np.ndarray]:
    """{node id: (n,) samples} by tbnpy. Discrete nodes hold the index into their quantity's states.

    ``bindings`` restricts the model to those bindings, which must be closed under
    ``upstream`` (``Compiled.ancestors`` gives such a set): a query about an edge
    need not pay for travel times. ``notes``, if given, collects what the
    probability objects report, such as which ShakeMap version supplied the shaking.
    ``nodes``, if given, samples only the objects owning them and their ancestors.
    ``rules``, if given, receives how the system states' stored rsr rules served (``rules_report``).
    ``on_missing_rules(threshold, origins)``, if given, is called before sampling with the origins that
    have no rules of their own at the system event's threshold, so their extraction can start at once.
    """
    import torch
    from tbnpy import inference

    if n < 1:
        raise ValueError("n must be at least 1")
    _, probs = build_tbn(compiled, bindings)
    if do:
        from .interventions import apply
        apply(probs, do)
    keys = [k for k, p in probs.items() if nodes is None or any(v.name in nodes for v in p.childs)]
    if on_missing_rules is not None:
        missing, threshold = missing_rules([probs[k] for k in keys if probs[k].spec["role"] == "system"])
        if missing:
            on_missing_rules(threshold, missing)
    with _SAMPLING_LOCK, torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        sampled = inference.sample(probs, keys, n_sample=n, batch_size=BATCH_SIZE)
    if notes is not None:
        for prob in probs.values():
            notes.extend(w for w in prob.notes if w not in notes)
    systems = [probs[k] for k in keys if probs[k].spec["role"] == "system"]
    if systems:
        report = rules_report(systems)
        if rules is not None:
            rules.update(report)
        if notes is not None:
            notes.append(report["note"])
    return {v.name: sampled[key].Cs[:, j].cpu().numpy()
            for key, prob in sampled.items() for j, v in enumerate(prob.childs)}


def object_scope(specs: Mapping[str, Mapping[str, Any]], keys: list[str]) -> list[str]:
    """The probability objects sampling ``keys`` needs: those, and every object owning one of their
    parents, transitively, in model order. Nothing else is statistically relevant to a forward query."""
    owner = {child: key for key, spec in specs.items() for child in spec["childs"]}
    seen, stack = set(), list(keys)
    while stack:
        key = stack.pop()
        if key not in seen:
            seen.add(key)
            stack += [owner[p] for p in specs[key]["parents"] if p in owner]
    return [key for key in specs if key in seen]


def node_scope(compiled: Compiled, needed: set[str]) -> set[str]:
    """The query nodes and all their ancestors: the variables a forward query's answer depends on.
    (With evidence, the evidence nodes and their ancestors join it, which brings in any co-parents.)"""
    parents = {n["id"]: n["parents"] for n in compiled.nodes()}
    seen, stack = set(), list(needed)
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack += parents[node]
    return seen


def _needed(query: Mapping[str, Any], compiled: Compiled):
    """The nodes a query names: its targets, its events (each with its matches), and their union."""
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
    for event in query.get("joint_events", []):
        for leaf in leaves(event["expression"]):
            if leaf["node"] not in nodes:
                raise QueryError(f"joint events require exact node names: {leaf['node']!r}")
            needed.add(leaf["node"])
    needed.update(assignments(query, compiled, "evidence"))
    needed.update(assignments(query, compiled, "do"))
    if len(needed) > MAX_NODES:
        raise QueryError(f"the query touches {len(needed)} nodes; narrow the patterns to {MAX_NODES} or fewer")
    return nodes, targets, events, needed


def _computed(compiled: Compiled, bindings: set[int], needed: set[str]) -> int:
    """How many variables sampling a query's scope computes: its objects' every child (a joint field whole)."""
    from src.builder.bn_builder import tbn_data

    specs = tbn_data(compiled, bindings)["probabilities"]
    keys = [k for k, s in specs.items() if any(c in needed for c in s["childs"])]
    return sum(len(specs[k]["childs"]) for k in object_scope(specs, keys))


def scope(query: Mapping[str, Any]) -> dict[str, Any]:
    """Which part of the BN a query depends on, and what sampling it computes, without sampling anything."""
    from src.builder.bn_builder import tbn_data

    query = _validate(query)
    compiled = compile_query(query)
    nodes, _, _, needed = _needed(query, compiled)
    relevant = node_scope(compiled, needed)
    bindings = compiled.ancestors({compiled.producer[(nodes[i]["quantity"], nodes[i]["row"])] for i in needed})
    specs = tbn_data(compiled, bindings)["probabilities"]
    keys = [k for k, s in specs.items() if any(c in needed for c in s["childs"])]
    objects = object_scope(specs, keys)
    sampled = [c for k in objects for c in specs[k]["childs"]]

    def by_quantity(ids):
        return dict(Counter(nodes[i]["quantity"] for i in sorted(ids, key=list(nodes).index)))

    return {
        "assembly": query["assembly"], "statement": statement(query),
        "query_nodes": sorted(needed, key=list(nodes).index),
        "relevant": {"count": len(relevant), "of": len(nodes), "by_quantity": by_quantity(relevant)},
        "pruned": sorted(set(nodes) - relevant, key=list(nodes).index),
        "sampled": {"variables": len(sampled), "objects": len(objects), "by_quantity": by_quantity(sampled),
                    "beyond_relevant": sorted(set(sampled) - relevant, key=list(nodes).index)},
        "joint": sorted({specs[k]["function"] for k in objects if len(specs[k]["childs"]) > 1}),
    }


def missing_rules(systems: list[Any]) -> tuple[list[str], float | None]:
    """The origins without rules of their own at the threshold (known before any sampling), and the threshold."""
    from src.builder.bn_builder import rules_source, rules_threshold

    missing, threshold = [], None
    for prob in systems:
        threshold = rules_threshold(prob.spec)
        references = prob.spec.get("references") or [None] * len(prob.spec["rows"])
        for j, row in enumerate(prob.spec["rows"]):
            if rules_source(references[j] if prob.refs[j] is not None else None, threshold) != "exact":
                missing.append(row["id"])
    return missing, threshold


def rules_report(systems: list[Any]) -> dict[str, Any]:
    """Per origin: where its rules came from and the share of samples they left to ndtools; and which
    origins have no rules of their own at this threshold. A threshold is part of the system function, so
    each one gets its own rules: until they exist, rules that still hold from other thresholds classify
    what they can and ndtools decides the rest, so the answer is the same, only slower."""
    from src.builder.bn_builder import rules_source, rules_threshold

    origins, threshold = [], None
    for prob in systems:
        threshold = rules_threshold(prob.spec)
        references = prob.spec.get("references") or [None] * len(prob.spec["rows"])
        for j, row in enumerate(prob.spec["rows"]):
            ref = references[j] if prob.refs[j] is not None else None
            seen, left = prob.tally["seen"][j], prob.tally["unclassified"][j]
            origins.append({"origin": row["id"], "rules": rules_source(ref, threshold),
                            "unclassified": left / seen if seen else 0.0})
    extract = [o["origin"] for o in origins if o["rules"] != "exact"]
    exact = sum(o["rules"] == "exact" for o in origins)
    borrowed = sum(o["rules"] not in ("exact", "no stored rules") for o in origins)
    share = sum(o["unclassified"] for o in origins) / len(origins)
    note = (f"System states at target_add_h={threshold:g}: rsr rules exact for {exact} of {len(origins)} origin(s)"
            + (f", borrowed from other thresholds for {borrowed}" if borrowed else "")
            + (f", none for {len(origins) - exact - borrowed}" if len(origins) - exact - borrowed else "")
            + f"; on average {share:.1%} of samples left to ndtools.")
    if extract:
        note += (f" No rules of their own at this threshold for {', '.join(extract[:6])}"
                 f"{' ...' if len(extract) > 6 else ''}: they need extracting.")
    return {"target_add_h": threshold, "origins": origins, "extract": extract, "note": note}


def summarise(quantity: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    """State probabilities for a discrete quantity, moments for a continuous one."""
    if quantity["kind"] == "discrete":
        counts = np.bincount(np.asarray(x, int), minlength=len(quantity["states"])) / len(x)
        return {"P": dict(zip(map(str, quantity["states"]), counts.round(6).tolist()))}
    finite = x[np.isfinite(x)]
    stats: dict[str, Any] = {"P_inf": float(np.mean(~np.isfinite(x)))}
    if finite.size:
        p5, p50, p95 = np.percentile(finite, [5, 50, 95])
        stats.update(mean_finite=float(finite.mean()), p5=float(p5), p50=float(p50), p95=float(p95))
    return stats


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


def joint_mask(expr, samples, compiled):
    if "node" in expr:
        qty = compiled.library.quantities[expr["node"].split("::")[0]]
        return _event_mask(qty, samples[expr["node"]], expr["op"], expr["value"])
    if "not" in expr:
        return ~joint_mask(expr["not"], samples, compiled)
    key = "all" if "all" in expr else "any"
    masks = [joint_mask(child, samples, compiled) for child in expr[key]]
    return (np.logical_and if key == "all" else np.logical_or).reduce(masks)


def answer(query: Mapping[str, Any],
           on_missing_rules: Callable[[float, list[str]], None] | None = None,
           record: Mapping[str, Any] | None = None, samples_policy: str | None = None) -> dict[str, Any]:
    """The query's answer. With ``record`` (who asked: ``{"source": "page"}`` and the like), the query,
    its numbers and their provenance are appended to records/queries.jsonl (``records.log_query``).

    ``samples_policy`` says where the samples come from (default: "bank" when recorded, else "scope"):

    * "bank": the scenario's sample bank, every variable of the whole BN, read when it exists, sampled
      and stored when it fits under records.SAMPLE_LIMIT_BYTES; a follow-up question on the same
      scenario reads the same draws. When the whole BN would not fit, as "scope".
    * "scope": sample only the query's scope, its nodes and their ancestors, on the fly; keep nothing.
    * "whole": sample the whole BN as a bank would, without reading or writing one (a replay's check).
    """
    started = time.perf_counter()
    query = _validate(query)
    query.setdefault("samples", DEFAULT_SAMPLES)
    query.setdefault("seed", DEFAULT_SEED)
    compiled = compile_query(query)
    lib = compiled.library
    nodes, targets, events, needed = _needed(query, compiled)
    do = assignments(query, compiled, "do")
    evidence = assignments(query, compiled, "evidence")
    if any(n in do and do[n] != v for n, v in evidence.items()):
        raise QueryError("Evidence contradicts a forced node state")

    bindings = compiled.ancestors({compiled.producer[(nodes[i]["quantity"], nodes[i]["row"])] for i in needed})
    kernel_warnings: list[str] = []
    rules: dict[str, Any] = {}
    from . import records

    policy = samples_policy or ("bank" if record is not None else "scope")
    if policy not in ("bank", "scope", "whole"):
        raise ValueError(f"samples_policy must be bank, scope or whole, not {policy!r}")
    samples, bank = None, {"path": None, "reused": False, "bytes": None, "reason": None}
    try:
        if policy in ("bank", "whole"):
            from src.builder.bn_builder import _tbn_fingerprint, tbn_data

            whole = tbn_data(compiled)
            if do:
                whole = {**whole, "node_interventions": do}
            kinds = {name: "discrete" if "states" in v else "continuous" for name, v in whole["variables"].items()}
            bank["bytes"] = query["samples"] * sum(1 if k == "discrete" else 4 for k in kinds.values())
            path = records.bank_path(_tbn_fingerprint(whole), query["samples"], query["seed"])
            if policy == "bank" and path.is_file():
                samples, meta = records.load_bank(path)
                kernel_warnings += meta["notes"]
                rules.update(meta["rules"] or {})
                bank.update(path=path, reused=True)
            elif bank["bytes"] <= records.SAMPLE_LIMIT_BYTES:
                samples = records.as_stored(sample_nodes(compiled, query["samples"], query["seed"], None,
                                                         kernel_warnings, None, rules, on_missing_rules, do), kinds)
                if policy == "bank":
                    records.save_bank(path, samples, kinds, {"notes": kernel_warnings,
                                                             "rules": {k: v for k, v in rules.items() if k != "note"}})
                    bank["path"] = path
            else:
                bank["reason"] = (f"the whole BN's samples would be {bank['bytes'] / 2**20:.0f} MB, over the "
                                  f"{records.SAMPLE_LIMIT_BYTES / 2**20:.0f} MB limit: sampled the query's scope only")
        if samples is None:
            samples = sample_nodes(compiled, query["samples"], query["seed"], bindings, kernel_warnings, needed, rules,
                                   on_missing_rules, do)
    except ValueError as exc:            # a kernel refusing its inputs, e.g. a site outside a ShakeMap
        raise QueryError(f"the scenario could not be computed: {exc}") from exc

    accepted = query["samples"]
    if evidence:
        keep = np.logical_and.reduce([samples[n] == v for n, v in evidence.items()])
        accepted = int(keep.sum())
        if accepted < MIN_ACCEPTED:
            raise QueryError(f"Evidence matched only {accepted} of {query['samples']} draws; need at least "
                             f"{MIN_ACCEPTED}. Increase samples or use a more efficient evidence method. "
                             "Zero matches do not prove the observation is impossible.")
        samples = {k: x[keep] for k, x in samples.items()}
        kernel_warnings.append(f"Discrete evidence: rejection conditioning retained {accepted}/{query['samples']} draws.")
    out_targets = [{"node": i, "quantity": nodes[i]["quantity"], "row": nodes[i]["row"],
                    "summary": summarise(lib.quantities[nodes[i]["quantity"]], samples[i])} for i in targets]
    out_events = []
    for event, matched in events:
        for i in matched:
            mask = _event_mask(lib.quantities[nodes[i]["quantity"]], samples[i], event["op"], event["value"])
            hits = int(mask.sum())
            out_events.append({"node": i, "op": event["op"], "value": event["value"],
                               "p": hits / accepted, "ci95": _wilson(hits, accepted)})
    for event in query.get("joint_events", []):
        hits = int(joint_mask(event["expression"], samples, compiled).sum())
        out_events.append({"node": event["name"], "op": "joint", "value": True,
                           "expression": event["expression"], "p": hits / accepted, "ci95": _wilson(hits, accepted)})

    computed = [compiled.bindings[i] for i in compiled.order if i in bindings]
    warnings = kernel_warnings
    result = {
        "assembly": query["assembly"],
        "label": compiled.assembly["label"],
        "statement": statement(query),
        "query": query,
        "samples": query["samples"],
        "inference": {"method": "rejection" if evidence else "forward", "drawn": query["samples"],
                      "accepted": accepted, "effective_samples": accepted, "acceptance_rate": accepted / query["samples"]},
        "seed": query["seed"],
        "targets": out_targets,
        "events": out_events,
        "computed": [b.function["id"] for b in computed],
        "warnings": warnings,
        "rules": {k: v for k, v in rules.items() if k != "note"} or None,
        # relevant: what the answer depends on; computed: the variables sampled for it now (none from a bank).
        "scope": {"relevant": len(node_scope(compiled, needed)), "of": len(nodes),
                  "computed": 0 if bank["reused"] else len(samples) if bank["path"] or policy == "whole" and not bank["reason"]
                  else _computed(compiled, bindings, needed)},
        "samples_from": ("the scenario's stored samples" if bank["reused"] else
                         "the whole BN, sampled now and stored" if bank["path"] else
                         "the whole BN, sampled now" if policy == "whole" and not bank["reason"] else
                         "the query's scope, sampled now"),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    if record is not None:
        result["record"] = records.log_query(result, compiled, bindings, bank, record)
    return result


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
        # A singleton table a function reads (the Event) is the scenario: its numbers can be set.
        for table in fn.get("tables", []):
            entity = lib.entities[table]
            if not entity.get("singleton") or any(r["entity"] == table for r in rows):
                continue
            for row in network.rows(table):
                for attr, spec in entity["attributes"].items():
                    if spec.get("type") == "number":
                        rows.append({"row": row["id"], "entity": table, "attribute": attr,
                                     "value": row.get(attr), "unit": spec.get("unit"), "type": spec["type"],
                                     "minimum": spec.get("minimum"), "step": spec.get("step")})
        if "parameters" in b.spec:
            props = fn["instance_schema"].get("properties", {})
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
