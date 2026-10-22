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
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .load import (Library, Network, RegistryError, assembly_path, load_assembly, load_library, load_network,
                   resolve_quantities, via_kind)

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
        for table in fn.get("tables", []):
            if lib.entities[table].get("singleton") and len(network.rows(table)) != 1:
                problems.append(f"{where_label}: {fn['id']} reads the {table} row, network {network.id} has "
                                f"{len(network.rows(table))}")

        sets: list[str | None] = [None] * len(rows)
        values: list[dict[str, Any] | None] = [None] * len(rows)
        if "instance_schema" not in fn:
            if any(k in spec for k in ("parameter_set", "parameters", "parameter_set_by_location")):
                problems.append(f"{where_label}: {fn['id']} takes no parameters")
        elif "parameter_set_by_location" in spec:
            sets, values = _by_location(spec["parameter_set_by_location"], fn, rows, lib, where_label, problems)
        elif "parameters" in spec:
            errors = list(Draft202012Validator(fn["instance_schema"]).iter_errors(spec["parameters"]))
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


def tbn_data(compiled: Compiled, bindings: set[int] | None = None) -> dict[str, Any]:
    """Scalar tbnpy variables and one probability specification per binding (or output).

    Every row and output of a binding are the childs of one object, so a hazard
    binding's whole field is sampled together. A deterministic binding with several
    outputs becomes one object per output: nothing is lost, and a question about one
    output (a system state) need not compute the other (a travel time). ``bindings``
    restricts the data to those bindings, which must be closed under ``upstream``.
    The JSON is a portable snapshot of the selected inputs, not sampled data.
    """
    variables, probabilities = {}, {}
    lib = compiled.library

    for index in compiled.order:
        if bindings is not None and index not in bindings:
            continue
        b = compiled.bindings[index]
        fn = b.function

        # One shared parent list, with explicit indices for each row and port.
        parents, input_columns = [], []
        for r, row in enumerate(b.rows):
            ports = {}
            for port in fn["inputs"]:
                quantity = b.quantities[r][port["name"]]
                columns = []
                for pid in b.parents[port["name"]][r]:
                    name = node_id(quantity, pid)
                    if name not in parents:
                        parents.append(name)
                    columns.append(parents.index(name))
                ports[port["name"]] = columns
            input_columns.append(ports)

        # The states of each discrete port, which must agree across the binding's rows.
        states = {}
        for port in fn["inputs"] + fn["outputs"]:
            used = {b.quantities[r][port["name"]] for r in range(len(b.rows))}
            if {lib.quantities[q]["kind"] for q in used} == {"discrete"}:
                if len({tuple(lib.quantities[q]["states"]) for q in used}) != 1:
                    raise ValueError(f"{fn['id']}: {port['name']} has different states across rows")
                states[port["name"]] = lib.quantities[next(iter(used))]["states"]

        split = fn["deterministic"] and len(fn["outputs"]) > 1
        for ports in ([[p] for p in fn["outputs"]] if split else [fn["outputs"]]):
            childs = []
            for r, row in enumerate(b.rows):
                for port in ports:
                    q = lib.quantities[b.quantities[r][port["name"]]]
                    name = node_id(q["id"], row["id"])
                    childs.append(name)
                    variables[name] = ({"states": q["states"]} if q["kind"] == "discrete"
                                       else {"range": q.get("range", [None, None])})
            key = f"{fn['id']}:{ports[0]['name']}" if split else fn["id"]
            if key in probabilities:
                key = f"{key}#{index}"
            spec = {
                "function": fn["id"], "implementation": fn["implementation"], "binding": index,
                "childs": childs, "parents": parents, "input_columns": input_columns,
                "parent_rows": b.parents, "rows": b.rows, "parameters": b.parameters,
                "parameter_sets": b.parameter_sets, "states": states,
                "tables": {t: compiled.network.rows(t) for t in fn.get("tables", [])},
            }
            if split:
                spec["output"] = ports[0]["name"]
            probabilities[key] = spec

    return {
        "format": "quakenet.tbnpy.v2", "assembly": compiled.assembly,
        "network": compiled.network.id,
        "network_assumptions": compiled.network.assumptions,
        "instance_sources": {
            s: {"file": lib.parameter_sets[s]["file"], "source": lib.parameter_sets[s]["source"]}
            for i, b in enumerate(compiled.bindings) if bindings is None or i in bindings
            for s in b.parameter_sets if s
        },
        "variables": variables, "probabilities": probabilities,
    }


def define_tbn_variables(data: dict[str, Any]) -> dict[str, Any]:
    """Construct tbnpy Variable objects from generated model data."""
    from tbnpy.variable import Variable

    return {
        name: Variable(name, spec["states"] if "states" in spec else tuple(
            (-float("inf") if i == 0 else float("inf")) if v is None else v
            for i, v in enumerate(spec["range"])))
        for name, spec in data["variables"].items()
    }


def define_tbn_probabilities(data: dict[str, Any], variables: dict[str, Any], device: str = "cpu",
                             directory: Path | str | None = None) -> dict[str, Any]:
    """Construct each probability object from its spec, with the class its function names."""
    from .load import resolve_implementation

    if "source_sha256" in data and _tbn_fingerprint(data) != data["source_sha256"]:
        raise ValueError("Data belong to a different generated model; regenerate")
    if directory:
        for spec in data["probabilities"].values():
            for j, ref in enumerate(spec.get("references", [])):
                stored = _rules_meta(Path(directory) / ref["upper"]).get("rules_sha256") if ref else None
                if ref and stored != rules_fingerprint(spec, j):
                    raise ValueError("RSR references belong to a different system definition; regenerate")
    return {
        key: resolve_implementation(spec["implementation"])(
            [variables[n] for n in spec["childs"]], [variables[n] for n in spec["parents"]],
            spec, device=device, directory=directory)
        for key, spec in data["probabilities"].items()
    }


def _tbn_fingerprint(data: dict[str, Any]) -> str:
    snapshot = {k: v for k, v in data.items() if k not in ("source_sha256", "versions")}
    snapshot["probabilities"] = {
        k: {name: value for name, value in spec.items() if name != "references"}
        for k, spec in data["probabilities"].items()
    }
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def rules_fingerprint(spec: dict[str, Any], j: int) -> str:
    """What origin j's system rules depend on: the graph, the destinations, the origin, its
    threshold and the edge order. Not the hazard: rules extracted once serve every scenario."""
    snapshot = {"function": spec["function"], "tables": spec["tables"], "origin": spec["rows"][j]["id"],
                "parameters": spec["parameters"][j], "edges": spec["parent_rows"]["open"][j]}
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def _rules_meta(path: Path) -> dict[str, Any]:
    meta = path.with_name("model.json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}


def model_dir(assembly_file: Path | str) -> Path:
    """Where an assembly's generated tbnpy model lives: examples/<n>/tbnpy/<assembly name>/."""
    assembly_file = Path(assembly_file)
    return assembly_file.parent / "tbnpy" / assembly_file.name.removesuffix(".assembly.json")


def attach_rules(data: dict[str, Any], directory: Path) -> int:
    """Give each system-state origin the stored rules extracted for exactly its definition. Returns how many."""
    found = 0
    for spec in data["probabilities"].values():
        if spec.get("output") != "state":
            continue
        spec["references"] = []
        for j, row in enumerate(spec["rows"]):
            upper = directory / "rsr" / row["id"] / "refs_up_1.pt"
            ok = upper.is_file() and _rules_meta(upper).get("rules_sha256") == rules_fingerprint(spec, j)
            spec["references"].append({"upper": f"rsr/{row['id']}/refs_up_1.pt",
                                       "lower": f"rsr/{row['id']}/refs_low_0.pt"} if ok else None)
            found += ok
    return found


def build_tbn(compiled: Compiled, bindings: set[int] | None = None, device: str = "cpu",
              rules: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """tbnpy's two dictionaries, variables and probabilities, for a compiled assembly.

    With ``rules``, system states reuse the rsr rules stored with the assembly's
    generated model wherever they were extracted for the same system definition;
    any other origin is evaluated by ndtools throughout.
    """
    data = tbn_data(compiled, bindings)
    directory = None
    if rules:
        try:
            candidate = model_dir(assembly_path(compiled.assembly["id"]))
        except RegistryError:
            candidate = None
        if candidate is not None and attach_rules(data, candidate):
            directory = candidate
    variables = define_tbn_variables(data)
    return variables, define_tbn_probabilities(data, variables, device=device, directory=directory)


def generate_tbn(path: Path | str, *, extract_rules: bool = True, seed: int = 20261022) -> Path:
    """Write a runnable tbnpy model beneath the assembly's example folder."""
    import importlib.metadata
    import random
    import torch

    path = Path(path).resolve()
    compiled = compile_path(path)
    data = tbn_data(compiled)
    variables = define_tbn_variables(data)
    probabilities = define_tbn_probabilities(data, variables)
    fingerprint = _tbn_fingerprint(data)
    out = model_dir(path)
    out.mkdir(parents=True, exist_ok=True)
    data["source_sha256"] = fingerprint
    data["versions"] = {n: importlib.metadata.version(n) for n in ("tbnpy", "torch", "ndtools-duco", "rsr-duco")}

    # Extraction's independent 50/50 proposal explores topology; it does not
    # replace the BN's component distribution or its hazard-induced dependence.
    if extract_rules:
        from rsr.rsr import run_ref_extraction_by_mcs

        python_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                random.seed(seed)
                for key, prob in probabilities.items():
                    spec = data["probabilities"][key]
                    if spec.get("output") != "state":
                        continue
                    references = []
                    for j, row in enumerate(spec["rows"]):
                        folder = out / "rsr" / row["id"]
                        folder.mkdir(parents=True, exist_ok=True)
                        # RSR appends metrics; regeneration starts a fresh run.
                        (folder / "metrics.json").write_text("", encoding="utf-8")
                        print(f"Extracting system rules for {row['id']}...")
                        evaluate = prob.system_function(j)

                        def binary_system(comps):
                            # RSR 0.2.0 rounds its diagnostic fval and cannot
                            # accept infinite travel times. Extract the binary
                            # event itself; actual times remain unchanged.
                            _, state, _ = evaluate(comps)
                            return state, state, None

                        edges = spec["parent_rows"]["open"][j]
                        run_ref_extraction_by_mcs(
                            sfun=binary_system,
                            probs=torch.full((len(edges), 2), 0.5),
                            row_names=edges, n_state=2, sys_upper_st=1,
                            unk_prob_thres=0.001, unk_prob_opt="abs", max_rounds=128,
                            prob_update_every=10, save_every=128, n_sample=4096,
                            sample_batch_size=4096, max_search_loops=2,
                            ref_update_verbose=False, output_dir=str(folder))
                        references.append({
                            "upper": (folder / "refs_up_1.pt").relative_to(out).as_posix(),
                            "lower": (folder / "refs_low_0.pt").relative_to(out).as_posix(),
                        })
                        (folder / "model.json").write_text(json.dumps({
                            "rules_sha256": rules_fingerprint(spec, j), "origin": row["id"],
                            "edge_order": edges, "seed": seed,
                            "proposal": "independent Bernoulli(0.5), for rule search only",
                            "system_value": "binary state (travel times may be infinite)",
                            "unclassified": "evaluate with ndtools",
                        }, indent=2) + "\n", encoding="utf-8")
                    spec["references"] = references
        finally:
            random.setstate(python_state)

    (out / "model.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    script = _GENERATED_MODEL.replace("python -m src build toy-ca/demo", f"python -m src build {compiled.assembly['id']}")
    (out / "model.py").write_text(script, encoding="utf-8")
    return out


_GENERATED_MODEL = '''"""Generated tbnpy model. Regenerate with: python -m src build toy-ca/demo.

Run from any directory: python path/to/model.py --samples 2000 --device cpu
Or import define_variables() and define_probs(varis, device="cpu").
"""
from pathlib import Path
import argparse
import json
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "src/builder/bn_builder.py").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.builder.bn_builder import define_tbn_variables, define_tbn_probabilities


def read_data():
    return json.loads((BASE / "model.json").read_text(encoding="utf-8"))


def define_variables():
    return define_tbn_variables(read_data())


def define_probs(varis, device="cpu"):
    return define_tbn_probabilities(read_data(), varis, device=device, directory=BASE)


if __name__ == "__main__":
    import torch
    from tbnpy import inference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    varis = define_variables()
    probs = define_probs(varis, device=args.device)
    sampled = inference.sample(probs, list(probs), n_sample=args.samples)
    summary = {}
    for prob in sampled.values():
        for j, var in enumerate(prob.childs):
            x = prob.Cs[:, j]
            if isinstance(var.values, list):
                summary[var.name] = {"P": {str(s): (x == k).double().mean().item()
                                          for k, s in enumerate(var.values)}}
            else:
                finite = x[torch.isfinite(x)]
                summary[var.name] = {"mean_finite": finite.mean().item() if len(finite) else None,
                                     "P_inf": torch.isinf(x).double().mean().item()}
    result = {"samples": args.samples, "seed": args.seed, "device": args.device, "variables": summary}
    output = BASE / "sample_summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\\n", encoding="utf-8")
    print(f"Sampled {len(varis)} variables in {len(probs)} probability objects; wrote {output}")
'''
