"""What the page draws: the schema and every compiled BN, as Supabase-style tables.

Two drawings, as in Supabase's schema visualiser:

* Functions: each function (or binding) is a table, its ports are its columns,
  and a link joins an output to every input of the same quantity on the same
  plate. The library view shows every connection that is possible; an assembly
  view shows the ones it makes, with the node counts it expands to.
* Plates: the entity types and their foreign keys, which are what route a
  variable from one plate to another.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.builder.compile import compile_path
from src.builder.load import (Library, RegistryError, assembly_paths, load_assembly, load_network, port_choices,
                              quantity_template, via_kind)
from src.inference import engine


def _type_label(q: dict[str, Any]) -> str:
    if q["kind"] == "discrete":
        states = q["states"]
        return "{0,1}" if states == [0, 1] else f"enum({len(states)})"
    if q["kind"] == "record":
        return f"record({len(q['fields'])})"
    return q["unit"]


def _ports(fn: dict[str, Any], ports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each port, with the quantities it may carry: one, or for an "{imt}" port every IM its parameter allows."""
    return [{**p, "choices": port_choices(fn, p), "template": quantity_template(p["quantity"])} for p in ports]


def _card(fn: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "key": key,
        "function": fn["id"],
        "label": fn["label"],
        "role": fn["role"],
        "per": fn["per"],
        "inputs": [{**p, "kind": via_kind(p["via"])} for p in _ports(fn, fn["inputs"])],
        "outputs": _ports(fn, fn["outputs"]),
        "attributes": fn.get("attributes", []),
        "tables": fn.get("tables", []),
        "has_parameters": "parameters" in fn,
        "implementation": fn["implementation"],
        "deterministic": fn["deterministic"],
        "drawKind": fn["drawKind"],
        "capabilities": fn["capabilities"],
        "description": fn.get("description", ""),
    }


def _library_view(lib: Library) -> dict[str, Any]:
    by_function = Counter(s["function"] for s in lib.parameter_sets.values())
    cards = []
    for fn in lib.functions.values():
        card = _card(fn, fn["id"])
        card["parameter_sets"] = [{"id": sid, "label": s["label"], "count": None}
                                  for sid, s in lib.parameter_sets.items() if s["function"] == fn["id"]]
        card["n_parameter_sets"] = by_function[fn["id"]]
        cards.append(card)
    links = []
    for consumer in lib.functions.values():
        for port in consumer["inputs"]:
            for producer in lib.functions.values():
                for out in producer["outputs"]:
                    common = set(port_choices(producer, out)) & set(port_choices(consumer, port))
                    if common and producer["per"] == port["over"]:
                        # Templated on both sides, the link holds for whichever IM the sets agree on.
                        shown = port["quantity"] if quantity_template(port["quantity"]) else sorted(common)[0]
                        links.append({"from": producer["id"], "out": out["name"], "to": consumer["id"],
                                      "in": port["name"], "quantity": shown,
                                      "via": port["via"], "kind": via_kind(port["via"]), "parents": None})
    return {"id": "library", "label": "Schema: every possible connection", "network": None,
            "cards": cards, "links": links, "rows": {}, "problems": []}


def _assembly_view(path: Path, lib: Library) -> dict[str, Any]:
    assembly = load_assembly(path)
    try:
        c = compile_path(path, lib)
    except RegistryError as exc:
        return {"id": assembly["id"], "label": assembly["label"], "network": assembly["network"],
                "cards": [], "links": [], "rows": {}, "problems": exc.problems, "menu": None}

    nodes = {n["id"]: n for n in c.nodes()}
    cards, links = [], []
    for b in (c.bindings[i] for i in c.order):
        key = f"b{b.index}"
        card = _card(b.function, key)
        card["binding"] = b.index
        card["rows"] = len(b.rows)
        card["nodes"] = len(b.rows) * len(b.function["outputs"])
        card["where"] = b.spec.get("where")
        card["note"] = b.spec.get("note")
        card["inline_parameters"] = b.spec.get("parameters")
        template = b.spec.get("parameter_set", "")
        card["parameter_template"] = template if "{" in template else None
        card["parameter_sets"] = [{"id": sid, "label": lib.parameter_sets[sid]["label"], "count": n}
                                  for sid, n in Counter(s for s in b.parameter_sets if s).most_common()]
        for port in card["inputs"] + card["outputs"]:
            port["quantity"] = b.port_quantity(port["name"])           # what "{imt}" resolved to here
        card["parameter_by_location"] = b.spec.get("parameter_set_by_location")
        first_row = b.rows[0]["id"]
        example = nodes[f"{b.quantities[0][b.function['outputs'][0]['name']]}::{first_row}"]
        card["example"] = {"id": example["id"], "parameter_set": example["parameter_set"],
                           "parents": example["parents"][:6], "n_parents": len(example["parents"])}
        for port in card["inputs"]:
            counts = [len(p) for p in b.parents[port["name"]]]
            lo, hi = min(counts), max(counts)
            port["parents"] = str(lo) if lo == hi else f"{lo}-{hi}"
            for up in sorted(b.upstream):
                producer = c.bindings[up]
                for out in producer.function["outputs"]:
                    made = set(producer.port_quantity(out["name"]).split(" | "))
                    if made & set(port["quantity"].split(" | ")) and producer.function["per"] == port["over"]:
                        links.append({"from": f"b{up}", "out": out["name"], "to": key, "in": port["name"],
                                      "quantity": port["quantity"], "via": port["via"],
                                      "kind": port["kind"], "parents": port["parents"]})
        cards.append(card)
    rows = {name: len(table) for name, table in c.network.tables.items()}
    return {"id": assembly["id"], "label": assembly["label"], "network": assembly["network"],
            "description": assembly.get("description", ""), "cards": cards, "links": links,
            "rows": rows, "n_nodes": c.n_nodes, "assumptions": c.network.assumptions, "problems": [],
            "menu": engine.describe(assembly["id"])}


def view_data(schema_only: bool = False) -> dict[str, Any]:
    """Every view the page can draw; with ``schema_only``, just the schema's, nothing compiled."""
    lib = engine.library()
    views = [_library_view(lib)]
    if not schema_only:
        views += [_assembly_view(p, lib) for p in assembly_paths()]
    networks = {}
    for view in views:
        if view["network"] and view["network"] not in networks:
            network = load_network(view["network"], lib)
            networks[network.id] = {"title": network.title, "rows": {k: len(v) for k, v in network.tables.items()}}
    quantities = {qid: {**q, "type": _type_label(q)} for qid, q in lib.quantities.items()}
    return {"quantities": quantities, "entities": list(lib.entities.values()),
            "views": views, "networks": networks}


SCHEMA_PAGE = Path(__file__).resolve().parents[2] / "schema" / "diagram.html"


def static_page(data: dict[str, Any], indent: int | None = None) -> str:
    """The page with ``data`` embedded, so it opens without the server: drawings and details, no queries."""
    embedded = json.dumps(data, ensure_ascii=False, indent=indent).replace("</", "<\\/")   # no "</script>" inside
    page = Path(__file__).with_name("page.html").read_text(encoding="utf-8")
    marker = "<script>\nlet DATA = "
    if marker not in page:
        raise RuntimeError("page.html no longer has the marker the data is injected before")
    return page.replace(marker, f"<script>window.BN_DATA = {embedded};</script>\n{marker}", 1)


def export_page(output: Path) -> Path:
    """Every view, as a dated snapshot for sharing; written outside the repository by default."""
    from datetime import datetime, timezone

    data = {**view_data(), "generated_by": "python -m src export",
            "exported": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(static_page(data), encoding="utf-8")
    return output


def schema_page() -> str:
    """schema/diagram.html: the schema's views only, undated, indented, so it is committed and diffable."""
    return static_page({**view_data(schema_only=True), "generated_by": "python -m src diagram"}, indent=1)


def write_schema_page(path: Path = SCHEMA_PAGE) -> Path:
    path.write_text(schema_page(), encoding="utf-8")
    return path


def schema_page_is_current(path: Path = SCHEMA_PAGE) -> bool:
    return path.is_file() and path.read_text(encoding="utf-8") == schema_page()
