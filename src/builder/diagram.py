"""Draw the schema as Mermaid diagrams in schema/diagram.md: plain text that GitHub renders.

    python -m src diagram            rewrite schema/diagram.md
    python -m src diagram --check    fail if it is stale (the tests run this too)

Two entity-relationship diagrams, in the manner of Supabase's schema visualiser:

* Functions as tables: each output is a key (it is the BN node the function
  defines) and each input a foreign key to the node it reads. A link joins an
  output to every input of the same quantity on the same plate, and its ends say
  how many nodes meet: one parent per node (via self), many nodes sharing one
  parent (via a foreign key), many parents per node (via a reverse key, or all).
* Plates as tables: entity types, their attributes and their foreign keys.

Output is deterministic, so regenerating an unchanged schema changes nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .load import SCHEMA, Library, load_library, port_choices, quantity_template, via_kind

OUTPUT = SCHEMA / "diagram.md"
ROLES = ["scenario", "hazard", "fragility", "edge", "system"]

# Crow's-foot ends, producer on the left, consumer on the right, by how the consumer reaches it.
_ENDS = {"self": "||--||", "fk": "||--o{", "reverse": "}o--||", "all": "}o--o{"}
_PARENTS = {"self": "one parent, the same row", "fk": "many nodes share one parent",
            "reverse": "many parents per node", "all": "every node of the plate is a parent"}


def _name(text: str) -> str:
    """A Mermaid identifier: letters, digits and underscores only."""
    return re.sub(r"\W", "_", text)


def _quantity(port: dict) -> str:
    """The row's type column: the quantity, or by_imt for a port whose quantity its parameters name."""
    param = quantity_template(port["quantity"])
    return _name(port["quantity"]) if param is None else f"by_{param}"


def _comment(text: str) -> str:
    return '"' + text.replace('"', "'") + '"'


def _type_of(q: dict) -> str:
    if q["kind"] == "discrete":
        return "{0,1}" if q["states"] == [0, 1] else ", ".join(map(str, q["states"]))
    if q["kind"] == "record":
        return "record of " + ", ".join(q["fields"])
    return q["unit"]


def _functions(lib: Library) -> list[str]:
    sets = Counter(s["function"] for s in lib.parameter_sets.values())
    fns = sorted(lib.functions.values(), key=lambda f: (ROLES.index(f["role"]), f["id"]))
    lines = ["```mermaid", "erDiagram"]
    for fn in fns:
        lines.append(f"    {_name(fn['id'])} {{")
        for port in fn["outputs"]:
            lines.append(f"        {_quantity(port)} {port['name']} PK "
                         + _comment(f"output, one node per {fn['per']}"))
        for port in fn["inputs"]:
            lines.append(f"        {_quantity(port)} {port['name']} FK "
                         + _comment(f"input from {port['over']} via {port['via']}"))
        for attr in fn.get("attributes", []):
            lines.append(f"        fixed {attr} " + _comment(f"{fn['per']}.{attr}"))
        for table in fn.get("tables", []):
            lines.append(f"        table {table} " + _comment("whole table, fixed"))
        if "instance_schema" in fn:
            where = (f"{sets[fn['id']]} set(s) in instances/{fn['role']}" if sets[fn["id"]]
                     else "inline, in each assembly that binds it")
            lines.append("        theta parameters " + _comment(where))
        lines.append(f"        {fn['role']} role " + _comment(
            f"{'deterministic' if fn['deterministic'] else 'stochastic'}, {fn['drawKind']}"))
        lines.append("    }")
    for consumer in fns:
        for port in consumer["inputs"]:
            for producer in fns:
                if producer["per"] == port["over"] and any(
                        set(port_choices(producer, o)) & set(port_choices(consumer, port)) for o in producer["outputs"]):
                    kind = via_kind(port["via"])
                    label = port["quantity"] if kind == "self" else f"{port['quantity']} via {port['via']}"
                    lines.append(f"    {_name(producer['id'])} {_ENDS[kind]} {_name(consumer['id'])} : {_comment(label)}")
    lines.append("```")
    return lines


def _plates(lib: Library) -> list[str]:
    lines = ["```mermaid", "erDiagram"]
    for e in lib.entities.values():
        lines.append(f"    {e['id']} {{")
        lines.append("        string id PK" + (' "singleton: its variables are global"' if e.get("singleton") else ""))
        for name, a in e["attributes"].items():
            if "ref" in a:
                lines.append(f"        string {name} FK {_comment('to ' + a['ref'])}")
            else:
                note = ", ".join(x for x in (a.get("unit"), "optional" if a.get("required") is False else None) if x)
                lines.append(f"        {a['type']} {name}" + (f" {_comment(note)}" if note else ""))
        lines.append("    }")
    for e in lib.entities.values():
        for name, a in e["attributes"].items():
            if "ref" in a:
                lines.append(f"    {e['id']} }}o--|| {a['ref']} : {name}")
    lines.append("```")
    return lines


def render(lib: Library) -> str:
    uses: dict[str, list[str]] = {q: [] for q in lib.quantities}
    for fn in lib.functions.values():
        for port in fn["outputs"]:
            for q in port_choices(fn, port):
                uses[q].append(f"{fn['id']} (out)")
        for port in fn["inputs"]:
            for q in port_choices(fn, port):
                uses[q].append(f"{fn['id']} (in)")
    rows = [f"| `{q['id']}` | {q['kind']} | {_type_of(q)} | {q['label']} | "
            + ", ".join(f"`{u}`" for u in sorted(uses[q['id']])) + " |" for q in lib.quantities.values()]
    legend = " · ".join(f"`{end}` {text}" for end, text in
                        ((_ENDS[k], _PARENTS[k]) for k in ("self", "fk", "reverse", "all")))
    return "\n".join([
        "# Schema diagram",
        "",
        "<!-- Generated by `python -m src diagram` from schema/*.json; edit those, not this. "
        "The tests fail if it is stale. -->",
        "",
        f"{len(lib.functions)} functions, {len(lib.entities)} plates and {len(lib.quantities)} quantities. "
        "GitHub draws the diagrams below; in VS Code the Markdown preview needs a Mermaid extension. "
        "[`diagram.html`](diagram.html), beside this file, draws the same schema interactively: "
        "open it in a browser.",
        "",
        "## Functions: what can connect to what",
        "",
        "Each function is a table. Its output (PK) is the BN node it defines, one per row of its plate; "
        "its inputs (FK) are the nodes it reads. A link joins an output to every input of the same "
        "quantity on the same plate; where several hazard functions feed one input, they are "
        "alternatives, and an assembly binds one.",
        "",
        f"Link ends, producer left, consumer right: {legend}.",
        "",
        *_functions(lib),
        "",
        "## Plates: the tables functions are replicated over",
        "",
        "Attributes are fixed data, never BN nodes. Foreign keys are what an input's `via` follows.",
        "",
        *_plates(lib),
        "",
        "## Quantities: the types on the links",
        "",
        "| Quantity | Kind | Unit or states | Meaning | Used by |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ])


def write(path: Path = OUTPUT, lib: Library | None = None) -> Path:
    path.write_text(render(lib or load_library()), encoding="utf-8")
    return path


def is_current(path: Path = OUTPUT, lib: Library | None = None) -> bool:
    return path.is_file() and path.read_text(encoding="utf-8") == render(lib or load_library())
