"""python -m src {check,compile,query,ask,serve}

    check     validate schema/ and instances/, compile every assembly        (1, 2, 3)
    compile   show the BN one assembly expands to, or write its nodes         (3)
    query     answer a probabilistic query                                    (4)
    ask       answer a question in words through the configured model         (5)
    serve     the page: the BN, the query form and the chat                   (6)
    export    the page's drawings as one file that opens without the server   (6)
    diagram   rewrite schema/diagram.md and .html, the schema drawn            (1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def cmd_check(_: argparse.Namespace) -> int:
    from src.builder.compile import compile_path
    from src.builder.load import RegistryError, assembly_paths, load_library, rel

    lib = load_library()
    print(f"schema ok: {len(lib.quantities)} quantities, {len(lib.entities)} entity types, "
          f"{len(lib.functions)} functions; instances: {len(lib.parameter_sets)} parameter sets")
    failed = 0
    for path in assembly_paths():
        try:
            c = compile_path(path, lib)
        except RegistryError as exc:
            failed += 1
            print(f"FAIL {rel(path)}\n  " + "\n  ".join(exc.problems))
            continue
        print(f"ok   {c.assembly['id']}: {len(c.order)} bindings -> {c.n_nodes} BN nodes")
    return 1 if failed else 0


def cmd_compile(args: argparse.Namespace) -> int:
    from src.builder.compile import compile_path
    from src.builder.load import assembly_path

    c = compile_path(assembly_path(args.assembly))
    if args.output:
        graph = {"assembly": c.assembly["id"], "bindings": c.summary(), "nodes": c.nodes()}
        Path(args.output).write_text(json.dumps(graph, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.output}: {c.n_nodes} nodes")
        return 0
    for b in c.summary():
        sets = ", ".join(f"{k} x{v}" for k, v in b["parameter_sets"].items()) or (
            "inline" if b["inline_parameters"] else "-")
        print(f"#{b['binding']} {b['function']:<34} per {b['per']:<7} {b['nodes']:>4} nodes   params: {sets}")
        for port in b["inputs"]:
            k = port["parents_min"] if port["parents_min"] == port["parents_max"] else \
                f"{port['parents_min']}-{port['parents_max']}"
            print(f"     <- {port['name']}: {port['quantity']} over {port['over']} via {port['via']} ({k} per node)")
    print(f"{c.n_nodes} BN nodes")
    return 0


def _parse_value(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def cmd_query(args: argparse.Namespace) -> int:
    from src.inference.engine import answer

    if args.json:
        query = json.loads(Path(args.json).read_text(encoding="utf-8") if Path(args.json).is_file() else args.json)
    else:
        query = {"assembly": args.assembly}
        if args.target:
            query["targets"] = args.target
        if args.event:
            events = []
            for text in args.event:          # "TravelTime::n1>n22:time<=60"; event ids contain ">", so split at the last op
                for op in ("<=", ">=", "==", "!=", "<", ">"):
                    node, found, value = text.rpartition(op)
                    if found:
                        events.append({"node": node, "op": op, "value": _parse_value(value)})
                        break
                else:
                    raise SystemExit(f"--event {text!r} needs an operator: < <= > >= == !=")
            query["events"] = events
        for text in args.set or []:         # "ema-reference.mw=7" or "hazard.shakemap.event_id=..."
            key, _, value = text.partition("=")
            owner, _, name = key.rpartition(".")
            bucket = "parameters" if owner.count(".") >= 1 else "rows"
            query.setdefault(bucket, {}).setdefault(owner, {})[name] = _parse_value(value)
        for text in args.use or []:         # "functionality.usable-from-damage=policy.closed-at-moderate"
            fn, _, set_id = text.partition("=")
            query.setdefault("parameter_sets", {})[fn] = set_id
        query["samples"] = args.samples
    result = answer(query)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(result["statement"])
    print(f"{result['samples']} samples in {result['elapsed_s']} s; computed {' -> '.join(result['computed'])}")
    for w in result["warnings"]:
        print(f"note: {w}")
    for e in result["events"]:
        print(f"  P({e['node']} {e['op']} {e['value']}) = {e['p']:.4f}  [{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}]")
    for t in result["targets"]:
        print(f"  {t['node']:<36} {json.dumps(t['summary'])}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from src.llm import chat

    if args.llm:
        os.environ["BN_LLM"] = args.llm
    if args.model:
        os.environ["BN_MODEL"] = args.model

    def show(event: dict) -> None:
        if event["type"] == "query":
            print(f"query: {event['statement']}")
        elif event["type"] == "refused":
            print(f"refused: {event['error']}")

    reply = chat.ask(args.assembly, args.question, on_event=show)
    print(f"\n{reply['answer']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    if args.llm:
        os.environ["BN_LLM"] = args.llm
    if args.model:
        os.environ["BN_MODEL"] = args.model
    from src.app.server import serve

    serve(args.host, args.port, args.open)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import webbrowser

    from src.app.views import export_page

    out = export_page(Path(args.output))
    print(f"wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def cmd_diagram(args: argparse.Namespace) -> int:
    import webbrowser

    from src.app import views
    from src.builder import diagram

    if args.check:
        stale = [name for name, ok in (("schema/diagram.md", diagram.is_current()),
                                       ("schema/diagram.html", views.schema_page_is_current())) if not ok]
        if not stale:
            print("schema/diagram.md and schema/diagram.html are current")
            return 0
        print(f"{', '.join(stale)} stale; run: python -m src diagram", file=sys.stderr)
        return 1
    print(f"wrote {diagram.write()}")
    page = views.write_schema_page()
    print(f"wrote {page}")
    if args.open:
        webbrowser.open(page.resolve().as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate the schema and instances; compile every assembly").set_defaults(run=cmd_check)

    p = sub.add_parser("compile", help="show (or write) the BN one assembly expands to")
    p.add_argument("assembly", help="e.g. ema-highway/gmm")
    p.add_argument("-o", "--output", help="write nodes and parents as JSON")
    p.set_defaults(run=cmd_compile)

    p = sub.add_parser("query", help="answer a probabilistic query")
    p.add_argument("assembly", nargs="?", help="e.g. ema-highway/gmm (or give --json)")
    p.add_argument("--target", action="append", help="node id or pattern, e.g. 'TravelTime::*>n22:time'")
    p.add_argument("--event", action="append", help="e.g. 'SystemState::*>n22:*==0' or 'LinkOpen::*==0'")
    p.add_argument("--set", action="append", help="row.attr=value or function.param=value, e.g. ema-reference.mw=7")
    p.add_argument("--use", action="append", help="function=parameter-set, swapping a parameter set")
    p.add_argument("--json", help="a whole query, as a file or inline JSON (overrides the options above)")
    p.add_argument("-n", "--samples", type=int, default=2000)
    p.add_argument("-o", "--output", help="write the full result as JSON")
    p.set_defaults(run=cmd_query)

    p = sub.add_parser("ask", help="answer a question in words")
    p.add_argument("assembly")
    p.add_argument("question")
    p.add_argument("--llm", choices=["ollama", "gemini", "anthropic"])
    p.add_argument("--model")
    p.set_defaults(run=cmd_ask)

    p = sub.add_parser("serve", help="the interactive page")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open it in the browser")
    p.add_argument("--llm", choices=["ollama", "gemini", "anthropic"])
    p.add_argument("--model")
    p.set_defaults(run=cmd_serve)

    p = sub.add_parser("export", help="write the drawings as one self-contained HTML file")
    p.add_argument("-o", "--output", default=".cache/schema.html", help="default .cache/schema.html (gitignored)")
    p.add_argument("--open", action="store_true", help="open it in the browser")
    p.set_defaults(run=cmd_export)

    p = sub.add_parser("diagram", help="rewrite schema/diagram.md and schema/diagram.html from the schema")
    p.add_argument("--check", action="store_true", help="fail if either is stale instead of rewriting them")
    p.add_argument("--open", action="store_true", help="open schema/diagram.html in the browser")
    p.set_defaults(run=cmd_diagram)

    args = parser.parse_args(argv)
    if args.command == "query" and not (args.assembly or args.json):
        parser.error("query needs an assembly or --json")
    from src.builder.load import RegistryError
    from src.inference.engine import QueryError

    try:
        return args.run(args)
    except (RegistryError, QueryError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
