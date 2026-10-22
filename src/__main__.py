"""python -m src {check,compile,build,rules,query,ask,serve}

    check     validate schema/ and instances/, compile every assembly        (1, 2, 3)
    compile   show the BN one assembly expands to, or write its nodes         (3)
    build     generate tbnpy models and RSR rules beneath each example        (3)
    rules     extract RSR rules for another target_add_h, stored beside them  (3)
    replay    re-run a recorded query and check it reproduces exactly         (4)
    scope     which part of the BN a query depends on, and what it samples    (4)
    query     answer a probabilistic query                                    (4)
    ask       answer a question in words through the configured model         (5)
    serve     the page: the BN, the query form and the chat                   (6)
    export    the page's drawings as one file that opens without the server   (6)
    diagram   rewrite schema/diagram.md and .html, the schema drawn            (1)
    vs30      fill missing bridge Vs30 from the USGS map                      (2)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def cmd_check(_: argparse.Namespace) -> int:
    from src.builder.bn_builder import compile_path
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
    from src.builder.bn_builder import compile_path
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


def cmd_build(args: argparse.Namespace) -> int:
    from src.builder.bn_builder import generate_tbn
    from src.builder.load import assembly_path

    out = generate_tbn(assembly_path(args.assembly), extract_rules=not args.no_rules)
    print(f"wrote tbnpy model: {out}")
    return 0


def cmd_rules(args: argparse.Namespace) -> int:
    from src.builder.bn_builder import compile_assembly, extract_system_rules, model_dir
    from src.builder.load import (RegistryError, assembly_path, check_tables, load_assembly, load_library,
                                  load_network, rel)

    path, lib = assembly_path(args.assembly), load_library()
    assembly = load_assembly(path)
    network = load_network(assembly["network"], lib)
    for event in network.rows("SystemEvent"):
        if args.target_add_h is not None:
            event["target_add_h"] = args.target_add_h
        threshold = event["target_add_h"]
    _, problems = check_tables(network.tables, lib, "--target-add-h")
    if problems:
        raise RegistryError(problems)
    out = model_dir(path)
    done = extract_system_rules(compile_assembly(assembly, lib, network), out, set(args.origins) if args.origins else None)
    print(f"stored rsr rules for {len(done)} origin(s) at target_add_h={threshold:g} under {rel(out)}/rsr/")
    return 0


def cmd_scope(args: argparse.Namespace) -> int:
    from src.inference.engine import scope

    s = scope(_query_from_args(args))
    rel, smp = s["relevant"], s["sampled"]
    print(s["statement"])
    print(f"query nodes ({len(s['query_nodes'])}): {', '.join(s['query_nodes'][:8])}{' ...' if len(s['query_nodes']) > 8 else ''}")
    print(f"relevant: {rel['count']} of {rel['of']} variables, the query nodes and their ancestors")
    for quantity, count in rel["by_quantity"].items():
        print(f"  {quantity:<26} {count}")
    print(f"pruned ({len(s['pruned'])}): {', '.join(s['pruned'][:8])}{' ...' if len(s['pruned']) > 8 else ''}")
    print(f"sampled: {smp['variables']} variables in {smp['objects']} probability objects"
          + (f"; {len(smp['beyond_relevant'])} beyond the relevant ones, because "
             f"{', '.join(s['joint'])} draws its whole field as one joint object" if smp["beyond_relevant"] else ""))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from src.inference import engine, records

    record = records.find(args.record)
    if record["kind"] != "query":
        raise SystemExit(f"{args.record} is an {record['kind']} record; replay one of its queries: "
                         + ", ".join(t["record"] for t in record["translations"] if t.get("record")))
    now = records._code()
    was = record["provenance"]["code"]
    if now["commit"] != was["commit"] or was["uncommitted_changes"] or now["uncommitted_changes"]:
        print(f"note: recorded at commit {(was['commit'] or '?')[:10]}"
              f"{' with uncommitted changes' if was['uncommitted_changes'] else ''}; now at {(now['commit'] or '?')[:10]}"
              f"{' with uncommitted changes' if now['uncommitted_changes'] else ''}: a difference may be the code's")
    # Sampled as the record was, afresh: the whole BN if it was answered from a bank, else its scope.
    result = engine.answer(record["query"], samples_policy="whole" if record["samples_file"]["bank"] else "scope")
    same = records.digest(result) == records.digest(record)
    print(f"{record['statement']}\n{'reproduced exactly' if same else 'NOT reproduced'}: "
          f"{result['samples']} samples, seed {result['seed']}, {result['elapsed_s']} s")
    if not same:
        for old, new in zip(record["events"], result["events"]):
            if old["p"] != new["p"]:
                print(f"  {old['node']}: recorded {old['p']:.4f}, now {new['p']:.4f}")
    return 0 if same else 1


def _parse_value(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _query_from_args(args: argparse.Namespace) -> dict:
    """A query from the query and scope commands' options, or their --json."""
    if args.json:
        query = json.loads(Path(args.json).read_text(encoding="utf-8") if Path(args.json).is_file() else args.json)
    else:
        query = {"assembly": args.assembly}
        if args.target:
            query["targets"] = args.target
        if args.event:
            events = []
            for text in args.event:          # "SystemState::n4==0"; split at the last operator
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
    return query


def cmd_query(args: argparse.Namespace) -> int:
    from src.inference.engine import answer

    result = answer(_query_from_args(args), record={"source": "cli"})
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(result["statement"])
    print(f"{result['samples']} samples in {result['elapsed_s']} s; computed {' -> '.join(result['computed'])}")
    print(f"samples from {result['samples_from']}; recorded as {result['record']['id']}")
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


def cmd_analyze(args: argparse.Namespace) -> int:
    from src.inference.comparisons import execute
    task = json.loads(Path(args.json).read_text(encoding="utf-8"))
    result = execute(task)
    print(json.dumps(result, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from src.llm.cases import check_contracts, evaluate
    result = evaluate() if args.live else check_contracts()
    print(json.dumps(result, indent=2))
    return int(args.live and result["passed"] != result["total"])


def cmd_proposals(args: argparse.Namespace) -> int:
    from src import proposals
    if args.action == "list":
        result = proposals.list_proposals()
    elif args.action == "draft":
        if not args.json:
            raise SystemExit("draft needs --json <file>")
        result = proposals.draft(json.loads(Path(args.json).read_text(encoding="utf-8")))
    else:
        if not (args.id and args.reviewer and args.sha256):
            raise SystemExit("activate needs --id, --reviewer and --sha256 from the reviewed draft")
        result = proposals.activate(args.id, args.reviewer, args.sha256)
    print(json.dumps(result, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    if args.llm:
        os.environ["BN_LLM"] = args.llm
    if args.model:
        os.environ["BN_MODEL"] = args.model
    from src.app.server import serve

    serve(args.host, args.port, args.open, args.example)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import webbrowser

    from src.app.views import export_page

    out = export_page(Path(args.output))
    print(f"wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def cmd_vs30(args: argparse.Namespace) -> int:
    from src.builder import vs30

    if args.check:
        data = json.loads((vs30.NETWORKS / args.network / "network.json").read_text(encoding="utf-8"))
        missing = [b["id"] for b in data["tables"].get("Bridge", []) if b.get("vs30") is None]
        print(f"{len(missing)} bridge(s) without vs30" + (f": {', '.join(missing[:8])}" if missing else ""))
        return 0
    try:
        done = vs30.fill(args.network)
    except vs30.Vs30Error as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"{done['file']}: filled {done['filled']} from the USGS mosaic ({done['near_water']} from a nearby "
          f"land cell), kept {done['kept']} given")
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

    p = sub.add_parser("build", help="generate tbnpy model files beneath the example folder")
    p.add_argument("assembly", help="e.g. toy-ca/demo")
    p.add_argument("--no-rules", action="store_true", help="use ndtools directly, without extracting RSR rules")
    p.set_defaults(run=cmd_build)

    p = sub.add_parser("rules", help="extract RSR rules for a threshold, kept beside the ones already stored")
    p.add_argument("assembly", help="e.g. toy-ca/demo")
    p.add_argument("--target-add-h", type=float, help="the threshold, in hours (default: the network's own)")
    p.add_argument("--origins", nargs="+", help="only these origin nodes (default: every origin)")
    p.set_defaults(run=cmd_rules)

    p = sub.add_parser("replay", help="re-run a recorded query and check it reproduces exactly")
    p.add_argument("record", help="a query record id from records/queries.jsonl")
    p.set_defaults(run=cmd_replay)

    def query_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("assembly", nargs="?", help="e.g. ema-highway/gmm (or give --json)")
        p.add_argument("--target", action="append", help="node id or pattern, e.g. 'SA(1.0)::*'")
        p.add_argument("--event", action="append", help="e.g. 'SystemState::*>n22:*==0' or 'EdgeOpen::*==0'")
        p.add_argument("--set", action="append", help="row.attr=value or function.param=value, e.g. ema-reference.mw=7")
        p.add_argument("--use", action="append", help="function=parameter-set, swapping a parameter set")
        p.add_argument("--json", help="a whole query, as a file or inline JSON (overrides the options above)")
        p.add_argument("-n", "--samples", type=int, default=100_000)

    p = sub.add_parser("query", help="answer a probabilistic query")
    query_options(p)
    p.add_argument("-o", "--output", help="write the full result as JSON")
    p.set_defaults(run=cmd_query)

    p = sub.add_parser("scope", help="which part of the BN a query depends on, and what it samples")
    query_options(p)
    p.set_defaults(run=cmd_scope)

    p = sub.add_parser("ask", help="answer a question in words")
    p.add_argument("assembly")
    p.add_argument("question")
    p.add_argument("--llm", choices=["ollama", "gemini", "anthropic"])
    p.add_argument("--model")
    p.set_defaults(run=cmd_ask)

    p = sub.add_parser("analyze", help="execute a deterministic comparison, importance, paths or VOI plan")
    p.add_argument("--json", required=True, help="analysis plan JSON file")
    p.set_defaults(run=cmd_analyze)

    p = sub.add_parser("evaluate", help="validate answer cases, or evaluate held-out translations with --live")
    p.add_argument("--live", action="store_true", help="call the configured LLM; log translations, without numerical runs")
    p.set_defaults(run=cmd_evaluate)

    p = sub.add_parser("proposals", help="draft, inspect or activate reviewed model inputs")
    p.add_argument("action", choices=["list", "draft", "activate"])
    p.add_argument("--json")
    p.add_argument("--id")
    p.add_argument("--reviewer")
    p.add_argument("--sha256")
    p.set_defaults(run=cmd_proposals)

    p = sub.add_parser("serve", help="the interactive page")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open it in the browser")
    p.add_argument("--example", help="the example to open on, e.g. toy-ca/demo")
    p.add_argument("--llm", choices=["ollama", "gemini", "anthropic"])
    p.add_argument("--model")
    p.set_defaults(run=cmd_serve)

    p = sub.add_parser("export", help="write the drawings as one self-contained HTML file")
    p.add_argument("-o", "--output", default=".cache/schema.html", help="default .cache/schema.html (gitignored)")
    p.add_argument("--open", action="store_true", help="open it in the browser")
    p.set_defaults(run=cmd_export)

    p = sub.add_parser("vs30", help="fill missing Bridge.vs30 from the USGS Global Vs30 Mosaic")
    p.add_argument("network", help="e.g. toy-ca")
    p.add_argument("--check", action="store_true", help="only report which bridges lack vs30")
    p.set_defaults(run=cmd_vs30)

    p = sub.add_parser("diagram", help="rewrite schema/diagram.md and schema/diagram.html from the schema")
    p.add_argument("--check", action="store_true", help="fail if either is stale instead of rewriting them")
    p.add_argument("--open", action="store_true", help="open schema/diagram.html in the browser")
    p.set_defaults(run=cmd_diagram)

    args = parser.parse_args(argv)
    if args.command in ("query", "scope") and not (args.assembly or args.json):
        parser.error(f"{args.command} needs an assembly or --json")
    from src.builder.load import RegistryError
    from src.inference.engine import QueryError

    try:
        return args.run(args)
    except (RegistryError, QueryError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
