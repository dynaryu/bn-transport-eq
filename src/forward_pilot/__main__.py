"""Command-line interface for assembling, querying, and serving the pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import run_forward_query
from .package import DEFAULT_OUTPUT, assemble_all, assemble_package, load_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="assemble portable BN packages")
    build.add_argument("--case", choices=["ema-highway", "hebei-transport"])
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    query = sub.add_parser("query", help="run a forward query against a package")
    query.add_argument("package", type=Path)
    query.add_argument("query", type=Path)
    query.add_argument("--output", type=Path)
    shake = sub.add_parser("shakemap", help="find and fetch USGS ShakeMaps for a package")
    shake_sub = shake.add_subparsers(dest="action", required=True)
    search = shake_sub.add_parser("search", help="events with a ShakeMap around a package's network")
    search.add_argument("--case", default="hebei-transport", help="package whose node lat/lon set the area")
    search.add_argument("--start", default="1900-01-01")
    search.add_argument("--end")
    search.add_argument("--min-magnitude", type=float)
    search.add_argument("--source", choices=["comcat", "shakecast"])
    fetch = shake_sub.add_parser("fetch", help="download (or locate) one event's ShakeMap")
    fetch.add_argument("event_id")
    fetch.add_argument("--version", type=int)
    fetch.add_argument("--source", choices=["comcat", "shakecast"])
    serve = sub.add_parser("serve", help="serve the local stakeholder UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--packages", type=Path, default=DEFAULT_OUTPUT)
    serve.add_argument("--open", action="store_true", help="open the UI in the default browser")
    args = parser.parse_args()

    if args.command == "build":
        paths = [assemble_package(args.case, args.output)] if args.case else assemble_all(args.output)
        print("\n".join(str(path) for path in paths))
    elif args.command == "query":
        network = load_package(args.package)
        request = json.loads(args.query.read_text(encoding="utf-8"))
        result = run_forward_query(network, request)
        rendered = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(args.output)
        else:
            print(rendered)
    elif args.command == "shakemap":
        from . import shakemap

        if args.action == "search":
            from .catalog import load_case

            events = shakemap.search_events(load_case(args.case), args.source, start=args.start,
                                            end=args.end, min_magnitude=args.min_magnitude)
            for e in events:
                print(f"{(e.get('time') or '')[:10]:10}  M{e.get('magnitude') or 0:<4}  {e['event_id']:<16} {e.get('place') or ''}")
            print(f"{len(events)} event(s)")
        else:
            grid = shakemap.load(args.event_id, args.source, args.version)
            print(json.dumps({**grid.meta, "extent": grid.extent,
                              "has_uncertainty": grid.std_psa10 is not None}, indent=2))
    else:
        from .server import serve

        serve(args.host, args.port, args.packages, open_browser=args.open)


if __name__ == "__main__":
    main()
