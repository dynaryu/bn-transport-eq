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
    else:
        from .server import serve

        serve(args.host, args.port, args.packages, open_browser=args.open)


if __name__ == "__main__":
    main()
