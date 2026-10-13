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
    search.add_argument("--source", choices=["auto", "shakecast", "comcat"])
    fetch = shake_sub.add_parser("fetch", help="download (or locate) one event's ShakeMap")
    fetch.add_argument("event_id")
    fetch.add_argument("--version", type=int)
    fetch.add_argument("--source", choices=["auto", "shakecast", "comcat"])
    watch = shake_sub.add_parser("watch", help="poll the USGS feed and run packages on new ShakeMaps")
    watch.add_argument("--once", action="store_true", help="poll once and exit (for cron or a scheduled job)")
    watch.add_argument("--interval", type=float, default=120, help="seconds between polls when looping")
    watch.add_argument("--feed", default="2.5_day", help="USGS summary feed, e.g. 2.5_hour, 2.5_day, all_week")
    watch.add_argument("--min-magnitude", type=float, default=4.0)
    watch.add_argument("--samples", type=int, default=5000)
    watch.add_argument("--state-dir", type=Path, help="where state, results and the heartbeat are kept")
    watch.add_argument("--packages", type=Path, default=DEFAULT_OUTPUT)
    beat = shake_sub.add_parser("heartbeat", help="exit 0 if the watcher is alive and well, 1 if not")
    beat.add_argument("--max-age-minutes", type=float, default=10)
    beat.add_argument("--state-dir", type=Path)
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

        if args.action in ("watch", "heartbeat"):
            import sys

            from . import watch as watcher

            state_dir = args.state_dir or watcher.DEFAULT_STATE_DIR
            if args.action == "heartbeat":
                ok, message = watcher.check_heartbeat(state_dir, args.max_age_minutes)
                print(message)
                sys.exit(0 if ok else 1)
            options = dict(feed=args.feed, state_dir=state_dir, packages_root=args.packages,
                           min_magnitude=args.min_magnitude, samples=args.samples)
            if not args.once:
                watcher.run_forever(args.interval, **options)
            try:
                beat = watcher.poll(**options)
            except watcher.PollInProgress as exc:
                print(exc)
                return
            except watcher.WatchError as exc:
                print(f"poll failed: {exc}")
                sys.exit(1)
            print(f"{beat['status']}: {beat['events_in_feed']} event(s) in feed, "
                  f"{beat['events_matched']} near a package, {len(beat['runs'])} run(s)")
            return
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
