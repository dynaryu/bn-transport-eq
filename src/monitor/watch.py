"""Watch the USGS feed for new ShakeMaps and run each affected network's BN on them.

The part of ShakeCast quakenet needs, and nothing more: poll the real-time feed,
keep events with a ShakeMap whose epicentre lies around a network that has a
ShakeMap assembly (``examples/<network>/shakemap.assembly.json``), download each
new ShakeMap version, and start that assembly's session on it, exactly as the
page's Run does. No language model is involved: when a ShakeMap arrives there is
nothing to choose, the event is the scenario.

One poll:

1. Read the USGS real-time summary feed (``feedwatch.FEED_URL``). USGS asks that
   frequent polling use these feeds rather than the search API.
2. Keep events whose ``types`` include ``shakemap``, at or above the magnitude
   floor, with the epicentre inside a network's box (``shakemap.bbox`` of its nodes).
3. Skip an event whose ``updated`` time is unchanged since the last poll. A
   changed one is looked up; ShakeMaps are revised for hours as data arrives,
   and a new *version* is downloaded (``ComcatSource``, the same cache as the
   page) and run again.
4. For each assembly not yet run on that version: sample its BN with the event's
   map and write the result to ``results/<event>/<event>-<version>/<assembly>.json``.
   The page's own saved sessions are never touched.

Every poll ends by writing ``heartbeat.json``, whether it succeeded or not, so a
watcher that has died or keeps failing is visible *before* an earthquake.
``check_heartbeat`` turns that file into an exit code for cron or an alarm.

Run it once per minute from cron or a systemd timer (``--once``), or as a
long-running loop (``--interval``). A lock file keeps two polls from overlapping.
``feedwatch`` is the lighter variant deployed as a Lambda: it runs no BN, and
emails new earthquakes instead.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from src.builder import shakemap
from src.builder.load import ROOT, assembly_id_of, assembly_path, assembly_paths, load_assembly, load_network

from .feedwatch import DEFAULT_FEED, DEFAULT_MIN_MAGNITUDE, WatchError, _inside, fetch_feed

DEFAULT_STATE_DIR = Path(os.environ.get("QUAKENET_WATCH_DIR") or ROOT / ".cache" / "watch")
DEFAULT_SAMPLES = 100_000
LOCK_STALE_SECONDS = 30 * 60
FUNCTION = "hazard.shakemap"


class PollInProgress(WatchError):
    """Another poll holds the lock. Not a failure: this poll simply does not run,
    and leaves the heartbeat to the poll that is running."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    """Write atomically, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


# ---- which assemblies take a ShakeMap ------------------------------------------------


def shakemap_parameters(assembly_id: str, settings: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The assembly's ``hazard.shakemap`` parameters with the scenario's settings applied, or None
    when it binds no ShakeMap (or takes one from a parameter set, which nothing here can change)."""
    for binding in load_assembly(assembly_path(assembly_id))["bindings"]:
        if binding["function"] == FUNCTION and "parameters" in binding:
            return {**binding["parameters"], **((settings or {}).get("parameters") or {}).get(FUNCTION, {})}
    return None


def shakemap_assemblies(ids: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """{assembly id: {network, bbox, sites}} for every assembly that binds a ShakeMap (or only ``ids``),
    with the box events are matched against (its network's nodes, plus a degree) and the sites a map
    must cover for the assembly to run on it (see ``sites``)."""
    lib = None
    out = {}
    for path in assembly_paths():
        assembly_id = assembly_id_of(path)
        if (ids is not None and assembly_id not in ids) or shakemap_parameters(assembly_id) is None:
            continue
        if lib is None:
            from src.inference import engine
            lib = engine.library()
        network = load_network(load_assembly(path)["network"], lib)
        try:
            out[assembly_id] = {"network": network.id, "bbox": shakemap.bbox(network.rows("Node")),
                                "sites": sites(network)}
        except shakemap.ShakeMapError:
            continue                      # a network without coordinates: nothing to match an epicentre to
    return out


def sites(network) -> list[dict[str, float]]:
    """Where the ShakeMap is sampled: every bridge (the hazard function's sites), else every node."""
    rows = network.rows("Bridge") or network.rows("Node")
    return [{"lat": float(r["lat"]), "lon": float(r["lon"])} for r in rows if r.get("lat") is not None]


def regions(ids: list[str] | None = None) -> dict[str, list[float]]:
    """Each ShakeMap-capable network's box, for ``feedwatch`` (which has no registry)."""
    return {a["network"]: [round(v, 4) for v in a["bbox"]] for a in shakemap_assemblies(ids).values()}


# ---- one poll ------------------------------------------------------------------------


@contextmanager
def _lock(state_dir: Path) -> Iterator[None]:
    """One poll at a time. A lock older than ``LOCK_STALE_SECONDS`` is from a poll that
    died, and is taken over rather than blocking forever."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "watch.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if time.time() - path.stat().st_mtime < LOCK_STALE_SECONDS:
            raise PollInProgress(f"another poll holds {path}; skipping this one") from None
        path.unlink(missing_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, f"{os.getpid()} {_now()}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def run_session(assembly_id: str, event_id: str, version: int, samples: int) -> dict[str, Any]:
    """The page's Run on one ShakeMap version, headless: its result, or why it could not run."""
    from src.app.server import session_reply, start_session

    # The version the watcher just fetched, from the catalogue cache it wrote.
    scenario = {"parameters": {FUNCTION: {"event_id": event_id, "version": version, "source": "comcat"}}}
    try:
        s = start_session(assembly_id, scenario, samples)
    except ValueError as exc:
        # Usually "a site lies outside the ShakeMap": a small event's map that does not reach every
        # bridge. Recorded, not retried: the same map gives the same answer.
        return {"status": "not_computed", "reason": str(exc), "scenario": scenario}
    result = session_reply(s)["result"]
    fail = [e for e in result["events"] if e["node"].startswith("SystemState::")]
    closed = [e for e in result["events"] if e["node"].startswith("EdgeOpen::")]
    worst, top = max(fail, key=lambda e: e["p"]), max(closed, key=lambda e: e["p"])
    # E[cut off] alone: s.estimates() would also evaluate the system under do(open) for every edge.
    from src.inference.session import cut_off
    L = cut_off(s.t, {k: s.rows[f"SystemState::{k}"] for k in s.t["systems"]}, s.population)
    est, err = s._means(L[:, None])
    expected, se = est[0].item(), err[0].item()
    summary = {
        "most_likely_cut_off": {"origin": worst["node"].split("::")[1], "p": worst["p"]},
        "most_likely_closed": {"edge": top["node"].split("::")[1], "p": top["p"]},
        "expected_cut_off": {"value": expected, "se": se, "unit": s.state()["unit"]},
    }
    return {"status": "computed", "summary": summary, "scenario": s.scenario, "result": result}


def _describe(outcome: dict[str, Any]) -> str:
    if outcome["status"] != "computed":
        return f"not computed ({outcome['reason'][:100]})"
    s = outcome["summary"]
    return (f"E[cut off] = {s['expected_cut_off']['value']:.3g} {s['expected_cut_off']['unit']}, "
            f"most likely closed {s['most_likely_closed']['edge']} P = {s['most_likely_closed']['p']:.3f}")


def poll(
    *,
    feed: str = DEFAULT_FEED,
    state_dir: Path = DEFAULT_STATE_DIR,
    assemblies: dict[str, dict[str, Any]] | None = None,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    samples: int = DEFAULT_SAMPLES,
    source: shakemap.ComcatSource | None = None,
    fetch: Callable[[str], dict[str, Any]] = fetch_feed,
    run: Callable[[str, str, int, int], dict[str, Any]] = run_session,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """One poll: feed, match, fetch new versions, run, record. Returns the heartbeat.

    The heartbeat is written whatever happens. A failure to read the feed is re-raised after
    it is recorded; a failure on one event is recorded against that event and the poll
    carries on with the others.
    """
    state_dir = Path(state_dir)
    source = source or shakemap.ComcatSource()
    started = time.monotonic()
    heartbeat: dict[str, Any] = {"last_poll_utc": _now(), "feed": feed, "status": "ok",
                                 "events_in_feed": 0, "events_matched": 0, "runs": [], "errors": []}
    with _lock(state_dir):            # PollInProgress escapes here, before any heartbeat is written
        try:
            state = _read_json(state_dir / "state.json", {"events": {}})
            watched = shakemap_assemblies() if assemblies is None else assemblies
            heartbeat["assemblies"] = sorted(watched)
            features = fetch(feed).get("features", [])
            heartbeat["events_in_feed"] = len(features)

            for feature in features:
                props = feature.get("properties") or {}
                event_id = feature.get("id")
                lon, lat = (feature.get("geometry") or {}).get("coordinates", [None, None])[:2]
                if (not event_id or lon is None or "shakemap" not in (props.get("types") or "").split(",")
                        or (props.get("mag") or 0) < min_magnitude):
                    continue
                matched = sorted(a for a, w in watched.items() if _inside(w["bbox"], lon, lat))
                if not matched:
                    continue
                heartbeat["events_matched"] += 1
                record = state["events"].setdefault(event_id, {"runs": {}})
                record.update(magnitude=props.get("mag"), place=props.get("place"), lat=lat, lon=lon,
                              time=props.get("time"), networks=sorted({watched[a]["network"] for a in matched}))
                if record.get("updated") == props.get("updated") and all(
                        record["runs"].get(a, {}).get("version") == record.get("version") for a in matched):
                    continue                      # nothing new since the last poll

                try:
                    folder = source.directory(event_id)
                    version = int(folder.name.rsplit("-", 1)[1])
                except (shakemap.ShakeMapError, ValueError, IndexError) as exc:
                    heartbeat["errors"].append(f"{event_id}: {exc}")
                    record["error"] = str(exc)
                    continue
                record.update(updated=props.get("updated"), version=version, error=None)

                for assembly_id in matched:
                    if record["runs"].get(assembly_id, {}).get("version") == version:
                        continue
                    try:
                        outcome = run(assembly_id, event_id, version, samples)
                    except Exception as exc:      # noqa: BLE001 - one assembly's failure must not stop the poll
                        heartbeat["errors"].append(f"{event_id} {assembly_id}: {type(exc).__name__}: {exc}")
                        continue
                    out_path = (state_dir / "results" / event_id / f"{event_id}-{version}"
                                / f"{assembly_id.replace('/', '__')}.json")
                    _write_json(out_path, {"event_id": event_id, "version": version, "assembly": assembly_id,
                                           "magnitude": props.get("mag"), "place": props.get("place"),
                                           "computed_utc": _now(), **outcome})
                    # The short form is kept in the state too, so a list of recent events never
                    # has to open the full result files.
                    record["runs"][assembly_id] = {"version": version, "status": outcome["status"],
                                                   "file": str(out_path), "computed_utc": _now(),
                                                   "summary": outcome.get("summary"), "reason": outcome.get("reason")}
                    run_note = {"event_id": event_id, "version": version, "assembly": assembly_id,
                                "status": outcome["status"]}
                    if outcome["status"] == "computed":
                        run_note["summary"] = outcome["summary"]
                    heartbeat["runs"].append(run_note)
                    log(f"[watch] {event_id} v{version} M{props.get('mag')} {props.get('place')}: "
                        f"{assembly_id} {_describe(outcome)}")

            state["last_poll_utc"] = heartbeat["last_poll_utc"]
            _write_json(state_dir / "state.json", state)
        except WatchError as exc:
            heartbeat.update(status="error", errors=[*heartbeat["errors"], str(exc)])
            raise
        finally:
            if heartbeat["errors"] and heartbeat["status"] == "ok":
                heartbeat["status"] = "degraded"      # the feed worked, some events did not
            heartbeat["duration_s"] = round(time.monotonic() - started, 1)
            _write_json(state_dir / "heartbeat.json", heartbeat)
    return heartbeat


def run_forever(interval: float, **kwargs: Any) -> None:
    """Poll every ``interval`` seconds until interrupted; a failed poll does not stop it."""
    log = kwargs.get("log", print)
    while True:
        try:
            beat = poll(**kwargs)
            log(f"[watch] {beat['last_poll_utc']} {beat['status']}: {beat['events_matched']} matched, "
                f"{len(beat['runs'])} run(s)")
        except PollInProgress as exc:
            log(f"[watch] {exc}")
        except WatchError as exc:
            log(f"[watch] poll failed: {exc}")
        time.sleep(interval)


def check_heartbeat(state_dir: Path = DEFAULT_STATE_DIR, max_age_minutes: float = 10) -> tuple[bool, str]:
    """Whether the watcher is alive and well: (ok, one line saying why).

    Not ok when there is no heartbeat, when it is older than ``max_age_minutes`` (the watcher
    stopped), or when the last poll failed outright.
    """
    beat = _read_json(Path(state_dir) / "heartbeat.json", None)
    if beat is None:
        return False, f"no heartbeat in {state_dir}: the watcher has never run here"
    last = datetime.fromisoformat(beat["last_poll_utc"])
    age = (datetime.now(timezone.utc) - last).total_seconds() / 60
    if age > max_age_minutes:
        return False, f"last poll {age:.0f} min ago ({beat['last_poll_utc']}); the watcher looks stopped"
    if beat.get("status") == "error":
        return False, f"last poll failed: {'; '.join(beat.get('errors', []))[:300]}"
    note = f", {len(beat['errors'])} event error(s)" if beat.get("errors") else ""
    return True, f"ok: last poll {age:.1f} min ago, status {beat['status']}{note}"
