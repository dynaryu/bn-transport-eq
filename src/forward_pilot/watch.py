"""Watch the USGS feed for new ShakeMaps and run the forward query automatically.

The part of ShakeCast this pilot needs, and nothing more: poll the real-time
feed, keep events with a ShakeMap whose epicentre lies around one of the
packages, download each new ShakeMap version, and run each affected package's
forward query on it. No language model is involved: when a ShakeMap arrives
there is nothing to choose, the event is the scenario.

One poll:

1. Read the USGS real-time summary feed (``FEED_URL``). USGS asks that frequent
   polling use these feeds rather than the search API.
2. Keep events whose ``types`` include ``shakemap``, at or above the magnitude
   floor, with the epicentre inside a package's box (``shakemap.network_bbox``).
3. Skip an event whose ``updated`` time is unchanged since the last poll. A
   changed one is looked up; ShakeMaps are revised for hours as data arrives,
   and a new *version* is downloaded (``ComcatSource``, the same cache and folder
   layout as everywhere else) and run again.
4. For each package not yet run on that version: run the forward query and
   write the result to ``results/<event>/<event>-<version>/<package>.json``.

Every poll ends by writing ``heartbeat.json``, whether it succeeded or not, so
a watcher that has died or keeps failing is visible *before* an earthquake.
``check_heartbeat`` turns that file into an exit code for cron or an alarm.

Run it once per minute from cron, a systemd timer or a scheduled Lambda
(``--once``), or as a long-running loop (``--interval``). A lock file keeps two
polls from overlapping.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.request import Request, urlopen

from . import shakemap
from .engine import ForwardQueryError, run_forward_query
from .package import DEFAULT_OUTPUT, ROOT, load_package

FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson"
DEFAULT_FEED = "2.5_day"          # a day of history, so a watcher that was down a while catches up
DEFAULT_STATE_DIR = Path(
    os.environ.get("FORWARD_PILOT_WATCH_DIR") or ROOT / "experiments" / "forward-bn-pilot" / "watch"
)
DEFAULT_MIN_MAGNITUDE = 4.0       # below this no bridge is at risk, and New England fires M2.5s weekly
DEFAULT_SAMPLES = 5000
SEED = 20261004                   # the chat and the packages use the same seed
LOCK_STALE_SECONDS = 30 * 60


class WatchError(RuntimeError):
    """A poll could not complete."""


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


def fetch_feed(feed: str = DEFAULT_FEED, timeout: float = 30) -> dict[str, Any]:
    url = FEED_URL.format(feed=feed)
    try:
        with urlopen(Request(url, headers={"User-Agent": shakemap.USER_AGENT}), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise WatchError(f"could not read the USGS feed {url}: {exc}") from exc


def watched_packages(packages_root: Path) -> dict[str, dict[str, Any]]:
    """Packages that can take a ShakeMap, with the box events are matched against."""
    out = {}
    for folder in sorted(p for p in Path(packages_root).iterdir() if (p / "network.json").is_file()):
        network = load_package(folder)
        if "shakemap" not in network.get("hazard_models", ()):
            continue
        try:
            out[folder.name] = {"network": network, "bbox": shakemap.network_bbox(network)}
        except shakemap.ShakeMapError:
            continue                      # offers the model but has no coordinates: nothing to match
    return out


def _inside(bbox: tuple[float, float, float, float], lon: float, lat: float) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


@contextmanager
def _lock(state_dir: Path) -> Iterator[None]:
    """One poll at a time. A lock older than ``LOCK_STALE_SECONDS`` is from a
    poll that died, and is taken over rather than blocking forever."""
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


def _run_package(network: dict[str, Any], event_id: str, version: int, samples: int) -> dict[str, Any]:
    """The forward query on one ShakeMap version, or why it could not run."""
    query = {
        "query_type": "population_accessibility",
        "origins": list(network["default_origins"]),
        "destinations": list(network["destinations"]),
        "samples": samples,
        "seed": SEED,
        # The version the watcher just fetched, from the catalogue cache it wrote.
        "hazard": {"model": "shakemap", "event_id": event_id, "version": version, "source": "comcat"},
    }
    try:
        result = run_forward_query(network, query)
    except ForwardQueryError as exc:
        # Usually "bridges outside the ShakeMap": a small event's map that does not
        # reach the whole network. Recorded, not retried: the same map gives the same answer.
        return {"status": "not_computed", "reason": str(exc), "query": query}
    worst = result["origin_service"][0]
    top = result["components"][0]
    summary = {
        "lowest_service": {"origin": worst["origin"], "label": worst["label"], "probability": worst["probability"]},
        "highest_risk_bridge": {"edge": top["edge_id"], "p_unusable": top["p_unusable"], "sa1_g": top["sa1_g"]},
    }
    if "population_accessibility" in result:
        summary["expected_population_fraction"] = result["population_accessibility"]["expected_fraction"]
    return {"status": "computed", "summary": summary, "query": query, "result": result}


def poll(
    *,
    feed: str = DEFAULT_FEED,
    state_dir: Path = DEFAULT_STATE_DIR,
    packages_root: Path = DEFAULT_OUTPUT,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    samples: int = DEFAULT_SAMPLES,
    source: shakemap.ComcatSource | None = None,
    fetch: Callable[[str], dict[str, Any]] = fetch_feed,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """One poll: feed, match, fetch new versions, run, record. Returns the heartbeat.

    The heartbeat is written whatever happens. A failure to read the feed is
    re-raised after it is recorded; a failure on one event is recorded against
    that event and the poll carries on with the others.
    """
    state_dir = Path(state_dir)
    source = source or shakemap.ComcatSource()
    started = time.monotonic()
    heartbeat: dict[str, Any] = {"last_poll_utc": _now(), "feed": feed, "status": "ok",
                                 "events_in_feed": 0, "events_matched": 0, "runs": [], "errors": []}
    with _lock(state_dir):            # PollInProgress escapes here, before any heartbeat is written
        try:
            state = _read_json(state_dir / "state.json", {"events": {}})
            packages = watched_packages(packages_root)
            heartbeat["packages"] = sorted(packages)
            features = fetch(feed).get("features", [])
            heartbeat["events_in_feed"] = len(features)

            for feature in features:
                props = feature.get("properties") or {}
                event_id = feature.get("id")
                lon, lat = (feature.get("geometry") or {}).get("coordinates", [None, None])[:2]
                if (not event_id or lon is None or "shakemap" not in (props.get("types") or "").split(",")
                        or (props.get("mag") or 0) < min_magnitude):
                    continue
                matched = [name for name, p in packages.items() if _inside(p["bbox"], lon, lat)]
                if not matched:
                    continue
                heartbeat["events_matched"] += 1
                record = state["events"].setdefault(event_id, {"runs": {}})
                record.update(magnitude=props.get("mag"), place=props.get("place"), lat=lat, lon=lon,
                              time=props.get("time"))
                if record.get("updated") == props.get("updated") and all(
                        record["runs"].get(n, {}).get("version") == record.get("version") for n in matched):
                    continue                      # nothing new since the last poll

                try:
                    folder = source.directory(event_id)
                    version = int(folder.name.rsplit("-", 1)[1])
                except (shakemap.ShakeMapError, ValueError, IndexError) as exc:
                    heartbeat["errors"].append(f"{event_id}: {exc}")
                    record["error"] = str(exc)
                    continue
                record.update(updated=props.get("updated"), version=version, error=None)

                for name in matched:
                    if record["runs"].get(name, {}).get("version") == version:
                        continue
                    outcome = _run_package(packages[name]["network"], event_id, version, samples)
                    out_path = state_dir / "results" / event_id / f"{event_id}-{version}" / f"{name}.json"
                    _write_json(out_path, {"event_id": event_id, "version": version, "package": name,
                                           "magnitude": props.get("mag"), "place": props.get("place"),
                                           "computed_utc": _now(), **outcome})
                    # The short form is kept in the state too, so a list of recent events
                    # never has to open the full result files.
                    record["runs"][name] = {"version": version, "status": outcome["status"], "file": str(out_path),
                                            "computed_utc": _now(), "summary": outcome.get("summary"),
                                            "reason": outcome.get("reason")}
                    run = {"event_id": event_id, "version": version, "package": name, "status": outcome["status"]}
                    if outcome["status"] == "computed":
                        run["summary"] = outcome["summary"]
                    heartbeat["runs"].append(run)
                    log(f"[watch] {event_id} v{version} M{props.get('mag')} {props.get('place')}: "
                        f"{name} {outcome['status']}"
                        + (f", highest-risk bridge {outcome['summary']['highest_risk_bridge']['edge']} "
                           f"P(unusable)={outcome['summary']['highest_risk_bridge']['p_unusable']:.3f}"
                           if outcome["status"] == "computed" else f" ({outcome['reason'][:80]})"))

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


def recent_results(state_dir: Path = DEFAULT_STATE_DIR, limit: int = 20) -> list[dict[str, Any]]:
    """The watcher's runs, one row per event and package, newest event first."""
    state = _read_json(Path(state_dir) / "state.json", {"events": {}})
    rows = []
    for event_id, record in state.get("events", {}).items():
        for package, run in record.get("runs", {}).items():
            rows.append({
                "event_id": event_id, "package": package, "version": run.get("version"),
                "status": run.get("status"), "summary": run.get("summary"), "reason": run.get("reason"),
                "computed_utc": run.get("computed_utc"), "magnitude": record.get("magnitude"),
                "place": record.get("place"),
                "time": (datetime.fromtimestamp(record["time"] / 1000, timezone.utc).isoformat(timespec="seconds")
                         if record.get("time") else None),
            })
    rows.sort(key=lambda r: (r["time"] or r["computed_utc"] or ""), reverse=True)
    return rows[:limit]


def check_heartbeat(state_dir: Path = DEFAULT_STATE_DIR, max_age_minutes: float = 10) -> tuple[bool, str]:
    """Whether the watcher is alive and well: (ok, one line saying why).

    Not ok when there is no heartbeat, when it is older than ``max_age_minutes``
    (the watcher stopped), or when the last poll failed outright.
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
