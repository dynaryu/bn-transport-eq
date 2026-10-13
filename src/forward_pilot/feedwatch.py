"""Archive new ShakeMaps near the packages: the feed-only watcher.

The always-on part of the pilot, and nothing else. It polls the USGS real-time
feed, keeps events with a ShakeMap whose epicentre lies in one of the packages'
regions, and saves every new ShakeMap *version* of them. It does not run the
BN: that happens on the machine with the UI, on demand, from the archived grid.

What it adds over asking ComCat later: every version is kept as it is issued
(the catalogue serves only the current one), and a heartbeat shows whether the
watcher is alive *before* an earthquake.

Storage is keyed like the pilot's own folders, so one ``aws s3 sync`` of the
bucket into ``experiments/forward-bn-pilot/`` puts everything where it is read:

* ``shakemaps/<id>/<id>-<version>/{grid.xml,uncertainty.xml,info.json}`` -- the
  ComCat cache layout (also ShakeCast's), so ``ComcatSource`` uses it unchanged;
* ``watch/state.json`` and ``watch/heartbeat.json`` -- what the page's Recent
  earthquakes panel and ``shakemap heartbeat`` read.

``regions.json`` holds each package's box, written by ``shakemap lambda-zip``
from the packages, so this module needs neither the packages nor numpy.

Standard library only, plus boto3 when the store is S3 -- the AWS Lambda Python
runtime provides it. The zip is this file and ``regions.json``; the handler is
``feedwatch.lambda_handler``. Keep this module free of imports from the rest of
the pilot, or the Lambda will not start.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson"
DETAIL_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USER_AGENT = "bn-on-demand-pilot (forward BN pilot; ShakeMap feed watcher)"
DEFAULT_FEED = "2.5_day"          # a day of history, so a watcher that was down a while catches up
DEFAULT_MIN_MAGNITUDE = 4.0       # below this no bridge is at risk, and New England fires M2.5s weekly
FILES = ("uncertainty.xml", "grid.xml")   # grid.xml last: its presence marks a complete version
STATE_KEY = "watch/state.json"
HEARTBEAT_KEY = "watch/heartbeat.json"
METRIC_NAMESPACE = "BNPilot"


class WatchError(RuntimeError):
    """A poll could not complete."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get(url: str, timeout: float = 30):
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


def fetch_feed(feed: str = DEFAULT_FEED) -> dict[str, Any]:
    url = FEED_URL.format(feed=feed)
    try:
        with _get(url) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise WatchError(f"could not read the USGS feed {url}: {exc}") from exc


def fetch_detail(event_id: str) -> dict[str, Any]:
    """The event's catalogue entry, with its products and their download URLs."""
    with _get(f"{DETAIL_URL}?{urlencode({'eventid': event_id, 'format': 'geojson'})}") as response:
        return json.loads(response.read().decode("utf-8"))


def open_url(url: str) -> BinaryIO:
    return _get(url, timeout=120)


# ---- storage ---------------------------------------------------------------------


class DirStore:
    """Keys as files. Each top-level key folder maps to a directory, so the local
    layout can follow the pilot's settings (the ShakeMap cache may live elsewhere)."""

    def __init__(self, roots: dict[str, Path | str]):
        self.roots = {name: Path(path) for name, path in roots.items()}

    def _path(self, key: str) -> Path:
        head, _, rest = key.partition("/")
        return self.roots[head] / rest

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def read_json(self, key: str, default: Any = None) -> Any:
        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default

    def write_json(self, key: str, value: Any) -> None:
        self.put(key, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        partial.write_bytes(data)
        partial.replace(path)            # a reader never sees half a file

    def put_stream(self, key: str, stream: BinaryIO) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with partial.open("wb") as out:
                while chunk := stream.read(1 << 20):
                    out.write(chunk)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(path)


class S3Store:
    """Keys as objects under ``prefix`` in one bucket. An S3 write is atomic, and a
    grid is streamed up in parts, so a Lambda never holds a whole one in memory."""

    def __init__(self, bucket: str, prefix: str = "", client: Any = None):
        if client is None:
            import boto3                 # in the Lambda runtime; locally only when S3 is used

            client = boto3.client("s3")
        self.bucket, self.prefix, self.client = bucket, prefix.strip("/"), client

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def exists(self, key: str) -> bool:
        listing = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self._key(key), MaxKeys=1)
        return any(o["Key"] == self._key(key) for o in listing.get("Contents", ()))

    def read_json(self, key: str, default: Any = None) -> Any:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return default
        return json.loads(body.decode("utf-8"))

    def write_json(self, key: str, value: Any) -> None:
        self.put(key, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def put_stream(self, key: str, stream: BinaryIO) -> None:
        self.client.upload_fileobj(stream, self.bucket, self._key(key))


# ---- one poll --------------------------------------------------------------------


def _inside(bbox, lon: float, lat: float) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _archive(store, event_id: str, detail: dict[str, Any], opener: Callable[[str], BinaryIO]) -> tuple[int, bool]:
    """Save the event's preferred ShakeMap unless that version is already kept.
    Returns (version, whether it was new)."""
    products = detail.get("properties", {}).get("products", {}).get("shakemap") or []
    if not products:
        raise WatchError("the catalogue lists no ShakeMap for it")
    product = products[0]                # ComCat lists the preferred product first
    version = int(float(product.get("properties", {}).get("version", 1)))
    folder = f"shakemaps/{event_id}/{event_id}-{version}"
    if store.exists(f"{folder}/grid.xml"):
        return version, False
    contents = product.get("contents", {})
    if "download/grid.xml" not in contents:
        raise WatchError(f"ShakeMap version {version} has no grid.xml")
    for name in FILES:
        entry = contents.get(f"download/{name}")
        if entry:
            with opener(entry["url"]) as stream:
                store.put_stream(f"{folder}/{name}", stream)
    store.write_json(f"{folder}/info.json", {
        "event_id": event_id, "version": version, "source": product.get("source"),
        "code": product.get("code"), "update_time": product.get("updateTime"), "retrieved_utc": _now(),
    })
    return version, True


def poll(
    store,
    regions: dict[str, list[float]],
    *,
    feed: str = DEFAULT_FEED,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    fetch: Callable[[str], dict[str, Any]] = fetch_feed,
    detail: Callable[[str], dict[str, Any]] = fetch_detail,
    opener: Callable[[str], BinaryIO] = open_url,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """One poll: feed, match, archive new versions, record. Returns the heartbeat.

    The heartbeat is written whatever happens. A failure to read the feed is
    re-raised after it is recorded; a failure on one event is recorded against
    that event and the poll carries on with the others.
    """
    started = time.monotonic()
    heartbeat: dict[str, Any] = {"last_poll_utc": _now(), "feed": feed, "status": "ok", "mode": "fetch-only",
                                 "packages": sorted(regions), "events_in_feed": 0, "events_matched": 0,
                                 "fetched": [], "errors": []}
    try:
        state = store.read_json(STATE_KEY, {"events": {}})
        features = fetch(feed).get("features", [])
        heartbeat["events_in_feed"] = len(features)

        for feature in features:
            props = feature.get("properties") or {}
            event_id = feature.get("id")
            lon, lat = (feature.get("geometry") or {}).get("coordinates", [None, None])[:2]
            if (not event_id or lon is None or "shakemap" not in (props.get("types") or "").split(",")
                    or (props.get("mag") or 0) < min_magnitude):
                continue
            matched = sorted(name for name, bbox in regions.items() if _inside(bbox, lon, lat))
            if not matched:
                continue
            heartbeat["events_matched"] += 1
            record = state["events"].setdefault(event_id, {})
            record.update(magnitude=props.get("mag"), place=props.get("place"), lat=lat, lon=lon,
                          time=props.get("time"), packages=matched)
            if record.get("updated") == props.get("updated") and not record.get("error"):
                continue                          # nothing new since the last poll

            try:
                version, new = _archive(store, event_id, detail(event_id), opener)
            except (OSError, ValueError, WatchError) as exc:
                heartbeat["errors"].append(f"{event_id}: {exc}")
                record["error"] = str(exc)
                continue
            versions = sorted({*record.get("versions", []), version})
            record.update(updated=props.get("updated"), version=version, versions=versions, error=None)
            if new:
                heartbeat["fetched"].append({"event_id": event_id, "version": version, "packages": matched})
                log(f"[feedwatch] {event_id} v{version} M{props.get('mag')} {props.get('place')}: "
                    f"archived for {', '.join(matched)}")

        state["last_poll_utc"] = heartbeat["last_poll_utc"]
        store.write_json(STATE_KEY, state)
    except WatchError as exc:
        heartbeat.update(status="error", errors=[*heartbeat["errors"], str(exc)])
        raise
    finally:
        if heartbeat["errors"] and heartbeat["status"] == "ok":
            heartbeat["status"] = "degraded"      # the feed worked, some events did not
        heartbeat["duration_s"] = round(time.monotonic() - started, 1)
        store.write_json(HEARTBEAT_KEY, heartbeat)
    return heartbeat


# ---- AWS Lambda ------------------------------------------------------------------


def load_regions(path: Path | str | None = None) -> dict[str, list[float]]:
    path = Path(path or os.environ.get("FEEDWATCH_REGIONS") or Path(__file__).with_name("regions.json"))
    regions = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in regions.items() if not k.startswith("_")}


def _put_health(healthy: bool, cloudwatch: Any = None) -> None:
    """1 per good poll, 0 per failed one. The alarm treats a missing value as a
    failure too, so a schedule that stops firing is caught as well."""
    if cloudwatch is None:
        import boto3

        cloudwatch = boto3.client("cloudwatch")
    cloudwatch.put_metric_data(Namespace=os.environ.get("FEEDWATCH_METRIC_NAMESPACE", METRIC_NAMESPACE),
                               MetricData=[{"MetricName": "WatcherHealthy", "Value": 1.0 if healthy else 0.0}])


def lambda_handler(event: Any = None, context: Any = None, *, store: Any = None, cloudwatch: Any = None,
                   **poll_kwargs: Any) -> dict[str, Any]:
    """Entry point for a scheduled Lambda. Configured by environment:
    ``FEEDWATCH_BUCKET`` (required), ``FEEDWATCH_PREFIX``, ``FEEDWATCH_FEED``,
    ``FEEDWATCH_MIN_MAGNITUDE``. Run it with reserved concurrency 1, so polls
    never overlap."""
    if store is None:
        store = S3Store(os.environ["FEEDWATCH_BUCKET"], os.environ.get("FEEDWATCH_PREFIX", ""))
    poll_kwargs.setdefault("feed", os.environ.get("FEEDWATCH_FEED", DEFAULT_FEED))
    poll_kwargs.setdefault("min_magnitude", float(os.environ.get("FEEDWATCH_MIN_MAGNITUDE", DEFAULT_MIN_MAGNITUDE)))
    try:
        beat = poll(store, load_regions(), **poll_kwargs)
    except WatchError:
        _put_health(False, cloudwatch)
        raise                             # also counts in the function's own Errors metric
    _put_health(True, cloudwatch)
    return {k: beat[k] for k in ("status", "events_in_feed", "events_matched", "fetched", "errors")}
