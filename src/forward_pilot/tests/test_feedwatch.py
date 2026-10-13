"""Tests for the feed-only ShakeMap watcher (the one deployed as a Lambda).

The feed, the catalogue and S3 are stand-ins; grids are written in the real
format by the ShakeMap tests' helper.
"""

import ast
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from src.forward_pilot import feedwatch, shakemap, watch
from src.forward_pilot.tests.test_shakemap import write_grid
from src.forward_pilot.tests.test_watch import feature

REGIONS = {"hebei-transport": [112.0, 35.0, 121.0, 43.0], "ema-highway": [-74.0, 40.5, -69.0, 44.0]}


class World:
    """A feed and a catalogue that tests can change between polls."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.features = []
        self.versions = {}
        self.detail_calls = []
        self.store = feedwatch.DirStore({"shakemaps": tmp_path / "cache", "watch": tmp_path / "watch"})

    def publish(self, event_id, version, **feature_kwargs):
        upstream = write_grid(self.tmp / "upstream" / f"{event_id}-{version}", event_id, version)
        self.versions[event_id] = (version, upstream)
        self.features = [f for f in self.features if f["id"] != event_id] + [feature(event_id, **feature_kwargs)]

    def detail(self, event_id):
        self.detail_calls.append(event_id)
        if event_id not in self.versions:
            raise feedwatch.WatchError("the catalogue has no such event")
        version, upstream = self.versions[event_id]
        return {"properties": {"products": {"shakemap": [{
            "source": "us", "code": event_id, "updateTime": 0, "properties": {"version": str(version)},
            "contents": {f"download/{n}": {"url": str(upstream / n)} for n in ("grid.xml", "uncertainty.xml")},
        }]}}}

    def poll(self, store=None, **kwargs):
        return feedwatch.poll(store or self.store, REGIONS, fetch=lambda feed: {"features": self.features},
                              detail=self.detail, opener=lambda url: open(url, "rb"), log=lambda m: None, **kwargs)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_a_new_shakemap_is_archived_where_the_pilot_reads_it(world):
    world.publish("ev1", 1)
    beat = world.poll()
    assert beat["status"] == "ok" and beat["fetched"] == [{"event_id": "ev1", "version": 1,
                                                            "packages": ["hebei-transport"]}]
    folder = world.tmp / "cache" / "ev1" / "ev1-1"
    assert sorted(p.name for p in folder.iterdir()) == ["grid.xml", "info.json", "uncertainty.xml"]

    # The catalogue source finds it without going to the network.
    source = shakemap.ComcatSource(cache_dir=world.tmp / "cache")
    assert source.directory("ev1", version=1) == folder
    assert shakemap.read_grid(folder).std_psa10 is not None


def test_an_unchanged_event_is_not_looked_up_again(world):
    world.publish("ev1", 1)
    world.poll()
    assert world.poll()["fetched"] == [] and world.detail_calls == ["ev1"]


def test_a_revised_shakemap_is_kept_beside_the_old_one(world):
    world.publish("ev1", 1, updated=1000)
    world.poll()
    world.features = [feature("ev1", updated=1500)]          # updated, same map: looked up, nothing saved
    assert world.poll()["fetched"] == []
    world.publish("ev1", 2, updated=2000)
    assert [f["version"] for f in world.poll()["fetched"]] == [2]
    assert (world.tmp / "cache" / "ev1" / "ev1-1" / "grid.xml").is_file()
    assert (world.tmp / "cache" / "ev1" / "ev1-2" / "grid.xml").is_file()
    state = json.loads((world.tmp / "watch" / "state.json").read_text())
    assert state["events"]["ev1"]["versions"] == [1, 2]


@pytest.mark.parametrize("kwargs", [
    {"lonlat": (139.7, 35.7)},                      # Tokyo: near no package
    {"types": ",origin,dyfi,"},                     # no ShakeMap
    {"mag": 3.2},                                   # below the floor
])
def test_events_that_do_not_concern_any_package_are_ignored(world, kwargs):
    world.publish("ev1", 1, **kwargs)
    beat = world.poll()
    assert beat["events_matched"] == 0 and beat["fetched"] == [] and world.detail_calls == []


def test_a_failed_event_is_recorded_and_retried(world):
    world.publish("good", 1)
    world.features.append(feature("gone"))
    beat = world.poll()
    assert beat["status"] == "degraded" and beat["errors"][0].startswith("gone:")
    assert [f["event_id"] for f in beat["fetched"]] == ["good"]
    world.poll()
    assert world.detail_calls.count("gone") == 2              # an error is not taken as "seen"


def test_a_feed_failure_is_raised_and_recorded_in_the_heartbeat(world):
    def broken(feed):
        raise feedwatch.WatchError("could not read the USGS feed")

    with pytest.raises(feedwatch.WatchError):
        feedwatch.poll(world.store, REGIONS, fetch=broken, log=lambda m: None)
    assert watch.check_heartbeat(world.tmp / "watch")[0] is False


def test_the_page_lists_archived_events_ready_to_run(world):
    world.publish("ev1", 1)
    world.features.append(feature("gone", lonlat=(-71.0, 42.4)))
    world.poll()
    rows = {(r["event_id"], r["package"]): r for r in watch.recent_results(world.tmp / "watch")}
    assert rows["ev1", "hebei-transport"]["status"] == "fetched" and rows["ev1", "hebei-transport"]["version"] == 1
    assert rows["gone", "ema-highway"]["status"] == "error" and rows["gone", "ema-highway"]["reason"]
    assert watch.check_heartbeat(world.tmp / "watch")[0] is True


# ---- S3 and the Lambda entry point -----------------------------------------------


class FakeS3:
    """The few S3 calls the store makes, kept in a dict."""

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objects = {}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys):
        return {"Contents": [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)][:MaxKeys]}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def upload_fileobj(self, stream, Bucket, Key):
        self.objects[Key] = stream.read()


class FakeCloudWatch:
    def __init__(self):
        self.values = []

    def put_metric_data(self, Namespace, MetricData):
        self.values += [(Namespace, d["MetricName"], d["Value"]) for d in MetricData]


def test_the_lambda_archives_to_s3_and_reports_health(world, monkeypatch, tmp_path):
    (tmp_path / "regions.json").write_text(json.dumps(REGIONS))
    monkeypatch.setenv("FEEDWATCH_REGIONS", str(tmp_path / "regions.json"))
    s3, cloudwatch = FakeS3(), FakeCloudWatch()
    store = feedwatch.S3Store("bucket", "pilot", client=s3)
    world.publish("ev1", 1)
    common = dict(store=store, cloudwatch=cloudwatch, fetch=lambda feed: {"features": world.features},
                  detail=world.detail, opener=lambda url: open(url, "rb"), log=lambda m: None)

    out = feedwatch.lambda_handler({}, None, **common)
    assert out["status"] == "ok" and out["fetched"][0]["event_id"] == "ev1"
    assert {"pilot/shakemaps/ev1/ev1-1/grid.xml", "pilot/shakemaps/ev1/ev1-1/uncertainty.xml",
            "pilot/shakemaps/ev1/ev1-1/info.json", "pilot/watch/state.json",
            "pilot/watch/heartbeat.json"} <= set(s3.objects)
    assert feedwatch.lambda_handler({}, None, **common)["fetched"] == []      # the S3 copy counts as kept
    assert cloudwatch.values == [("BNPilot", "WatcherHealthy", 1.0)] * 2

    common["fetch"] = lambda feed: (_ for _ in ()).throw(feedwatch.WatchError("feed down"))
    with pytest.raises(feedwatch.WatchError):
        feedwatch.lambda_handler({}, None, **common)
    assert cloudwatch.values[-1] == ("BNPilot", "WatcherHealthy", 0.0)


def test_the_module_imports_only_the_standard_library():
    """The Lambda zip holds this one file; an import of numpy or of the pilot breaks it."""
    tree = ast.parse(Path(feedwatch.__file__).read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
    assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree))
    assert names - set(sys.stdlib_module_names) == {"boto3"}


def test_the_lambda_zip_carries_the_real_package_regions(tmp_path):
    from src.forward_pilot.package import DEFAULT_OUTPUT

    regions = watch.regions(DEFAULT_OUTPUT)
    assert set(regions) == {"ema-highway", "hebei-transport"}
    lon_min, lat_min, lon_max, lat_max = regions["hebei-transport"]
    assert lon_min < 115.0 < lon_max and lat_min < 37.5 < lat_max      # Xingtai
    lon_min, lat_min, lon_max, lat_max = regions["ema-highway"]
    assert lon_min < -71.06 < lon_max and lat_min < 42.36 < lat_max    # Boston
