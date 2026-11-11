"""Tests for the ShakeMap watcher.

The feed and the catalogue are stand-ins; grids are written in the real format
by the ShakeMap tests' helper. The assemblies are the real ones, so a run here
is the same session the page's Run starts.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.builder import shakemap
from src.monitor import watch

from src.builder.tests.test_shakemap import write_grid

HEBEI = (115.0, 37.5)        # Xingtai area: inside Hebei's box, far from the other networks'
RUN = "hebei-transport/shakemap"
FILE = "hebei-transport__shakemap.json"


@pytest.fixture(autouse=True)
def fresh_shakemap_memo():
    shakemap._load_cached.cache_clear()
    shakemap._resolved.clear()
    yield
    shakemap._load_cached.cache_clear()
    shakemap._resolved.clear()


def feature(event_id, *, lonlat=HEBEI, mag=6.0, updated=1000, types=",origin,shakemap,"):
    return {"id": event_id, "geometry": {"coordinates": [*lonlat, 10.0]},
            "properties": {"mag": mag, "place": f"test {event_id}", "updated": updated, "types": types}}


class World:
    """A feed and a catalogue that tests can change between polls."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.features = []
        self.versions = {}                  # event id -> current ShakeMap version upstream
        self.detail_calls = []
        self.source = shakemap.ComcatSource(cache_dir=tmp_path / "cache")
        monkeypatch.setattr(self.source, "_get_json", self._detail)
        # Queries name source "comcat"; point that at this world's cache too.
        monkeypatch.setattr(shakemap, "DEFAULT_CACHE", tmp_path / "cache")
        monkeypatch.setattr(shakemap.ComcatSource.__init__, "__defaults__", (tmp_path / "cache", 60))

    def publish(self, event_id, version, **feature_kwargs):
        upstream = write_grid(self.tmp / "upstream" / f"{event_id}-{version}", event_id, version)
        self.versions[event_id] = (version, upstream)
        self.features = [f for f in self.features if f["id"] != event_id] + [feature(event_id, **feature_kwargs)]

    def _detail(self, params):
        self.detail_calls.append(params["eventid"])
        if params["eventid"] not in self.versions:     # what ComcatSource raises on the catalogue's 404
            raise shakemap.UnknownEvent(f"USGS has no event with id {params['eventid']!r}")
        version, upstream = self.versions[params["eventid"]]
        return {"properties": {"products": {"shakemap": [{
            "source": "us", "code": params["eventid"], "updateTime": 0, "properties": {"version": str(version)},
            "contents": {f"download/{n}": {"url": (upstream / n).as_uri()} for n in ("grid.xml", "uncertainty.xml")},
        }]}}}

    def poll(self, **kwargs):
        return watch.poll(state_dir=self.tmp / "watch", source=self.source,
                          fetch=lambda feed: {"features": self.features}, samples=1000, log=lambda m: None, **kwargs)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_a_new_shakemap_near_a_network_is_run_and_recorded(world):
    world.publish("ev1", 1)
    beat = world.poll()

    assert beat["status"] == "ok" and beat["events_matched"] == 1
    assert [(r["event_id"], r["version"], r["assembly"], r["status"]) for r in beat["runs"]] == [
        ("ev1", 1, RUN, "computed")]
    saved = json.loads((world.tmp / "watch" / "results" / "ev1" / "ev1-1" / FILE).read_text())
    assert saved["status"] == "computed"
    assert saved["scenario"]["parameters"]["hazard.shakemap"] == {"event_id": "ev1", "version": 1, "source": "comcat"}
    closed = {e["node"].split("::")[1]: e["p"] for e in saved["result"]["events"] if e["node"].startswith("EdgeOpen::")}
    top = saved["summary"]["most_likely_closed"]
    assert closed[top["edge"]] == top["p"] == max(closed.values())
    assert saved["summary"]["expected_cut_off"]["unit"] == "towns"
    assert any("ShakeMap ev1 version 1" in w for w in saved["result"]["warnings"])
    assert json.loads((world.tmp / "watch" / "heartbeat.json").read_text())["status"] == "ok"


def test_an_unchanged_event_is_not_looked_up_or_run_again(world):
    world.publish("ev1", 1)
    world.poll()
    beat = world.poll()
    assert beat["runs"] == [] and world.detail_calls == ["ev1"]


def test_an_update_without_a_new_shakemap_version_is_not_rerun(world):
    world.publish("ev1", 1, updated=1000)
    world.poll()
    world.features = [feature("ev1", updated=2000)]        # e.g. a new felt report, same map
    beat = world.poll()
    assert beat["runs"] == [] and world.detail_calls == ["ev1", "ev1"]


def test_a_revised_shakemap_is_fetched_and_run_again(world):
    world.publish("ev1", 1, updated=1000)
    world.poll()
    world.publish("ev1", 2, updated=2000)
    beat = world.poll()
    assert [(r["version"], r["status"]) for r in beat["runs"]] == [(2, "computed")]
    assert (world.tmp / "watch" / "results" / "ev1" / "ev1-1" / FILE).is_file()
    assert (world.tmp / "watch" / "results" / "ev1" / "ev1-2" / FILE).is_file()


@pytest.mark.parametrize("kwargs", [
    {"lonlat": (139.7, 35.7)},                      # Tokyo: near no network
    {"types": ",origin,dyfi,"},                     # no ShakeMap
    {"mag": 3.2},                                   # below the floor
])
def test_events_that_do_not_concern_any_network_are_ignored(world, kwargs):
    world.publish("ev1", 1, **kwargs)
    beat = world.poll()
    assert beat["events_matched"] == 0 and beat["runs"] == [] and world.detail_calls == []


def test_a_map_that_misses_part_of_the_network_runs_with_its_edge_values_noted(world):
    upstream = write_grid(world.tmp / "upstream" / "small-1", "small", 1, extent=(114.0, 36.0, 116.0, 38.0))
    world.versions["small"] = (1, upstream)
    world.features = [feature("small")]
    beat = world.poll()
    assert [r["status"] for r in beat["runs"]] == ["computed"]
    saved = json.loads((world.tmp / "watch" / "results" / "small" / "small-1" / FILE).read_text())
    assert any("off ShakeMap small" in w and "nearest edge" in w for w in saved["result"]["warnings"])


def test_a_map_that_reaches_no_bridge_is_recorded_not_retried(world):
    upstream = write_grid(world.tmp / "upstream" / "far-1", "far", 1, extent=(100.0, 20.0, 105.0, 25.0))
    world.versions["far"] = (1, upstream)
    world.features = [feature("far")]
    beat = world.poll()
    assert [r["status"] for r in beat["runs"]] == ["not_computed"]
    saved = json.loads((world.tmp / "watch" / "results" / "far" / "far-1" / FILE).read_text())
    assert "ShakeMap far v1 reaches none of the 33 sites" in saved["reason"]
    assert world.poll()["runs"] == []


def test_one_bad_event_does_not_stop_the_others(world):
    world.publish("good", 1)
    world.features.append(feature("gone"))           # in the feed, but the catalogue has no such event
    beat = world.poll()
    assert beat["status"] == "degraded"
    assert any(e.startswith("gone:") for e in beat["errors"])
    assert [r["event_id"] for r in beat["runs"]] == ["good"]


def test_a_feed_failure_is_raised_and_recorded_in_the_heartbeat(world):
    def broken(feed):
        raise watch.WatchError("could not read the USGS feed")

    with pytest.raises(watch.WatchError):
        watch.poll(state_dir=world.tmp / "watch", source=world.source, fetch=broken, log=lambda m: None)
    beat = json.loads((world.tmp / "watch" / "heartbeat.json").read_text())
    assert beat["status"] == "error" and "could not read the USGS feed" in beat["errors"][0]
    assert watch.check_heartbeat(world.tmp / "watch")[0] is False


def test_an_overlapping_poll_is_skipped_without_touching_the_heartbeat(world):
    world.publish("ev1", 1)
    world.poll()
    before = (world.tmp / "watch" / "heartbeat.json").read_text()
    (world.tmp / "watch" / "watch.lock").write_text("other poll")
    with pytest.raises(watch.PollInProgress):
        world.poll()
    assert (world.tmp / "watch" / "heartbeat.json").read_text() == before


def test_a_lock_left_by_a_dead_poll_is_taken_over(world):
    lock = world.tmp / "watch" / "watch.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("dead poll")
    old = time.time() - watch.LOCK_STALE_SECONDS - 60
    os.utime(lock, (old, old))
    assert world.poll()["status"] == "ok"
    assert not lock.exists()


# ---- the heartbeat check ------------------------------------------------------


def _beat(tmp_path, minutes_ago, status="ok", errors=()):
    when = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")
    (tmp_path / "heartbeat.json").write_text(json.dumps({"last_poll_utc": when, "status": status,
                                                          "errors": list(errors)}))


def test_heartbeat_check(tmp_path):
    assert watch.check_heartbeat(tmp_path)[0] is False                      # never ran
    _beat(tmp_path, 2)
    assert watch.check_heartbeat(tmp_path) == (True, "ok: last poll 2.0 min ago, status ok")
    _beat(tmp_path, 2, "degraded", ["gone: no such event"])
    ok, message = watch.check_heartbeat(tmp_path)
    assert ok and "1 event error(s)" in message                             # the watcher works
    _beat(tmp_path, 30)
    ok, message = watch.check_heartbeat(tmp_path, max_age_minutes=10)
    assert not ok and "looks stopped" in message
    _beat(tmp_path, 1, "error", ["could not read the USGS feed"])
    assert watch.check_heartbeat(tmp_path)[0] is False
