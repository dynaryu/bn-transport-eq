"""Tests for the ShakeMap watcher.

The feed and the catalogue are stand-ins; grids are written in the real format
by the ShakeMap tests' helper. The packages are the real built ones, so a run
here is the same forward query the UI makes.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.forward_pilot import shakemap, watch
from src.forward_pilot.tests.test_shakemap import write_grid

HEBEI = (115.0, 37.5)        # Xingtai area: inside Hebei's box, far from EMA's


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
                          fetch=lambda feed: {"features": self.features}, samples=200, log=lambda m: None, **kwargs)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_a_new_shakemap_near_a_package_is_run_and_recorded(world):
    world.publish("ev1", 1)
    beat = world.poll()

    assert beat["status"] == "ok" and beat["events_matched"] == 1
    assert [(r["event_id"], r["version"], r["package"], r["status"]) for r in beat["runs"]] == [
        ("ev1", 1, "hebei-transport", "computed")]
    saved = json.loads((world.tmp / "watch" / "results" / "ev1" / "ev1-1" / "hebei-transport.json").read_text())
    assert saved["status"] == "computed"
    assert saved["query"]["hazard"] == {"model": "shakemap", "event_id": "ev1", "version": 1, "source": "comcat"}
    assert saved["summary"]["highest_risk_bridge"]["edge"] == saved["result"]["components"][0]["edge_id"]
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
    assert (world.tmp / "watch" / "results" / "ev1" / "ev1-1" / "hebei-transport.json").is_file()
    assert (world.tmp / "watch" / "results" / "ev1" / "ev1-2" / "hebei-transport.json").is_file()


@pytest.mark.parametrize("kwargs", [
    {"lonlat": (139.7, 35.7)},                      # Tokyo: near no package
    {"types": ",origin,dyfi,"},                     # no ShakeMap
    {"mag": 3.2},                                   # below the floor
])
def test_events_that_do_not_concern_any_package_are_ignored(world, kwargs):
    world.publish("ev1", 1, **kwargs)
    beat = world.poll()
    assert beat["events_matched"] == 0 and beat["runs"] == [] and world.detail_calls == []


def test_a_map_that_misses_part_of_the_network_is_recorded_not_retried(world):
    upstream = write_grid(world.tmp / "upstream" / "small-1", "small", 1, extent=(114.0, 36.0, 116.0, 38.0))
    world.versions["small"] = (1, upstream)
    world.features = [feature("small")]
    beat = world.poll()
    assert [r["status"] for r in beat["runs"]] == ["not_computed"]
    saved = json.loads((world.tmp / "watch" / "results" / "small" / "small-1" / "hebei-transport.json").read_text())
    assert "outside the ShakeMap" in saved["reason"]
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


# ---- the recent-events list the page shows ---------------------------------------


def test_recent_results_lists_runs_newest_event_first(world):
    world.publish("old", 1)
    world.features[-1]["properties"]["time"] = 1_000_000
    world.publish("new", 1)
    world.features[-1]["properties"]["time"] = 2_000_000
    world.poll()
    rows = watch.recent_results(world.tmp / "watch")
    assert [r["event_id"] for r in rows] == ["new", "old"]
    assert rows[0]["package"] == "hebei-transport" and rows[0]["status"] == "computed"
    assert rows[0]["summary"]["highest_risk_bridge"]["edge"].startswith("e")
    assert rows[0]["time"].startswith("1970-01-01T00:33")       # 2,000,000 ms after the epoch


def test_recent_results_carry_the_reason_a_run_was_not_computed(world):
    upstream = write_grid(world.tmp / "upstream" / "small-1", "small", 1, extent=(114.0, 36.0, 116.0, 38.0))
    world.versions["small"] = (1, upstream)
    world.features = [feature("small")]
    world.poll()
    (row,) = watch.recent_results(world.tmp / "watch")
    assert row["status"] == "not_computed" and "outside the ShakeMap" in row["reason"]


def test_the_page_endpoint_reports_the_watcher_and_its_events(world, monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen

    from src.forward_pilot.server import PilotHandler

    monkeypatch.setattr(watch, "DEFAULT_STATE_DIR", world.tmp / "watch")
    server = ThreadingHTTPServer(("127.0.0.1", 0), PilotHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/api/watch-events"
    try:
        empty = json.loads(urlopen(url, timeout=10).read())
        assert empty == {"heartbeat": None, "events": []}         # never ran: the page hides the card
        world.publish("ev1", 1)
        world.poll()
        data = json.loads(urlopen(url, timeout=10).read())
    finally:
        server.shutdown()
        server.server_close()
    assert data["heartbeat"]["ok"] is True
    assert [(e["event_id"], e["package"]) for e in data["events"]] == [("ev1", "hebei-transport")]
