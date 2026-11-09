"""Tests for the feed-only ShakeMap watcher (the one deployed as a Lambda).

The feed, the catalogue and S3 are stand-ins.
"""

import ast
import io
import json
import sys
from pathlib import Path

import pytest

from src.monitor import feedwatch, watch
from src.monitor.tests.test_watch import feature

REGIONS = {"hebei-transport": [112.0, 35.0, 121.0, 43.0], "ema-highway": [-74.0, 40.5, -69.0, 44.0]}


class World:
    """A feed and a catalogue that tests can change between polls."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.features = []
        self.versions = {}
        self.detail_calls = []
        self.store = feedwatch.DirStore({"watch": tmp_path / "watch"})

    def publish(self, event_id, version, **feature_kwargs):
        self.versions[event_id] = version
        self.features = [f for f in self.features if f["id"] != event_id] + [feature(event_id, **feature_kwargs)]

    def detail(self, event_id):
        self.detail_calls.append(event_id)
        if event_id not in self.versions:
            raise feedwatch.WatchError("the catalogue has no such event")
        return {"properties": {"products": {"shakemap": [{
            "source": "us", "code": event_id, "properties": {"version": str(self.versions[event_id])}}]}}}

    def poll(self, store=None, **kwargs):
        return feedwatch.poll(store or self.store, REGIONS, fetch=lambda feed: {"features": self.features},
                              detail=self.detail, log=lambda m: None, **kwargs)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_a_new_shakemap_is_noted_without_downloading_it(world):
    world.publish("ev1", 1)
    beat = world.poll()
    assert beat["status"] == "ok" and beat["updates"] == [{"event_id": "ev1", "version": 1,
                                                            "networks": ["hebei-transport"]}]
    state = json.loads((world.tmp / "watch" / "state.json").read_text())
    assert state["events"]["ev1"]["versions"] == [1]
    assert sorted(p.name for p in world.tmp.rglob("*") if p.is_file()) == ["heartbeat.json", "state.json"]


def test_an_unchanged_event_is_not_looked_up_again(world):
    world.publish("ev1", 1)
    world.poll()
    assert world.poll()["updates"] == [] and world.detail_calls == ["ev1"]


def test_each_revised_shakemap_version_is_recorded(world):
    world.publish("ev1", 1, updated=1000)
    world.poll()
    world.features = [feature("ev1", updated=1500)]          # updated, same map: looked up, nothing new
    assert world.poll()["updates"] == []
    world.publish("ev1", 2, updated=2000)
    assert [u["version"] for u in world.poll()["updates"]] == [2]
    state = json.loads((world.tmp / "watch" / "state.json").read_text())
    assert state["events"]["ev1"]["versions"] == [1, 2] and state["events"]["ev1"]["version"] == 2


@pytest.mark.parametrize("kwargs", [
    {"lonlat": (139.7, 35.7)},                      # Tokyo: near no network
    {"types": ",origin,dyfi,"},                     # no ShakeMap
    {"mag": 3.2},                                   # below the floor
])
def test_events_that_do_not_concern_any_network_are_ignored(world, kwargs):
    world.publish("ev1", 1, **kwargs)
    beat = world.poll()
    assert beat["events_matched"] == 0 and beat["updates"] == [] and world.detail_calls == []


def test_a_failed_event_is_recorded_and_retried(world):
    world.publish("good", 1)
    world.features.append(feature("gone"))
    beat = world.poll()
    assert beat["status"] == "degraded" and beat["errors"][0].startswith("gone:")
    assert [u["event_id"] for u in beat["updates"]] == ["good"]
    world.poll()
    assert world.detail_calls.count("gone") == 2              # an error is not taken as "seen"


def test_a_feed_failure_is_raised_and_recorded_in_the_heartbeat(world):
    def broken(feed):
        raise feedwatch.WatchError("could not read the USGS feed")

    with pytest.raises(feedwatch.WatchError):
        feedwatch.poll(world.store, REGIONS, fetch=broken, log=lambda m: None)
    assert watch.check_heartbeat(world.tmp / "watch")[0] is False


# ---- S3 and the Lambda entry point -----------------------------------------------


class FakeS3:
    """The few S3 calls the store makes, kept in a dict."""

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body


class FakeCloudWatch:
    def __init__(self):
        self.values = []

    def put_metric_data(self, Namespace, MetricData):
        self.values += [(Namespace, d["MetricName"], d["Value"]) for d in MetricData]


def test_the_lambda_keeps_its_state_in_s3_and_reports_health(world, monkeypatch, tmp_path):
    (tmp_path / "regions.json").write_text(json.dumps(REGIONS))
    monkeypatch.setenv("FEEDWATCH_REGIONS", str(tmp_path / "regions.json"))
    s3, cloudwatch = FakeS3(), FakeCloudWatch()
    store = feedwatch.S3Store("bucket", "quakenet", client=s3)
    world.publish("ev1", 1)
    common = dict(store=store, cloudwatch=cloudwatch, fetch=lambda feed: {"features": world.features},
                  detail=world.detail, log=lambda m: None)

    out = feedwatch.lambda_handler({}, None, **common)
    assert out["status"] == "ok" and out["updates"][0]["event_id"] == "ev1"
    assert set(s3.objects) == {"quakenet/watch/state.json", "quakenet/watch/heartbeat.json"}
    assert feedwatch.lambda_handler({}, None, **common)["updates"] == []      # the state in S3 remembers it
    assert cloudwatch.values == [("quakenet", "WatcherHealthy", 1.0)] * 2

    common["fetch"] = lambda feed: (_ for _ in ()).throw(feedwatch.WatchError("feed down"))
    with pytest.raises(feedwatch.WatchError):
        feedwatch.lambda_handler({}, None, **common)
    assert cloudwatch.values[-1] == ("quakenet", "WatcherHealthy", 0.0)


def test_the_module_imports_only_the_standard_library():
    """The Lambda zip holds this one file; an import of numpy or of the rest of src breaks it."""
    tree = ast.parse(Path(feedwatch.__file__).read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
    assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree))
    assert names - set(sys.stdlib_module_names) == {"boto3"}


def test_the_lambda_zip_carries_the_real_network_regions(tmp_path):
    regions = watch.regions()
    assert set(regions) == {"ema-highway", "hebei-transport", "toy-ca"}
    lon_min, lat_min, lon_max, lat_max = regions["hebei-transport"]
    assert lon_min < 115.0 < lon_max and lat_min < 37.5 < lat_max      # Xingtai
    lon_min, lat_min, lon_max, lat_max = regions["ema-highway"]
    assert lon_min < -71.06 < lon_max and lat_min < 42.36 < lat_max    # Boston
    lon_min, lat_min, lon_max, lat_max = regions["toy-ca"]
    assert lon_min < -121.88 < lon_max and lat_min < 37.04 < lat_max   # Loma Prieta


# ---- the email -------------------------------------------------------------------


class Outbox:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def __call__(self, events):
        if self.fail:
            raise RuntimeError("SNS unavailable")
        self.sent.append([e["event_id"] for e in events])


def test_a_new_earthquake_is_emailed_once_and_revisions_are_not(world):
    outbox = Outbox()
    world.publish("ev1", 1, updated=1000)
    assert world.poll(notify=outbox)["notified"] == ["ev1"]
    world.publish("ev1", 2, updated=2000)                     # a revised ShakeMap
    assert world.poll(notify=outbox)["notified"] == []
    assert outbox.sent == [["ev1"]]


def test_events_found_in_one_poll_share_one_email(world):
    outbox = Outbox()
    world.publish("main", 1)
    world.publish("after1", 1)
    world.publish("after2", 1)
    world.poll(notify=outbox)
    assert outbox.sent == [["main", "after1", "after2"]]


def test_a_failed_email_is_retried_on_the_next_poll(world):
    world.publish("ev1", 1)
    beat = world.poll(notify=Outbox(fail=True))
    assert beat["status"] == "degraded" and "notification failed" in beat["errors"][0]
    outbox = Outbox()
    assert world.poll(notify=outbox)["notified"] == ["ev1"] and outbox.sent == [["ev1"]]


def test_an_event_without_a_shakemap_yet_is_not_emailed_until_it_has_one(world):
    outbox = Outbox()
    world.features = [feature("slow")]                        # the catalogue has no ShakeMap for it yet
    world.poll(notify=outbox)
    world.publish("slow", 1, updated=2000)
    world.poll(notify=outbox)
    assert outbox.sent == [["slow"]]


def test_events_kept_from_before_notification_existed_are_not_emailed(world):
    (world.tmp / "watch").mkdir()
    (world.tmp / "watch" / "state.json").write_text(json.dumps(
        {"events": {"old": {"updated": 1000, "version": 1, "versions": [1]}}}))
    world.publish("old", 1, updated=1000)
    outbox = Outbox()
    world.poll(notify=outbox)
    assert outbox.sent == []


def test_the_email_names_the_event_the_networks_and_how_to_run_it():
    events = [{"event_id": "us7000abcd", "magnitude": 5.8, "place": "25 km SW of Xingtai, China",
               "time": 1_760_000_000_000, "lat": 37.0, "lon": 114.3, "networks": ["hebei-transport"],
               "version": 1, "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us7000abcd"}]
    subject, body = feedwatch.compose_email(events)
    assert subject == "M5.8 earthquake near hebei-transport: 25 km SW of Xingtai, China"
    assert "us7000abcd" in body and "ShakeMap version 1" in body and "eventpage/us7000abcd" in body
    assert "Networks: hebei-transport" in body and "python -m src serve" in body
    many = [{**events[0], "event_id": f"ev{i}", "magnitude": 4 + i / 10, "place": "x" * 80} for i in range(5)]
    subject, _ = feedwatch.compose_email(many)
    assert subject.startswith("5 earthquakes near hebei-transport, up to M4.4") and len(subject) <= 100


def test_the_lambda_emails_through_its_topic(world, monkeypatch, tmp_path):
    class FakeSNS:
        def __init__(self):
            self.published = []

        def publish(self, TopicArn, Subject, Message):
            self.published.append((TopicArn, Subject))

    sns = FakeSNS()
    monkeypatch.setattr(feedwatch, "sns_notifier",
                        lambda arn: lambda events: sns.publish(arn, *feedwatch.compose_email(events)))
    (tmp_path / "regions.json").write_text(json.dumps(REGIONS))
    monkeypatch.setenv("FEEDWATCH_REGIONS", str(tmp_path / "regions.json"))
    monkeypatch.setenv("FEEDWATCH_TOPIC_ARN", "arn:aws:sns:ap-southeast-2:1:topic")
    world.publish("ev1", 1)
    out = feedwatch.lambda_handler({}, None, store=feedwatch.S3Store("b", client=FakeS3()),
                                   cloudwatch=FakeCloudWatch(), fetch=lambda feed: {"features": world.features},
                                   detail=world.detail, log=lambda m: None)
    assert out["notified"] == ["ev1"] and sns.published[0][0] == "arn:aws:sns:ap-southeast-2:1:topic"
