"""The page's endpoints answer, and the view data shows every BN and what connects."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.app.server import make_server
from src.app.views import view_data


@pytest.fixture(scope="module")
def base_url():
    server = make_server(port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url):
    with urlopen(url, timeout=60) as response:
        return response.status, response.read()


@pytest.fixture(autouse=True)
def isolated_server_state(tmp_path, monkeypatch):
    """HTTP tests use private sessions and never start workers that write model artifacts."""
    from types import SimpleNamespace
    from src.app import server
    import subprocess
    monkeypatch.setattr(server, "subprocess", SimpleNamespace(
        Popen=lambda *a, **kw: SimpleNamespace(poll=lambda: None), STDOUT=subprocess.STDOUT,
        CREATE_NEW_PROCESS_GROUP=0))
    monkeypatch.setattr(server, "_extractions", {})
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "SESSIONS", tmp_path / "sessions")


def post(url, body):
    request = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_view_data_has_the_schema_and_every_bn():
    views = {v["id"]: v for v in view_data()["views"]}
    assert {"library", "ema-highway/gmm", "hebei-transport/shakemap"} <= views.keys()
    links = {(x["from"], x["to"]) for x in views["library"]["links"]}
    for hazard in ("hazard.gmm", "hazard.shakemap"):
        assert (hazard, "fragility.bridge") in links          # alternatives, all type-compatible
    assert views["ema-highway/gmm"]["n_nodes"] == 588            # 387 + 129 SA measurements + a system state for each of 72 origins
    assert views["ema-highway/gmm"]["menu"]["settable"]["rows"]


def test_the_export_is_one_file_with_the_views_inside(tmp_path):
    from src.app.views import export_page

    html = export_page(tmp_path / "schema.html").read_text(encoding="utf-8")
    assert html.count("window.BN_DATA = ") == 1 and '"ema-highway/gmm"' in html
    assert 'src="/diagram-navigation.js"' not in html
    assert "function diagramNavigation(" in html
    assert "</script>" not in html.split("window.BN_DATA = ", 1)[1].split(";</script>", 1)[0]


def test_the_schema_page_is_current():
    from src.app.views import schema_page_is_current

    assert schema_page_is_current(), "schema/diagram.html is stale; run: python -m src diagram"


def test_the_page_and_its_data_are_served(base_url):
    status, body = get(base_url + "/")
    assert status == 200 and b"<title>quakenet</title>" in body and b"/api/example?assembly=" in body
    assert b'id="panel-query"' in body and b'id="panel-evidence"' in body     # the two panels under the map
    with urlopen(base_url + "/diagram-navigation.js", timeout=10) as response:
        assert response.status == 200 and response.headers.get_content_type() == "text/javascript"
        assert b"function diagramNavigation(" in response.read()
    status, body = get(base_url + "/bn")
    assert status == 200 and b"/api/views" in body                # the BN drawing, embedded in quakenet's BN tab
    status, body = get(base_url + "/api/views")
    assert status == 200 and json.loads(body)["networks"]["hebei-transport"]["rows"]["Bridge"] == 33


def test_a_query_is_answered_and_a_bad_one_explained(base_url):
    status, body = post(base_url + "/api/query", {"assembly": "ema-highway/gmm",
                                                  "targets": ["EdgeOpen::e0001"], "samples": 200})
    assert status == 200 and body["targets"][0]["node"] == "EdgeOpen::e0001"
    status, body = post(base_url + "/api/query", {"assembly": "ema-highway/gmm",
                                                  "targets": ["SA(1.0)::*"], "evidence": {"x": 1}})
    assert status == 400 and "no node 'x'" in body["error"]


def test_a_session_runs_and_drafting_without_a_model_says_why(base_url, monkeypatch):
    status, body = post(base_url + "/api/session/draft", {"assembly": "toy-ca/demo", "kind": "evidence", "text": "hi"})
    assert status == 400 and "Run" in body["error"]                   # no session yet
    status, body = post(base_url + "/api/session", {"assembly": "toy-ca/demo", "samples": 10000})
    assert status == 200 and body["n"] == 10000 and body["result"]["events"]
    assert body["timeline"][0]["elapsed_s"] >= 0
    status, body = post(base_url + "/api/session/query", {"assembly": "toy-ca/demo", "query": {
        "name": "Towns cut off", "target": {"kind": "cut_off"}, "decision": "track"}})
    assert status == 200 and body["timeline"][-1]["kind"] == "query"
    assert body["timeline"][-1]["elapsed_s"] >= 0 and body["timeline"][-1]["values"]["q1"]
    status, body = post(base_url + "/api/session/evidence", {"assembly": "toy-ca/demo", "variable": "EdgeOpen::e6",
                                                             "value": "closed"})
    assert status == 200 and body["timeline"][-1]["kind"] == "evidence"
    timings = [entry["elapsed_s"] for entry in body["timeline"]]
    assert len(timings) == 3 and all(elapsed >= 0 for elapsed in timings)
    from src.app import server
    server._sessions.clear()                         # reopen the saved session, as after a restart
    _, saved = get(base_url + "/api/session?assembly=toy-ca/demo")
    assert [entry["elapsed_s"] for entry in json.loads(saved)["session"]["timeline"]] == timings
    monkeypatch.setenv("BN_LLM", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    status, body = post(base_url + "/api/session/draft", {"assembly": "toy-ca/demo", "kind": "evidence", "text": "hi"})
    assert status == 503 and "GEMINI_API_KEY" in body["error"]


@pytest.mark.parametrize("failure,status,expected", [
    (ModuleNotFoundError("No module named 'rsr.rsr'", name="rsr.rsr"), 503, "rsr.rsr"),
    (RuntimeError("sampling failed"), 500, "RuntimeError"),
])
def test_inference_failures_return_json_instead_of_dropping_the_connection(
        base_url, monkeypatch, failure, status, expected):
    from src.app import server

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(server.engine, "answer", fail)
    code, body = post(base_url + "/api/query", {"assembly": "ema-highway/gmm", "samples": 200})
    assert code == status and expected in body["error"]
    if status == 503:
        assert "conda run -n quakenet python -m src serve" in body["error"]
    assert get(base_url + "/api/examples")[0] == 200


def test_an_example_carries_its_network_menu_and_rules(base_url):
    status, body = get(base_url + "/api/examples")
    assert status == 200 and "toy-ca/demo" in {e["id"] for e in json.loads(body)["examples"]}
    status, body = get(base_url + "/api/example?assembly=toy-ca/demo")
    ex = json.loads(body)
    assert status == 200 and set(ex["network"]["tables"]) >= {"Node", "Edge", "Bridge", "Event", "SystemEvent"}
    assert ex["objects"] == 37 and ex["rules"] == {"stored": 6, "origins": 6}
    assert len(ex["nodes"]) == 46 and ex["joint"] == ["hazard.gmm"]          # what the BN tab draws, layer by layer
    settable = {(r["entity"], r["attribute"]) for r in ex["menu"]["settable"]["rows"]}
    assert {("Event", "mw"), ("SystemEvent", "target_add_h")} <= settable
    with pytest.raises(HTTPError) as missing:
        get(base_url + "/api/example?assembly=nowhere/x")
    assert missing.value.code == 404 and "nowhere/x" in json.loads(missing.value.read())["error"]


def test_a_map_run_returns_every_family_and_says_rules_were_used(base_url):
    events = [{"node": f"{q}::*", "op": "==", "value": 0} for q in ("SystemState", "EdgeOpen")]
    status, body = post(base_url + "/api/query", {"assembly": "toy-ca/demo", "events": events, "samples": 500})
    assert status == 200 and len(body["events"]) == 6 + 10
    assert any("rsr rules exact for 6 of 6" in w for w in body["warnings"])


@pytest.fixture
def stored_at_network_threshold(tmp_path, monkeypatch):
    """The toy-ca rules as the repository ships them, at the network's threshold only, whatever else
    the app may have extracted since: a private copy the model directory points at."""
    import shutil

    from src.builder import bn_builder

    model = Path(__file__).resolve().parents[3] / "examples/toy-ca/tbnpy/demo"
    copy = tmp_path / "model"
    for folder in (model / "rsr").glob("*/target_add_h=0.5"):
        shutil.copytree(folder, copy / "rsr" / folder.parent.name / folder.name)
    monkeypatch.setattr(bn_builder, "model_dir", lambda assembly_file: copy)
    return copy


def test_a_run_left_mostly_unclassified_starts_extraction_in_the_background(base_url, monkeypatch,
                                                                         stored_at_network_threshold):
    from src.app import server

    started = []
    import subprocess
    from types import SimpleNamespace

    job = type("Job", (), {"poll": lambda self: None})
    # Only the server's own subprocess: git, which the query records call, keeps the real one.
    monkeypatch.setattr(server, "subprocess", SimpleNamespace(
        Popen=lambda cmd, **kw: started.append(cmd) or job(), STDOUT=subprocess.STDOUT, CREATE_NEW_PROCESS_GROUP=0))
    monkeypatch.setattr(server, "_extractions", {})
    query = {"assembly": "toy-ca/demo", "events": [{"node": "SystemState::*", "op": "==", "value": 0}], "samples": 500,
             "rows": {"reach-key-destination": {"target_add_h": 0.25}}}
    status, body = post(base_url + "/api/query", query)
    assert status == 200 and any("in the background" in w for w in body["warnings"])
    assert started[0][2:6] == ["src", "rules", "toy-ca/demo", "--target-add-h"] and started[0][6] == "0.25"
    status, body = post(base_url + "/api/query", query)                 # the same scenario again: read from its bank
    assert len(started) == 1 and body["samples_from"] == "the scenario's stored samples"
    other = {**query, "rows": {**query["rows"], "demo-san-andreas": {"mw": 7.0}}}
    status, body = post(base_url + "/api/query", other)                 # new samples, same threshold: one job only
    assert len(started) == 1 and any("still being extracted" in w for w in body["warnings"])


def test_a_negative_threshold_is_refused(base_url):
    status, body = post(base_url + "/api/query", {"assembly": "toy-ca/demo", "targets": ["SystemState::n4"],
                                                  "rows": {"reach-key-destination": {"target_add_h": -0.1}}})
    assert status == 400 and "below its minimum" in body["error"]
