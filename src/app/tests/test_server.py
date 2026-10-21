"""The page's endpoints answer, and the view data shows every BN and what connects."""

from __future__ import annotations

import json
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
        assert (hazard, "fragility.lognormal") in links          # alternatives, all type-compatible
    assert views["ema-highway/gmm"]["n_nodes"] == 1093           # 805 + one SystemState per event
    assert views["ema-highway/gmm"]["menu"]["settable"]["rows"]


def test_the_export_is_one_file_with_the_views_inside(tmp_path):
    from src.app.views import export_page

    html = export_page(tmp_path / "schema.html").read_text(encoding="utf-8")
    assert html.count("window.BN_DATA = ") == 1 and '"ema-highway/gmm"' in html
    assert "</script>" not in html.split("window.BN_DATA = ", 1)[1].split(";</script>", 1)[0]


def test_the_schema_page_is_current():
    from src.app.views import schema_page_is_current

    assert schema_page_is_current(), "schema/diagram.html is stale; run: python -m src diagram"


def test_the_page_and_its_data_are_served(base_url):
    status, body = get(base_url + "/")
    assert status == 200 and b"/api/views" in body
    status, body = get(base_url + "/api/views")
    assert status == 200 and json.loads(body)["networks"]["hebei-transport"]["rows"]["Bridge"] == 33


def test_a_query_is_answered_and_a_bad_one_explained(base_url):
    status, body = post(base_url + "/api/query", {"assembly": "ema-highway/gmm",
                                                  "targets": ["TravelTime::n1>n22:time"], "samples": 200})
    assert status == 200 and body["targets"][0]["node"] == "TravelTime::n1>n22:time"
    status, body = post(base_url + "/api/query", {"assembly": "ema-highway/gmm",
                                                  "targets": ["SA(1.0)::*"], "evidence": {"x": 1}})
    assert status == 400 and "forward inference only" in body["error"]


def test_ask_without_a_model_says_why(base_url, monkeypatch):
    monkeypatch.setenv("BN_LLM", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    status, body = post(base_url + "/api/ask", {"assembly": "ema-highway/gmm", "question": "hi"})
    assert status == 503 and "GEMINI_API_KEY" in body["error"]
