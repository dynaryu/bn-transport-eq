"""Tests for the natural-language layer.

None of these call the Anthropic API. The two things worth pinning do not need
it: that the tool routes through the real engine (so the forward-only boundary
and the validation still apply), and that the loop is wired correctly, which a
stub client proves without spending tokens or needing a credential.
"""

import json

import pytest

from src.forward_pilot import chat
from src.forward_pilot.catalog import load_case


@pytest.fixture(scope="module")
def ema():
    return load_case("ema-highway")


@pytest.fixture(scope="module")
def hebei():
    return load_case("hebei-transport")


# ---- the tool runs the real engine ---------------------------------------


def test_tool_runs_the_engine_and_records_the_full_result(ema):
    runs = []
    summary = json.loads(chat.run_query_for_chat(ema, runs, magnitude=7.0, samples=200))

    assert summary["scenario"]["magnitude"] == 7.0
    assert summary["samples"] == 200
    assert 0.0 <= summary["lowest_reliability_origins"][0]["p_reaches_destination"] <= 1.0

    # The model gets a summary; the page gets the whole result.
    assert len(runs) == 1
    assert len(runs[0]["result"]["components"]) == len(ema["edges"])
    assert len(summary["highest_risk_bridges"]) == 5


def test_tool_defaults_to_the_package_scenario(ema):
    runs = []
    chat.run_query_for_chat(ema, runs, samples=100)
    assert runs[0]["query"]["hazard"] == dict(ema["default_hazard"])


def test_tool_honours_the_package_hazard_model(hebei):
    runs = []
    summary = json.loads(chat.run_query_for_chat(hebei, runs, sa1_g=0.9, samples=100))
    assert summary["scenario"] == {"model": "uniform_sa1", "sa1_g": 0.9}


def test_stronger_shaking_lowers_reliability_through_the_tool(hebei):
    """The tool is a thin wrapper, so the engine's behaviour must survive it."""
    mild, severe = [], []
    low = json.loads(chat.run_query_for_chat(hebei, mild, sa1_g=0.3, samples=2000))
    high = json.loads(chat.run_query_for_chat(hebei, severe, sa1_g=1.1, samples=2000))
    assert (low["lowest_reliability_origins"][0]["p_reaches_destination"]
            > high["lowest_reliability_origins"][0]["p_reaches_destination"])


def test_samples_are_clamped_to_the_engine_limits(ema):
    runs = []
    chat.run_query_for_chat(ema, runs, samples=10 ** 9)
    assert runs[0]["query"]["samples"] == 20000


def test_unknown_node_is_reported_not_invented(ema):
    """Claude naming a node that does not exist must come back as a refusal it
    can read and correct, not as a crash or a quietly substituted node."""
    runs = []
    out = json.loads(chat.run_query_for_chat(ema, runs, origins=["n9999"], samples=100))
    assert out["refused"] is True
    assert "n9999" in out["reason"]
    assert runs == []


def test_population_absence_is_stated_rather_than_omitted(hebei):
    out = json.loads(chat.run_query_for_chat(hebei, [], samples=100))
    assert "no population weights" in out["population"]


# ---- the query is stated before it is computed ----------------------------


def test_the_query_is_stated_before_the_engine_runs(ema, monkeypatch):
    """The order is the feature: a statement emitted after the engine returned
    would look the same on screen and mean something weaker."""
    events = []
    seen_by_engine = []
    real = chat.run_forward_query

    def spy(network, query):
        seen_by_engine.append([e["type"] for e in events])
        return real(network, query)

    monkeypatch.setattr(chat, "run_forward_query", spy)
    chat.run_query_for_chat(ema, [], on_event=events.append, magnitude=7.0, samples=200)

    assert seen_by_engine == [["query"]]
    assert [e["type"] for e in events] == ["query", "computed"]


def test_statement_uses_the_model_variables_and_the_scenario(ema):
    events = []
    runs = []
    chat.run_query_for_chat(ema, runs, on_event=events.append, magnitude=7.0, samples=200)
    query, computed = events
    targets = query["statement"]["targets"]

    given = "Mw = 7, epicentre = (1, 10)"
    assert [t["expr"] for t in targets] == [
        f"P(S::o = 1 | {given})",
        f"P(X::e = 0 | {given})",
        f"E[PopFraction | {given}]",
    ]
    assert targets[0]["over"] == f"each of {len(ema['default_origins'])} origins o"
    assert "n22 or n66" in targets[0]["meaning"]
    assert "200 forward samples" in targets[0]["how"]
    # The bridge marginal is exact in both backends, and is not passed off as sampled.
    assert targets[1]["how"] == "closed form"

    # One value per target, in order, and they are the engine's numbers.
    result = runs[0]["result"]
    assert [v["value"] for v in computed["values"]] == [
        result["origin_service"][0]["probability"],
        result["components"][0]["p_unusable"],
        result["population_accessibility"]["expected_fraction"],
    ]


def test_statement_names_a_single_origin_and_omits_absent_population(hebei):
    events = []
    chat.run_query_for_chat(hebei, [], on_event=events.append, sa1_g=0.9, samples=100)
    query, computed = events
    targets = query["statement"]["targets"]

    origin = hebei["default_origins"][0]
    assert targets[0]["expr"] == f"P(S::{origin} = 1 | SA1 = 0.9 g at every bridge)"
    assert "Handan" in targets[0]["meaning"]
    # No population weights, so no PopFraction node and no query on one.
    assert len(targets) == len(computed["values"]) == 2
    assert "95% interval" in computed["values"][0]["note"]


def test_a_refused_query_is_stated_then_marked_not_computed(ema):
    events = []
    chat.run_query_for_chat(ema, [], on_event=events.append, origins=["n9999"], samples=100)
    assert [e["type"] for e in events] == ["query", "not_computed"]
    assert "n9999" in events[1]["reason"]


# ---- the boundary survives the natural-language layer ---------------------


def test_backward_questions_are_still_refused_by_the_engine(ema):
    """The tool exposes no way to ask one, and the engine would refuse anyway.

    This is the property the whole design rests on: the boundary is enforced in
    code, so no phrasing of the question can get round it.
    """
    from src.forward_pilot.engine import ForwardQueryError, run_forward_query

    with pytest.raises(ForwardQueryError, match="forward inference only"):
        run_forward_query(ema, {"query_type": "critical_components", "samples": 100})

    prompt = chat.system_prompt(ema)
    assert "FORWARD questions only" in prompt
    assert "Never state a probability" in prompt


def test_system_prompt_describes_the_actual_package(ema, hebei):
    ema_prompt = chat.system_prompt(ema)
    assert "ema_reference_scenario" in ema_prompt
    assert "n22" in ema_prompt and "n66" in ema_prompt
    assert f"{len(ema['edges'])}" in ema_prompt

    hebei_prompt = chat.system_prompt(hebei)
    # Hebei carries city names, so a question can say "Handan" instead of "n22".
    assert "Handan" in hebei_prompt
    assert "ema_reference_scenario" not in hebei_prompt


# ---- availability --------------------------------------------------------


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    """No key, no token, and an empty CLI config dir, whatever the host has.

    The config dir has to be redirected too, or the result of these tests would
    depend on whether whoever runs them happens to have signed in with ``ant``.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _write_profile(config_dir, name="default"):
    creds = config_dir / "credentials"
    creds.mkdir(exist_ok=True)
    (creds / f"{name}.json").write_text("{}", encoding="utf-8")


def test_reports_why_it_is_unavailable_without_credentials(no_credentials):
    ok, reason = chat.available()
    assert ok is False
    assert "credential" in reason.lower()
    # Both routes are named, because the cheaper one is the less obvious one.
    assert "ant auth login" in reason and "ANTHROPIC_API_KEY" in reason


def test_a_cli_login_counts_as_a_credential(no_credentials):
    """``ant auth login`` sets no environment variable, only a profile on disk.

    An env-only check reported the panel as unconfigured for someone already
    signed in, whose bare ``Anthropic()`` client would have worked fine.
    """
    _write_profile(no_credentials)
    assert chat.available() == (True, "")


def test_an_empty_api_key_is_reported_rather_than_ignored(no_credentials, monkeypatch):
    """An empty variable still wins its slot in the SDK's resolution order.

    So it authenticates with nothing *and* shadows a working profile. Falling
    through to the profile here would report the panel as available and then fail
    on the first question, which is the harder failure to read.
    """
    _write_profile(no_credentials)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    ok, reason = chat.available()
    assert ok is False
    assert "empty" in reason and "Unset it" in reason


def test_a_named_profile_that_is_not_stored_is_an_error(no_credentials, monkeypatch):
    # ANTHROPIC_PROFILE naming a missing profile is an error, not a fall-through
    # to the default profile, so the reason has to name the profile asked for.
    _write_profile(no_credentials)
    monkeypatch.setenv("ANTHROPIC_PROFILE", "demo")
    ok, reason = chat.available()
    assert ok is False
    assert "demo" in reason


def test_ask_refuses_an_empty_question(ema):
    with pytest.raises(chat.ChatUnavailable, match="Ask a question"):
        chat.ask(ema, "   ")


# ---- the loop, with a stub client ----------------------------------------


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Message:
    def __init__(self, text):
        self.content = [_Block(text)]


class _StubRunner:
    """Stands in for the SDK tool runner: calls the tool once, then answers."""

    def __init__(self, tool, tool_kwargs):
        self._tool, self._kwargs = tool, tool_kwargs

    def __iter__(self):
        payload = json.loads(self._tool.call(self._kwargs))
        worst = payload["lowest_reliability_origins"][0]
        yield _Message(f"Lowest reliability is {worst['name']} at "
                       f"{100 * worst['p_reaches_destination']:.1f}%.")


class _StubClient:
    def __init__(self, tool_kwargs):
        self._kwargs = tool_kwargs
        self.seen = {}
        outer = self

        class _Messages:
            def tool_runner(self, *, model, max_tokens, system, tools, messages):
                outer.seen = {"model": model, "system": system, "messages": messages}
                return _StubRunner(tools[0], outer._kwargs)

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


def test_ask_wires_the_tool_the_history_and_the_result_together(ema):
    client = _StubClient({"magnitude": 6.5, "samples": 300})
    out = chat.ask(
        ema, "How bad is a 6.5?",
        history=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        client=client,
    )

    assert out["ran_query"] is True
    assert "Lowest reliability is" in out["answer"]
    # The query it ran is returned for display, and the full result for the map.
    assert out["runs"][0]["hazard"]["magnitude"] == 6.5
    assert len(out["result"]["components"]) == len(ema["edges"])
    # History is passed through, with the new question last.
    assert client.seen["messages"][-1] == {"role": "user", "content": "How bad is a 6.5?"}
    assert len(client.seen["messages"]) == 3
    assert client.seen["model"] == chat.MODEL


def test_ask_drops_malformed_history_turns(ema):
    client = _StubClient({"samples": 100})
    chat.ask(ema, "q", history=[{"role": "system", "content": "x"}, {"role": "user"}], client=client)
    assert client.seen["messages"] == [{"role": "user", "content": "q"}]


def test_ask_reports_each_query_while_the_loop_is_running(ema):
    events = []
    out = chat.ask(ema, "How bad is a 6.5?", client=_StubClient({"magnitude": 6.5, "samples": 300}),
                   on_event=events.append)
    assert [e["type"] for e in events] == ["query", "computed"]
    assert events[0]["query"] == out["runs"][0]


# ---- the stream the page reads --------------------------------------------


@pytest.fixture
def pilot_url():
    """A real server on a free port, so the test reads what a browser would."""
    import threading
    from http.server import ThreadingHTTPServer

    from src.forward_pilot.server import PilotHandler

    server = ThreadingHTTPServer(("127.0.0.1", 0), PilotHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post_ask(url, body):
    from urllib.request import Request, urlopen

    request = Request(f"{url}/api/ask", data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    return urlopen(request, timeout=30)


def test_ask_endpoint_streams_the_query_then_the_values_then_the_answer(pilot_url, ema, monkeypatch):
    real = chat.ask
    monkeypatch.setattr(chat, "ask", lambda network, question, history, **kw: real(
        network, question, history, client=_StubClient({"magnitude": 6.5, "samples": 300}), **kw))

    with _post_ask(pilot_url, {"package_id": "ema-highway", "question": "How bad is a 6.5?"}) as response:
        assert response.headers["Content-Type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.read().decode("utf-8").splitlines()]

    assert [e["type"] for e in events] == ["query", "computed", "answer"]
    assert events[0]["statement"]["targets"][0]["expr"].startswith("P(S::o = 1 | Mw = 6.5")
    assert "Lowest reliability is" in events[2]["answer"]
    assert len(events[2]["result"]["components"]) == len(ema["edges"])


def test_ask_endpoint_keeps_a_status_code_when_it_fails_before_the_stream(pilot_url):
    """Headers wait for the first event, so an early failure is still an HTTP
    error the page can read as one, not a 200 with the bad news inside it."""
    from urllib.error import HTTPError

    with pytest.raises(HTTPError) as caught:
        _post_ask(pilot_url, {"package_id": "ema-highway", "question": "   "})
    assert caught.value.code == 503
    assert "Ask a question" in json.loads(caught.value.read())["error"]


def test_ask_endpoint_reports_a_late_failure_inside_the_stream(pilot_url, monkeypatch):
    def fails_after_the_query(network, question, history, on_event=None):
        on_event({"type": "query", "query": {}, "statement": {"given": "", "targets": []}})
        raise RuntimeError("the API went away")

    monkeypatch.setattr(chat, "ask", fails_after_the_query)
    with _post_ask(pilot_url, {"package_id": "ema-highway", "question": "q"}) as response:
        events = [json.loads(line) for line in response.read().decode("utf-8").splitlines()]

    assert [e["type"] for e in events] == ["query", "error"]
    assert "RuntimeError" in events[1]["error"]
