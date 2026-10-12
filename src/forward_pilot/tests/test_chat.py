"""Tests for the natural-language layer.

None of these call a model. The two things worth pinning do not need one: that
the tool routes through the real engine (so the forward-only boundary and the
validation still apply), and that the loop is wired correctly, which a stub
client proves without a running Ollama server, tokens or a credential. The
Anthropic-backend tests skip when the ``anthropic`` package is not installed.
"""

import copy
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


@pytest.fixture(autouse=True)
def empty_result_cache():
    """Each test sees the engine run, unless it is the cache under test."""
    chat.clear_result_cache()
    yield
    chat.clear_result_cache()


@pytest.fixture(autouse=True)
def no_catalogue_search(monkeypatch):
    """Both packages offer ShakeMaps now, so ask() would search the USGS
    catalogue for the prompt's event list; tests must not touch the network."""
    monkeypatch.setattr(chat, "_shakemap_events", lambda network: [])


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


def test_top_n_sets_how_many_bridges_the_model_is_shown(ema):
    """Five was fixed once, so "the top 10 bridges" could not be answered
    without the model inventing the other five."""
    runs = []
    summary = json.loads(chat.run_query_for_chat(ema, runs, samples=100, top_n=10))
    bridges = summary["highest_risk_bridges"]
    assert len(bridges) == 10 and len(summary["lowest_reliability_origins"]) == 10
    assert [b["edge"] for b in bridges] == [c["edge_id"] for c in runs[0]["result"]["components"][:10]]
    assert len(json.loads(chat.run_query_for_chat(ema, [], samples=100, top_n=999))["highest_risk_bridges"]) \
        == chat.MAX_TOP_N


def test_samples_are_clamped_to_the_engine_limits(ema):
    runs = []
    chat.run_query_for_chat(ema, runs, samples=10 ** 9)
    assert runs[0]["query"]["samples"] == 20000


def test_unknown_node_is_reported_not_invented(ema):
    """The agent naming a node that does not exist must come back as a refusal it
    can read and correct, not as a crash or a quietly substituted node."""
    runs = []
    out = json.loads(chat.run_query_for_chat(ema, runs, origins=["n9999"], samples=100))
    assert out["refused"] is True
    assert "n9999" in out["reason"]
    assert runs == []


def test_population_absence_is_stated_rather_than_omitted(hebei):
    out = json.loads(chat.run_query_for_chat(hebei, [], samples=100))
    assert "no population weights" in out["population"]


# ---- an identical query reuses the earlier run -------------------------------


@pytest.fixture
def engine_calls(monkeypatch):
    calls = []
    real = chat.run_forward_query

    def counting(network, query):
        calls.append(query)
        return real(network, query)

    monkeypatch.setattr(chat, "run_forward_query", counting)
    return calls


def test_an_identical_query_is_not_run_twice(ema, engine_calls):
    first_events, second_events, runs = [], [], []
    first = json.loads(chat.run_query_for_chat(ema, runs, on_event=first_events.append, magnitude=7.0, samples=200))
    second = json.loads(chat.run_query_for_chat(ema, runs, on_event=second_events.append, magnitude=7.0,
                                                samples=200, top_n=10))
    assert len(engine_calls) == 1
    # The same numbers, with the longer list built from the stored result.
    assert second["highest_risk_bridges"][:5] == first["highest_risk_bridges"]
    assert len(second["highest_risk_bridges"]) == 10
    assert runs[0]["result"] is runs[1]["result"]
    # Still stated first, and marked so the page can say it was reused.
    assert [e["type"] for e in second_events] == ["query", "computed"]
    assert [e["reused"] for e in first_events] == [False, False]
    assert [e["reused"] for e in second_events] == [True, True]


def test_a_reused_result_equals_a_fresh_run(ema):
    """The premise of the cache: the fixed seed makes a rerun identical."""
    from src.forward_pilot.engine import run_forward_query

    runs = []
    chat.run_query_for_chat(ema, runs, magnitude=7.0, samples=200)
    chat.run_query_for_chat(ema, runs, magnitude=7.0, samples=200)
    assert runs[1]["result"] == run_forward_query(ema, runs[0]["query"])


def test_any_change_to_the_query_runs_the_engine(ema, engine_calls):
    chat.run_query_for_chat(ema, [], magnitude=7.0, samples=200)
    chat.run_query_for_chat(ema, [], magnitude=7.0, samples=300)
    chat.run_query_for_chat(ema, [], magnitude=7.1, samples=200)
    chat.run_query_for_chat(ema, [], magnitude=7.0, samples=200, origins=["n1"])
    assert len(engine_calls) == 4


def test_a_rebuilt_package_is_not_served_an_old_result(ema, engine_calls):
    chat.run_query_for_chat(ema, [], magnitude=7.0, samples=200)
    changed = {**ema, "edges": {**ema["edges"]}}
    first_edge = next(iter(changed["edges"]))
    changed["edges"][first_edge] = {**changed["edges"][first_edge], "bridge_class": "HWB1"}
    chat.run_query_for_chat(changed, [], magnitude=7.0, samples=200)
    assert len(engine_calls) == 2


def test_refusals_are_not_kept(ema, engine_calls):
    for _ in range(2):
        out = json.loads(chat.run_query_for_chat(ema, [], origins=["n9999"], samples=100))
        assert out["refused"] is True
    assert len(engine_calls) == 2


def test_the_cache_keeps_only_the_most_recent_results(ema, engine_calls, monkeypatch):
    monkeypatch.setattr(chat, "RESULT_CACHE_SIZE", 2)
    for samples in (100, 200, 300):
        chat.run_query_for_chat(ema, [], samples=samples)
    chat.run_query_for_chat(ema, [], samples=300)       # still kept
    chat.run_query_for_chat(ema, [], samples=100)       # evicted, so run again
    assert [q["samples"] for q in engine_calls] == [100, 200, 300, 100]


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


# ---- choosing a backend ---------------------------------------------------


@pytest.fixture
def ollama_backend(monkeypatch):
    monkeypatch.setenv("FORWARD_PILOT_LLM", "ollama")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


def test_ollama_is_the_default_backend(monkeypatch):
    monkeypatch.delenv("FORWARD_PILOT_LLM", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert chat.provider() == "ollama"
    assert chat.model_name() == chat.OLLAMA_MODEL


def test_an_unknown_backend_is_named_in_the_reason(monkeypatch):
    monkeypatch.setenv("FORWARD_PILOT_LLM", "gpt")
    ok, reason = chat.available()
    assert ok is False
    assert "'gpt'" in reason and "ollama" in reason


def test_ollama_unavailable_without_a_server(ollama_backend, monkeypatch):
    # Port 9 (discard) on loopback: nothing listens, so the connection is refused.
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    ok, reason = chat.available()
    assert ok is False
    assert "ollama serve" in reason and "127.0.0.1:9" in reason


def test_ollama_reports_a_model_that_has_not_been_pulled(ollama_backend, monkeypatch):
    monkeypatch.setattr(chat, "_ollama_request",
                        lambda path, payload=None, **kw: {"models": [{"name": "llama3.2:3b"}]})
    ok, reason = chat.available()
    assert ok is False
    assert f"ollama pull {chat.OLLAMA_MODEL}" in reason
    assert "llama3.2:3b" in reason


def test_ollama_accepts_a_model_named_without_its_tag(ollama_backend, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3")
    monkeypatch.setattr(chat, "_ollama_request",
                        lambda path, payload=None, **kw: {"models": [{"name": "qwen3:latest"}]})
    assert chat.available() == (True, "")


def test_gemini_needs_a_key(monkeypatch):
    monkeypatch.setenv("FORWARD_PILOT_LLM", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    ok, reason = chat.available()
    assert ok is False and "GEMINI_API_KEY" in reason
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert chat.available() == (True, "")
    assert chat.model_name() == chat.GEMINI_MODEL


class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _gemini_network(monkeypatch, outcomes):
    """Replace the network with a script: each item is a status code to fail
    with, "timeout", or a dict to return. Returns the list of attempts made."""
    import io
    from urllib.error import HTTPError

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(chat, "GEMINI_RETRY_DELAY", 0)
    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(timeout)
        outcome = outcomes.pop(0)
        if outcome == "timeout":
            raise TimeoutError("timed out")
        if isinstance(outcome, int):
            body = io.BytesIO(json.dumps([{"error": {"message": "high demand"}}]).encode())
            raise HTTPError(request.full_url, outcome, "error", {}, body)
        return _FakeResponse(outcome)

    monkeypatch.setattr(chat, "urlopen", fake_urlopen)
    return attempts


def test_gemini_overload_is_retried_once(monkeypatch):
    attempts = _gemini_network(monkeypatch, [503, {"choices": []}])
    assert chat._gemini_request("/chat/completions", {}) == {"choices": []}
    assert len(attempts) == 2
    assert attempts[0] == chat.GEMINI_TIMEOUT < chat.OLLAMA_TIMEOUT


def test_gemini_overload_that_persists_says_what_to_do(monkeypatch):
    _gemini_network(monkeypatch, [503, 503])
    with pytest.raises(chat.ChatUnavailable, match="overloaded.*FORWARD_PILOT_LLM=ollama"):
        chat._gemini_request("/chat/completions", {})


def test_a_hanging_gemini_request_gives_up_after_two_short_waits(monkeypatch):
    attempts = _gemini_network(monkeypatch, ["timeout", "timeout"])
    with pytest.raises(chat.ChatUnavailable, match="did not answer"):
        chat._gemini_request("/chat/completions", {})
    assert len(attempts) == 2


def test_a_bad_key_is_not_retried(monkeypatch):
    attempts = _gemini_network(monkeypatch, [401])
    with pytest.raises(chat.ChatUnavailable, match="key was rejected"):
        chat._gemini_request("/chat/completions", {})
    assert len(attempts) == 1


# ---- tool arguments from a small model -----------------------------------


def test_arguments_sent_as_strings_are_converted():
    args = chat._coerce_tool_arguments('{"magnitude": "7.0", "samples": "500", "origins": "n22"}')
    assert args == {"magnitude": 7.0, "samples": 500, "origins": ["n22"]}


def test_an_unknown_argument_is_refused_not_dropped():
    """Dropping it would run a query the model did not mean, and say nothing."""
    with pytest.raises(ValueError, match="unknown parameter.*depth_km"):
        chat._coerce_tool_arguments({"magnitude": 7.0, "depth_km": 10})


def test_tool_schema_lists_every_parameter_of_the_tool():
    import inspect

    schema = chat._tool_schema()["function"]["parameters"]["properties"]
    accepted = set(inspect.signature(chat.run_query_for_chat).parameters) - {"network", "runs", "on_event", "known_events"}
    assert set(schema) == accepted


# ---- availability of the Anthropic backend --------------------------------


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    """No key, no token, and an empty CLI config dir, whatever the host has.

    The config dir has to be redirected too, or the result of these tests would
    depend on whether whoever runs them happens to have signed in with ``ant``.
    """
    pytest.importorskip("anthropic")
    monkeypatch.setenv("FORWARD_PILOT_LLM", "anthropic")
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


# ---- the loop, with a stub Ollama server ---------------------------------


class _StubOllama:
    """Stands in for Ollama's /api/chat: asks for the tool once, then answers
    from what the tool returned, the way a model reading the result would."""

    def __init__(self, tool_arguments, tool_name="run_forward_bn_query"):
        self._arguments, self._name = tool_arguments, tool_name
        self.requests = []

    def __call__(self, path, payload):
        assert path == "/api/chat"
        self.requests.append(copy.deepcopy(payload))  # the loop keeps appending to its list
        last = payload["messages"][-1]
        if last["role"] != "tool":
            return {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": self._name, "arguments": self._arguments}}]}}
        result = json.loads(last["content"])
        if "lowest_reliability_origins" not in result:
            return {"message": {"role": "assistant", "content": f"Tool said: {last['content']}"}}
        worst = result["lowest_reliability_origins"][0]
        return {"message": {"role": "assistant", "content":
                "<think>work it out</think>Lowest reliability is "
                f"{worst['name']} at {100 * worst['p_reaches_destination']:.1f}%."}}


def test_ask_wires_the_tool_the_history_and_the_result_together(ema, ollama_backend):
    client = _StubOllama({"magnitude": 6.5, "samples": 300})
    out = chat.ask(
        ema, "How bad is a 6.5?",
        history=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        client=client,
    )

    assert out["ran_query"] is True
    # The reply is the prose only: inline thinking is stripped.
    assert out["answer"].startswith("Lowest reliability is")
    # The query it ran is returned for display, and the full result for the map.
    assert out["runs"][0]["hazard"]["magnitude"] == 6.5
    assert len(out["result"]["components"]) == len(ema["edges"])
    # System prompt first, history passed through, the new question last.
    first = client.requests[0]
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][1:] == [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "How bad is a 6.5?"},
    ]
    assert first["model"] == out["model"] == chat.model_name()
    assert first["tools"][0]["function"]["name"] == "run_forward_bn_query"


def test_ask_drops_malformed_history_turns(ema, ollama_backend):
    client = _StubOllama({"samples": 100})
    chat.ask(ema, "q", history=[{"role": "system", "content": "x"}, {"role": "user"}], client=client)
    assert client.requests[0]["messages"][1:] == [{"role": "user", "content": "q"}]


def test_ask_reports_each_query_while_the_loop_is_running(ema, ollama_backend):
    events = []
    out = chat.ask(ema, "How bad is a 6.5?", client=_StubOllama({"magnitude": 6.5, "samples": 300}),
                   on_event=events.append)
    assert [e["type"] for e in events] == ["query", "computed"]
    assert events[0]["query"] == out["runs"][0]


def test_ollama_requests_skip_thinking_and_keep_the_model_loaded(ema, ollama_backend, monkeypatch):
    monkeypatch.delenv("OLLAMA_THINK", raising=False)
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
    client = _StubOllama({"samples": 100})
    chat.ask(ema, "q", client=client)
    assert all(r["think"] is False and r["keep_alive"] == "30m" for r in client.requests)

    monkeypatch.setenv("OLLAMA_THINK", "1")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "2h")
    client = _StubOllama({"samples": 100})
    chat.ask(ema, "q", client=client)
    assert client.requests[0]["think"] is True and client.requests[0]["keep_alive"] == "2h"


def test_a_model_that_rejects_the_think_option_is_asked_again_without_it(ema, ollama_backend):
    stub = _StubOllama({"samples": 100})

    def rejects_think(path, payload):
        if "think" in payload:
            raise chat.ChatUnavailable('Ollama refused the request (400): "llama3" does not support thinking')
        return stub(path, payload)

    out = chat.ask(ema, "q", client=rejects_think)
    assert out["ran_query"] is True
    assert all("think" not in r for r in stub.requests)


def test_preload_loads_the_configured_model(ollama_backend, monkeypatch):
    calls = []
    monkeypatch.setattr(chat, "_ollama_request", lambda path, payload=None, **kw: calls.append((path, payload)) or {})
    assert "Loaded" in chat.preload_ollama()
    assert calls == [("/api/generate", {"model": chat.model_name(), "keep_alive": "30m"})]

    def down(path, payload=None, **kw):
        raise chat.ChatUnavailable("Could not reach Ollama")

    monkeypatch.setattr(chat, "_ollama_request", down)
    assert chat.preload_ollama().startswith("Could not preload")


def test_bad_arguments_go_back_to_the_model_without_running_anything(ema, ollama_backend):
    events = []
    out = chat.ask(ema, "q", client=_StubOllama({"magnitude": 6.5, "depth_km": 10}),
                   on_event=events.append)
    assert out["ran_query"] is False and events == []
    assert "depth_km" in out["answer"]


def test_a_tool_that_does_not_exist_is_reported_to_the_model(ema, ollama_backend):
    out = chat.ask(ema, "q", client=_StubOllama({}, tool_name="estimate_probability"))
    assert out["ran_query"] is False
    assert "estimate_probability" in out["answer"]


def test_a_model_that_never_stops_calling_tools_is_cut_off(ema, ollama_backend):
    calls = []

    def loops_forever(path, payload):
        calls.append(payload)
        return {"message": {"content": "", "tool_calls": [
            {"function": {"name": "run_forward_bn_query", "arguments": {"samples": 100}}}]}}

    out = chat.ask(ema, "q", client=loops_forever)
    assert len(calls) == chat.MAX_TOOL_ROUNDS
    assert "Stopped after" in out["answer"]


# ---- "the scenario" means the one selected in the page -----------------------


def test_the_page_scenario_becomes_the_default_query(hebei, ollama_backend):
    client = _StubOllama({"samples": 100})          # no hazard parameters at all
    out = chat.ask(hebei, "which bridge should be inspected given the scenario", client=client,
                   scenario={"model": "uniform_sa1", "sa1_g": 1.1})
    assert out["runs"][0]["hazard"] == {"model": "uniform_sa1", "sa1_g": 1.1}
    system = client.requests[0]["messages"][0]["content"]
    assert "currently selected in the page is uniform SA(1 s) = 1.1 g" in system


def test_a_scenario_the_package_cannot_run_is_ignored(hebei, ollama_backend):
    client = _StubOllama({"samples": 100})
    out = chat.ask(hebei, "q", client=client, scenario={"model": "ema_reference_scenario", "magnitude": 7})
    assert out["runs"][0]["hazard"] == dict(hebei["default_hazard"])
    assert "currently selected" not in client.requests[0]["messages"][0]["content"]


def test_a_selected_shakemap_supplies_its_event_to_a_bare_query(hebei):
    network = {**hebei, "default_hazard": {**hebei["default_hazard"], "model": "shakemap", "event_id": "ev1"}}
    runs, events = [], []
    chat.run_query_for_chat(network, runs, on_event=events.append, samples=100)
    assert events[0]["query"]["hazard"] == {"model": "shakemap", "event_id": "ev1"}


def test_the_agent_is_told_it_cannot_save_files(ema):
    """qwen3 told a user it had saved the chat to ./chat.txt; no such file existed."""
    prompt = chat.system_prompt(ema)
    assert "ONLY ability is the query tool" in prompt
    assert "never claim to have done it" in prompt and "'Download conversation'" in prompt


def test_ranking_bridges_under_a_scenario_is_described_as_forward(ema):
    """Gemini refused "which bridge should be inspected" as a backward question
    when the rule only warned against criticality rankings."""
    prompt = chat.system_prompt(ema)
    assert "which bridges should be inspected after this earthquake?' is forward" in prompt
    assert "OBSERVED outage" in prompt


# ---- the loop through Gemini's OpenAI-compatible API, with a stub ---------


class _StubGemini:
    """Stands in for /chat/completions: one tool call (with a thought signature,
    as Gemini sends), then an answer read from the tool result."""

    SIGNATURE = {"google": {"thought_signature": "sig-123"}}

    def __init__(self, arguments):
        self._arguments = json.dumps(arguments)        # OpenAI format: a JSON string
        self.requests = []

    def __call__(self, path, payload):
        assert path == "/chat/completions"
        self.requests.append(copy.deepcopy(payload))
        last = payload["messages"][-1]
        if last["role"] != "tool":
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "extra_content": self.SIGNATURE,
                "function": {"name": "run_forward_bn_query", "arguments": self._arguments}}]}}]}
        bridges = json.loads(last["content"])["highest_risk_bridges"]
        return {"choices": [{"message": {"role": "assistant",
                                         "content": f"Listed {len(bridges)} bridges."}}]}


def test_gemini_loop_runs_the_tool_and_echoes_the_call_back(ema, monkeypatch):
    monkeypatch.setenv("FORWARD_PILOT_LLM", "gemini")
    client = _StubGemini({"magnitude": 6.5, "samples": 300, "top_n": 10})
    out = chat.ask(ema, "Top 10 bridges for inspection?", client=client)

    assert out["answer"] == "Listed 10 bridges."
    assert out["runs"][0]["hazard"]["magnitude"] == 6.5
    second = client.requests[1]["messages"]
    assistant, tool = second[-2], second[-1]
    # The thought signature goes back exactly as it came, and the result is tied to its call.
    assert assistant["tool_calls"][0]["extra_content"] == _StubGemini.SIGNATURE
    assert tool == {"role": "tool", "tool_call_id": "call_1", "content": tool["content"]}
    assert client.requests[0]["model"] == chat.model_name()


# ---- the loop through the Anthropic SDK, with a stub client ---------------


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


def test_anthropic_backend_wires_the_tool_and_the_history(ema, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("FORWARD_PILOT_LLM", "anthropic")
    client = _StubClient({"magnitude": 6.5, "samples": 300})
    out = chat.ask(
        ema, "How bad is a 6.5?",
        history=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        client=client,
    )

    assert out["ran_query"] is True
    assert "Lowest reliability is" in out["answer"]
    assert out["runs"][0]["hazard"]["magnitude"] == 6.5
    assert client.seen["messages"][-1] == {"role": "user", "content": "How bad is a 6.5?"}
    assert len(client.seen["messages"]) == 3
    assert client.seen["model"] == chat.ANTHROPIC_MODEL


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


def test_ask_endpoint_streams_the_query_then_the_values_then_the_answer(pilot_url, ema, monkeypatch,
                                                                        ollama_backend):
    real = chat.ask
    monkeypatch.setattr(chat, "ask", lambda network, question, history, **kw: real(
        network, question, history, client=_StubOllama({"magnitude": 6.5, "samples": 300}), **kw))

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
    def fails_after_the_query(network, question, history, on_event=None, scenario=None):
        on_event({"type": "query", "query": {}, "statement": {"given": "", "targets": []}})
        raise RuntimeError("the API went away")

    monkeypatch.setattr(chat, "ask", fails_after_the_query)
    with _post_ask(pilot_url, {"package_id": "ema-highway", "question": "q"}) as response:
        events = [json.loads(line) for line in response.read().decode("utf-8").splitlines()]

    assert [e["type"] for e in events] == ["query", "error"]
    assert "RuntimeError" in events[1]["error"]
