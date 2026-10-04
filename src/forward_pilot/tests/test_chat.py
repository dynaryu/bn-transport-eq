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
