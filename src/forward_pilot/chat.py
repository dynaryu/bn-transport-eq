"""Natural-language front door to the forward BN query.

The division of labour is the whole design, and it is not negotiable:

    The LLM agent chooses the query.  The engine computes the answer.

The agent reads a question, picks the scenario parameters, and calls the same
``run_forward_query`` the UI calls. The numbers in its reply are the numbers the
engine returned. It is never asked to estimate a probability itself, because it
would produce a plausible one and nothing downstream could catch it -- the same
reason ``hazard/gmpe`` refuses to invent coefficients.

Two properties follow from routing through the real engine rather than around it:

* The forward-only boundary still holds. ``_validate_query`` raises
  ``ForwardQueryError`` for diagnostic and criticality queries whatever the
  agent asks for, and the tool hands that refusal back so it can be explained rather
  than silently worked around.
* Every answer carries the query that produced it, so a reader can check the
  translation from question to parameters instead of trusting it.

The order is part of that second property. Each query is written out as a
probability statement in the model's own variables and reported *before* the
engine is called, then the values, then the prose. A reader sees what is about
to be computed while it can still be read as a claim to check, rather than as a
footnote to an answer they have already accepted.

Two backends can drive the loop, chosen by ``FORWARD_PILOT_LLM``:

* ``ollama`` (the default) -- a local model served by Ollama, reached over its
  HTTP API with the standard library alone. Free, offline, nothing to install in
  Python. ``OLLAMA_MODEL`` and ``OLLAMA_BASE_URL`` override the defaults.
* ``gemini`` -- Google Gemini through its OpenAI-compatible API, also with the
  standard library alone. Needs ``GEMINI_API_KEY``; ``GEMINI_MODEL`` overrides
  the model.
* ``anthropic`` -- Claude through the ``anthropic`` SDK and a Console credential.

The division of labour above holds for either: the model only picks parameters,
so a small local model costs accuracy in translating the question, which the
stated query exposes, never in the numbers.

This module is optional. With no backend available the rest of the pilot works
exactly as before; ``available()`` reports why, and the UI hides the panel.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any, Callable, Iterable, Mapping, Sequence

from .engine import ForwardQueryError, run_forward_query

PROVIDERS = ("ollama", "gemini", "anthropic")
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
# A hosted model answers in seconds or not at all: under load the API can hold a
# request open without replying, so wait far less than for a local model, and
# retry the failures Google itself calls temporary.
GEMINI_TIMEOUT = 45
GEMINI_ATTEMPTS = 2
GEMINI_RETRY_DELAY = 3.0
_GEMINI_TRANSIENT = {500, 502, 503, 504}
ANTHROPIC_MODEL = "claude-opus-5"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 300    # seconds per model call; a CPU-only 8B model can be slow
OLLAMA_KEEP_ALIVE = "30m"   # how long Ollama keeps the model loaded after a request
MAX_TOOL_ROUNDS = 8     # a small model can loop on tool calls; stop it here
MAX_TOKENS = 4000
DEFAULT_SAMPLES = 4000
MAX_TOP_N = 50          # longest bridge / origin list handed back to the model
MAX_TURNS = 12          # cap the stored history so a long session cannot grow unbounded


class ChatUnavailable(RuntimeError):
    """The natural-language layer is not configured or not installed."""


def provider() -> str:
    """Which backend drives the chat: ``FORWARD_PILOT_LLM``, Ollama by default."""
    return (os.environ.get("FORWARD_PILOT_LLM") or "ollama").strip().lower()


def model_name() -> str:
    if provider() == "anthropic":
        return ANTHROPIC_MODEL
    if provider() == "gemini":
        return os.environ.get("GEMINI_MODEL") or GEMINI_MODEL
    return os.environ.get("OLLAMA_MODEL") or OLLAMA_MODEL


def _ollama_keep_alive() -> str:
    return os.environ.get("OLLAMA_KEEP_ALIVE") or OLLAMA_KEEP_ALIVE


def _ollama_think() -> bool:
    """Thinking is off unless ``OLLAMA_THINK`` turns it on.

    Choosing a handful of tool parameters gains little from a written-out chain
    of reasoning, and a thinking model spends most of each round trip on it.
    """
    return (os.environ.get("OLLAMA_THINK") or "").strip().lower() in {"1", "true", "yes", "on"}


def preload_ollama() -> str:
    """Load the configured model now, so the first question does not wait for it.

    Best effort: a missing server or model is reported, not raised, because the
    panel already explains those when a question is asked.
    """
    if provider() != "ollama":
        return ""
    try:
        _ollama_request("/api/generate", {"model": model_name(), "keep_alive": _ollama_keep_alive()})
    except ChatUnavailable as exc:
        return f"Could not preload {model_name()}: {exc}"
    return f"Loaded {model_name()} in Ollama; it stays loaded for {_ollama_keep_alive()} after each question."


def _ollama_base_url() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or OLLAMA_BASE_URL).rstrip("/")


def _ollama_request(path: str, payload: Mapping[str, Any] | None = None, *,
                    timeout: float = OLLAMA_TIMEOUT) -> dict[str, Any]:
    """GET (no payload) or POST JSON to the Ollama server and decode the reply.

    Failures come back as ``ChatUnavailable`` with a sentence saying what to do,
    which the server shows as it stands rather than as an exception class name.
    """
    url = _ollama_base_url() + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        raise ChatUnavailable(f"Ollama refused the request ({exc.code}): {detail or exc.reason}") from exc
    except TimeoutError as exc:
        raise ChatUnavailable(
            f"Ollama took longer than {timeout:g}s to answer. A smaller model "
            "(set OLLAMA_MODEL) or a GPU will help; the controls still work."
        ) from exc
    except URLError as exc:
        raise ChatUnavailable(
            f"Could not reach Ollama at {_ollama_base_url()} ({exc.reason}). "
            "Start it with ollama serve."
        ) from exc


def _config_dir() -> Path:
    """Where the Anthropic CLI keeps its OAuth profiles.

    ``ant auth login`` writes ``credentials/<profile>.json`` here and sets no
    environment variable, which is why an unset ``ANTHROPIC_API_KEY`` does not
    mean there is no credential.
    """
    override = os.environ.get("ANTHROPIC_CONFIG_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Anthropic"
    return Path.home() / ".config" / "anthropic"


def _cli_profile() -> str | None:
    """The CLI profile the SDK would use, or None if there is no usable one.

    ``ANTHROPIC_PROFILE`` naming a profile that does not exist is an error rather
    than a fall-through, so it is reported as one instead of being ignored.
    """
    creds = _config_dir() / "credentials"
    named = os.environ.get("ANTHROPIC_PROFILE")
    if named:
        return named if (creds / f"{named}.json").is_file() else None
    try:
        found = sorted(creds.glob("*.json"))
    except OSError:
        return None
    return found[0].stem if found else None


def available() -> tuple[bool, str]:
    """Whether the chat layer can run, and why not when it cannot."""
    chosen = provider()
    if chosen == "ollama":
        return _ollama_available()
    if chosen == "anthropic":
        return _anthropic_available()
    if chosen == "gemini":
        if (os.environ.get("GEMINI_API_KEY") or "").strip():
            return True, ""
        return False, (
            "FORWARD_PILOT_LLM is gemini but GEMINI_API_KEY is not set. Create a key at "
            "aistudio.google.com/apikey, set it in the shell that starts the server, then restart it."
        )
    return False, (
        f"FORWARD_PILOT_LLM is {chosen!r}; it must be one of {', '.join(PROVIDERS)}. "
        "Fix it, then restart the server."
    )


def _gemini_request(path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """POST to Gemini's OpenAI-compatible API; failures become a sentence to act on.

    An overloaded model (503) or a request left hanging is retried once after a
    short pause before giving up, since Google reports both as temporary.
    """
    request = Request(
        GEMINI_BASE_URL + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('GEMINI_API_KEY', '').strip()}"},
    )
    for attempt in range(1, GEMINI_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=GEMINI_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in _GEMINI_TRANSIENT and attempt < GEMINI_ATTEMPTS:
                time.sleep(GEMINI_RETRY_DELAY)
                continue
            raise ChatUnavailable(_gemini_error_text(exc)) from exc
        except TimeoutError as exc:
            if attempt < GEMINI_ATTEMPTS:
                continue
            raise ChatUnavailable(
                f"Gemini did not answer within {GEMINI_TIMEOUT}s, twice. It is probably "
                "overloaded; ask again shortly, or switch to FORWARD_PILOT_LLM=ollama."
            ) from exc
        except URLError as exc:
            raise ChatUnavailable(f"Could not reach the Gemini API ({exc.reason}).") from exc
    raise AssertionError("unreachable")


def _gemini_error_text(exc: HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8"))
        detail = (body[0] if isinstance(body, list) else body).get("error", {}).get("message", "")
    except (ValueError, AttributeError, IndexError):
        detail = ""
    hint = {
        400: "The request was rejected.",
        401: "The API key was rejected; check GEMINI_API_KEY.",
        403: "The API key is not allowed to use this model.",
        404: f"Model {model_name()!r} was not found; set GEMINI_MODEL to a current one.",
        429: "Rate limit or free-tier quota reached; wait and ask again.",
        503: ("Gemini is overloaded right now and stayed so after a retry. Ask again in a "
              "minute, or switch to FORWARD_PILOT_LLM=ollama."),
    }.get(exc.code, "The Gemini API returned an error.")
    return f"{hint} ({exc.code}{': ' + detail if detail else ''})"


def _ollama_available() -> tuple[bool, str]:
    """The server answers and the configured model has been pulled."""
    try:
        tags = _ollama_request("/api/tags", timeout=2)
    except (ChatUnavailable, ValueError):
        return False, (
            f"No Ollama server answered at {_ollama_base_url()}. Install it from "
            "ollama.com and start it with ollama serve, then reload the page."
        )
    model = model_name()
    pulled = sorted(m.get("name", "") for m in tags.get("models", []))
    if model in pulled or f"{model}:latest" in pulled:
        return True, ""
    hint = f" Pulled models: {', '.join(pulled)}." if pulled else ""
    return False, (
        f"Ollama is running but {model!r} has not been pulled. Run ollama pull "
        f"{model}, or set OLLAMA_MODEL to a pulled model with tool support.{hint}"
    )


def _anthropic_available() -> tuple[bool, str]:
    """Whether the Anthropic backend has the SDK and a credential.

    This mirrors the order the SDK itself resolves credentials in --
    ``ANTHROPIC_API_KEY``, then ``ANTHROPIC_AUTH_TOKEN``, then an OAuth profile
    written by ``ant auth login`` -- because a check that only looked at the
    environment would hide the panel from someone who had signed in with the CLI
    and would be served perfectly well by a bare ``Anthropic()`` client.

    Whichever source wins, the credential stays on this side of the wire: the
    browser never sees it, it only talks to this server.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "The anthropic package is not installed. pip install anthropic"

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        value = os.environ.get(var)
        if value is None:
            continue
        if value.strip():
            return True, ""
        # A defined-but-empty variable still wins its slot in the SDK's
        # resolution order, so it authenticates with nothing and shadows a
        # perfectly good CLI profile. Say so: the fix is to unset it, not to
        # add a profile.
        return False, (
            f"{var} is set but empty, which overrides any CLI login and "
            "authenticates with no credential. Unset it, then restart the server."
        )

    if _cli_profile():
        return True, ""

    if os.environ.get("ANTHROPIC_PROFILE"):
        return False, (
            f"ANTHROPIC_PROFILE names {os.environ['ANTHROPIC_PROFILE']!r}, but no such "
            f"profile is stored in {_config_dir()}. Run ant auth login "
            f"--profile {os.environ['ANTHROPIC_PROFILE']}, then restart the server."
        )

    return False, (
        "No Anthropic credentials found. Either run ant auth login (no key to "
        "manage), or set ANTHROPIC_API_KEY. Then restart the server."
    )


# ---- what the agent is told ----------------------------------------------


def _node_directory(network: Mapping[str, Any], limit: int = 40) -> str:
    """Compact id -> name map, so a question can name a city instead of a node id."""
    named = [
        f"{node_id}={node.get('city') or node.get('label')}"
        for node_id, node in network["nodes"].items()
        if node.get("city") or node.get("label")
    ]
    if not named:
        return "Nodes have ids only (n1, n2, ...); there are no place names in this package."
    shown = ", ".join(named[:limit])
    return f"Node names: {shown}{', ...' if len(named) > limit else ''}"


def system_prompt(
    network: Mapping[str, Any],
    shakemap_events: Sequence[Mapping[str, Any]] = (),
    selected: Mapping[str, Any] | None = None,
) -> str:
    hazard = network["default_hazard"]
    models = network["hazard_models"]
    lines = [
        "You help an engineer interrogate an assembled Bayesian network for earthquake "
        "risk to a transport network. You answer by CALLING THE TOOL and reporting what "
        "it returns.",
        "",
        f"Package: {network['title']}. {network['description']}",
        f"{len(network['nodes'])} nodes, {len(network['edges'])} "
        f"{'directed arcs' if network['directed'] else 'undirected links'}. "
        f"Each one carries one highway bridge.",
        f"Destinations (reaching ANY of them counts as service): {', '.join(network['destinations'])}.",
        _node_directory(network),
        "",
        "Hazard representations available in this package:",
    ]
    if "ema_reference_scenario" in models:
        lines.append(
            "  - ema_reference_scenario: set magnitude (Mw) and epicentre_x / epicentre_y. "
            f"Defaults: magnitude {hazard.get('magnitude')}, epicentre "
            f"({hazard.get('epicentre_x')}, {hazard.get('epicentre_y')}). Coordinates are "
            "the dataset's own planar kilometres, not latitude and longitude."
        )
    if "uniform_sa1" in models:
        lines.append(
            "  - uniform_sa1: set sa1_g, one spectral acceleration at 1 s applied to every "
            f"site. Default {hazard.get('sa1_g', 0.3)} g."
        )
    if "shakemap" in models:
        lines.append(
            "  - shakemap: set shakemap_event_id to a USGS event id. The ground motion is the "
            "USGS ShakeMap of that real earthquake, read at each bridge. Use it when the "
            "question names a historical or recent event."
        )
        if shakemap_events:
            lines.append("    Events with a ShakeMap near this network (id: date, magnitude, place):")
            lines += [
                f"    {e['event_id']}: {(e.get('time') or '')[:10]}, M{e.get('magnitude')}, {e.get('place')}"
                for e in shakemap_events[:25]
            ]
    if selected:
        lines += [
            "",
            f"The scenario currently selected in the page is {_describe_hazard(selected)}. "
            "When the question says 'the scenario' or names none, use this one: call the "
            "tool without hazard parameters and it applies it.",
        ]
    lines += [
        "",
        "Rules:",
        "1. Never state a probability, percentage or count that the tool did not return. "
        "If you need a number, call the tool. If the tool did not give it, say so.",
        "2. Report what the tool returns, with the scenario it was run under. Round "
        "sensibly and say what the numbers mean for access to the destinations.",
        "3. To compare scenarios, call the tool once per scenario and compare the results.",
        "4. This model answers FORWARD questions only: given an earthquake, what happens. "
        "That includes ranking bridges by their failure probability under a scenario -- "
        "'which bridges should be inspected after this earthquake?' is forward: run the tool "
        "(set top_n to the number wanted) and list the highest P(unusable). "
        "Backward questions -- which component is most critical given an OBSERVED outage, "
        "diagnosing component states from evidence -- are outside its scope. The tool will "
        "refuse them. Explain why rather than estimating an answer: the network is built "
        "for forward simulation, and a criticality ranking given evidence needs inference "
        "running the other way, which this pilot does not do.",
        "5. Keep replies short -- a few sentences. Each query, written out as a "
        "probability statement, and its headline values are already on screen above "
        "your reply.",
        "6. Say when a result rests on an assumption the package makes, if it matters to "
        "the question. Bridge states are sampled independently given the scenario, so "
        "correlated ground motion is not represented.",
        "7. Your ONLY ability is the query tool. You cannot save, export, download, email or "
        "open files, browse the web, or change the page. If asked to, say plainly that you "
        "cannot, and never claim to have done it. To save this conversation, point to the "
        "'Download conversation' button under the chat box.",
    ]
    return "\n".join(lines)


def _describe_hazard(hazard: Mapping[str, Any]) -> str:
    if hazard.get("model") == "shakemap":
        return f"the USGS ShakeMap of event {hazard.get('event_id')}"
    if hazard.get("model") == "uniform_sa1":
        return f"uniform SA(1 s) = {_num(hazard.get('sa1_g'))} g at every bridge"
    return (f"Mw {_num(hazard.get('magnitude'))} with epicentre "
            f"({_num(hazard.get('epicentre_x'))}, {_num(hazard.get('epicentre_y'))})")


_SCENARIO_KEYS = {"model", "sa1_g", "magnitude", "epicentre_x", "epicentre_y", "event_id"}


def _selected_scenario(network: Mapping[str, Any], scenario: Any) -> dict[str, Any] | None:
    """The page's scenario, if it is one this package can run; otherwise None.

    It comes from the browser, so it is checked rather than trusted: an unknown
    model or stray keys are dropped and the package default stands.
    """
    if not isinstance(scenario, Mapping) or scenario.get("model") not in network.get("hazard_models", ()):
        return None
    if scenario["model"] == "shakemap" and not scenario.get("event_id"):
        return None
    return {k: v for k, v in scenario.items() if k in _SCENARIO_KEYS}


# ---- the query, stated before it is computed ------------------------------


def _node_name(network: Mapping[str, Any], node_id: str) -> str:
    node = network["nodes"].get(node_id) or {}
    name = node.get("city") or node.get("label")
    return f"{name} ({node_id})" if name else str(node_id)


def _num(value: Any) -> str:
    # Not every hazard field is guaranteed to be a number here: a model the
    # package does not carry has no defaults, and the engine has not refused yet.
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def query_statement(network: Mapping[str, Any], query: Mapping[str, Any]) -> dict[str, Any]:
    """The probabilistic query a run asks for, in the model's own variables.

    Built from the query alone, never from a result, so it can be shown before
    the engine is called. It lists every quantity a run produces -- not just the
    one the question was about -- because that is what is computed, and the
    names are the ones ``tbn.model.build`` gives the variables (``S::<origin>``,
    ``X::<edge>``, ``PopFraction``), so the statement can be read against the
    diagram.

    The scenario is written after the bar as what is given. It is an input, not
    a variable, but it is what every quantity here is conditional on.
    """
    hazard = query["hazard"]
    if hazard.get("model") == "shakemap":
        given = f"ShakeMap {hazard.get('event_id')}"
    elif hazard.get("model") == "uniform_sa1":
        given = f"SA1 = {_num(hazard.get('sa1_g'))} g at every bridge"
    else:
        given = (
            f"Mw = {_num(hazard.get('magnitude'))}, epicentre = "
            f"({_num(hazard.get('epicentre_x'))}, {_num(hazard.get('epicentre_y'))})"
        )

    origins = list(query["origins"])
    reach = " or ".join(_node_name(network, d) for d in query["destinations"])
    sampled = f"estimated from {int(query['samples']):,} forward samples"

    if len(origins) == 1:
        service = {
            "expr": f"P(S::{origins[0]} = 1 | {given})",
            "over": None,
            "meaning": f"{_node_name(network, origins[0])} still reaches {reach}",
        }
    else:
        service = {
            "expr": f"P(S::o = 1 | {given})",
            "over": f"each of {len(origins)} origins o",
            "meaning": f"origin o still reaches {reach}",
        }
    targets = [
        {**service, "how": sampled},
        {
            "expr": f"P(X::e = 0 | {given})",
            "over": f"each of {len(network['edges'])} bridges e",
            "meaning": "bridge e is unusable",
            # Both backends report this marginal exactly rather than sampling it.
            "how": "closed form",
        },
    ]
    # Mirrors model.build: no weights, no PopFraction node, so no such query.
    if sum(float((network["nodes"].get(o) or {}).get("population", 0.0)) for o in origins) > 0:
        targets.append({
            "expr": f"E[PopFraction | {given}]",
            "over": None,
            "meaning": "expected share of the population that still has access",
            "how": sampled,
        })
    return {"given": given, "targets": targets}


def _computed_values(network: Mapping[str, Any], result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One headline value per target of ``query_statement``, in the same order."""
    service = result["origin_service"]          # sorted, least reliable first
    worst = service[0]
    if len(service) == 1:
        low, high = worst["ci95"]
        note = f"95% interval {low:.1%}–{high:.1%}"
    else:
        note = f"lowest, at {_node_name(network, worst['origin'])}"
    riskiest = result["components"][0]          # sorted, most likely to fail first
    values = [
        {"value": worst["probability"], "note": note},
        {"value": riskiest["p_unusable"],
         "note": f"highest, at {riskiest['edge_id']} ({riskiest['from']} → {riskiest['to']})"},
    ]
    if "population_accessibility" in result:
        pa = result["population_accessibility"]
        values.append({"value": pa["expected_fraction"],
                       "note": f"of {round(pa['population_in_scope']):,} people in scope"})
    return values


# ---- reusing a run ---------------------------------------------------------
#
# Every chat query uses the same seed, so the same query on the same package
# gives the same result to the last digit: a second "top 10" after a "top 5" is
# the same computation. Keeping recent results lets the engine step be skipped
# without changing any number. A rerun would compute exactly what is stored.

RESULT_CACHE_SIZE = 32
_result_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_result_cache_lock = threading.Lock()


def _result_key(network: Mapping[str, Any], query: Mapping[str, Any]) -> str | None:
    """Everything the result depends on, or None if that cannot be pinned down.

    The package is fingerprinted rather than named, so a rebuilt package is not
    served an old result. A ShakeMap query also records which version of the map
    it would read, since a revised map is a different hazard under the same id.
    """
    parts: dict[str, Any] = {
        "network": hashlib.sha256(json.dumps(network, sort_keys=True, default=str).encode()).hexdigest(),
        "query": query,
    }
    hazard = query.get("hazard") or {}
    if hazard.get("model") == "shakemap":
        from . import shakemap

        try:
            grid = shakemap.load(hazard.get("event_id", ""), hazard.get("source"), hazard.get("version"))
        except shakemap.ShakeMapError:
            return None            # let the engine raise it, with its own message
        parts["shakemap"] = [grid.event_id, grid.version, grid.meta.get("directory")]
    return json.dumps(parts, sort_keys=True, default=str)


def _cached_result(key: str | None) -> dict[str, Any] | None:
    if key is None:
        return None
    with _result_cache_lock:
        result = _result_cache.get(key)
        if result is not None:
            _result_cache.move_to_end(key)
        return result


def _store_result(key: str | None, result: dict[str, Any]) -> None:
    if key is None:
        return
    with _result_cache_lock:
        _result_cache[key] = result
        _result_cache.move_to_end(key)
        while len(_result_cache) > RESULT_CACHE_SIZE:
            _result_cache.popitem(last=False)


def clear_result_cache() -> None:
    with _result_cache_lock:
        _result_cache.clear()


# ---- the tool ------------------------------------------------------------


def run_query_for_chat(
    network: Mapping[str, Any],
    runs: list[dict[str, Any]],
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    hazard_model: str | None = None,
    magnitude: float | None = None,
    epicentre_x: float | None = None,
    epicentre_y: float | None = None,
    sa1_g: float | None = None,
    shakemap_event_id: str | None = None,
    origins: Sequence[str] | None = None,
    destinations: Sequence[str] | None = None,
    samples: int = DEFAULT_SAMPLES,
    top_n: int = 5,
    known_events: Sequence[str] = (),
) -> str:
    """Build a query, run the real engine, record it, and summarise for the model.

    Separate from the decorated tool so it can be tested without the SDK or a
    network call.

    The summary returned to the agent is deliberately small: the full result holds
    129 components and 72 origins, which would cost tokens without helping it
    answer. The complete result goes to ``runs`` for the UI to draw.

    ``on_event`` is told what is happening as it happens: a ``query`` event with
    the probability statement before the engine is called, then ``computed``
    with the headline values, or ``not_computed`` with the reason. One call
    emits exactly one such pair, in that order.
    """
    emit = on_event or (lambda event: None)
    # Naming an event is enough to choose the ShakeMap representation.
    hazard_model = hazard_model or ("shakemap" if shakemap_event_id else None)
    hazard: dict[str, Any] = {"model": hazard_model or network["default_hazard"]["model"]}
    if hazard["model"] == "shakemap":
        hazard["event_id"] = shakemap_event_id or network["default_hazard"].get("event_id")
    elif hazard["model"] == "uniform_sa1":
        hazard["sa1_g"] = network["default_hazard"].get("sa1_g", 0.3) if sa1_g is None else sa1_g
    else:
        base = network["default_hazard"]
        hazard["magnitude"] = base.get("magnitude") if magnitude is None else magnitude
        hazard["epicentre_x"] = base.get("epicentre_x") if epicentre_x is None else epicentre_x
        hazard["epicentre_y"] = base.get("epicentre_y") if epicentre_y is None else epicentre_y

    query = {
        "query_type": "population_accessibility",
        "origins": list(origins) if origins else list(network["default_origins"]),
        "destinations": list(destinations) if destinations else list(network["destinations"]),
        "samples": max(100, min(20000, int(samples))),
        "seed": 20261004,
        "hazard": hazard,
    }

    key = _result_key(network, query)
    stored = _cached_result(key)

    # Stated before the engine runs, not reconstructed from what it returned: a
    # statement written afterwards could only ever agree with the answer.
    emit({"type": "query", "query": query, "statement": query_statement(network, query),
          "reused": stored is not None})

    try:
        result = stored if stored is not None else run_forward_query(network, query)
    except ForwardQueryError as exc:
        # Hand the refusal back as a tool result so it can be explained. The
        # boundary is enforced here, in the engine, not by the prompt.
        emit({"type": "not_computed", "reason": str(exc)})
        refusal: dict[str, Any] = {"refused": True, "reason": str(exc)}
        # A model copying a long id from the prompt can garble a character; name
        # the ids it most likely meant so it can retry rather than give up.
        if hazard["model"] == "shakemap" and shakemap_event_id not in known_events:
            from difflib import get_close_matches

            close = get_close_matches(str(shakemap_event_id), list(known_events), n=3, cutoff=0.6)
            if close:
                refusal["did_you_mean"] = close
        return json.dumps(refusal)
    except (ValueError, KeyError) as exc:
        emit({"type": "not_computed", "reason": f"{type(exc).__name__}: {exc}"})
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    if stored is None:
        _store_result(key, result)      # refusals and errors never reach here, so are never kept
    runs.append({"query": query, "result": result})
    emit({"type": "computed", "values": _computed_values(network, result), "reused": stored is not None})

    # The model sees only these lists, so their length is a parameter: "the top
    # 10 bridges" must be answerable without the model inventing the other five.
    top_n = max(1, min(MAX_TOP_N, int(top_n)))
    worst = result["origin_service"][:top_n]
    riskiest = result["components"][:top_n]
    summary: dict[str, Any] = {
        "scenario": hazard,
        "samples": result["samples"],
        "destinations": result["destinations"],
        "origins_evaluated": len(result["origin_service"]),
        "lowest_reliability_origins": [
            {"origin": o["origin"], "name": o["label"], "p_reaches_destination": round(o["probability"], 4)}
            for o in worst
        ],
        "highest_risk_bridges": [
            {"edge": c["edge_id"], "from": c["from"], "to": c["to"],
             "p_unusable": round(c["p_unusable"], 4), "sa1_g": round(c["sa1_g"], 4)}
            for c in riskiest
        ],
        "bridges_at_or_above_35pc_unusable": sum(
            1 for c in result["components"] if c["p_unusable"] >= 0.35
        ),
        "warnings": result["warnings"],
    }
    if "population_accessibility" in result:
        pa = result["population_accessibility"]
        summary["expected_population_fraction_with_access"] = round(pa["expected_fraction"], 4)
        summary["population_in_scope"] = round(pa["population_in_scope"])
    else:
        summary["population"] = "not available: this package carries no population weights"
    return json.dumps(summary)


# ---- the call ------------------------------------------------------------


_TOOL_NAME = "run_forward_bn_query"
_TOOL_DESCRIPTION = (
    "Run the forward Bayesian-network query for one earthquake scenario. Returns the "
    "probability that each origin still reaches a destination, the per-bridge failure "
    "probabilities, and the expected population share with access. Call once per "
    "scenario; call again to compare scenarios."
)

# The tool's parameters, for backends that take a JSON schema rather than reading
# a docstring. Keep in step with the Args of the Anthropic tool in _ask_anthropic.
_TOOL_PARAMETERS: dict[str, tuple[str, type, str]] = {
    "hazard_model": ("string", str,
                     '"ema_reference_scenario", "uniform_sa1" or "shakemap". Omit for the package default.'),
    "magnitude": ("number", float, "Moment magnitude Mw. Only for ema_reference_scenario."),
    "epicentre_x": ("number", float,
                    "Epicentre x in the dataset's planar kilometres. Only for ema_reference_scenario."),
    "epicentre_y": ("number", float,
                    "Epicentre y in the dataset's planar kilometres. Only for ema_reference_scenario."),
    "sa1_g": ("number", float,
              "Spectral acceleration at 1 s, in g, applied to every site. Only for uniform_sa1."),
    "shakemap_event_id": ("string", str,
                          "USGS event id whose ShakeMap supplies the ground motion. Only for shakemap."),
    "origins": ("array", list, "Node ids to evaluate. Omit for every configured origin."),
    "destinations": ("array", list,
                     "Node ids that count as reaching service. Omit for the package default."),
    "top_n": ("integer", int,
              "How many of the riskiest bridges and least reliable origins to list, 1 to 50. Default 5."),
    "samples": ("integer", int,
                "Monte Carlo samples, 100 to 20000. More samples narrow the interval."),
}


def _tool_schema() -> dict[str, Any]:
    properties = {}
    for name, (kind, _, description) in _TOOL_PARAMETERS.items():
        properties[name] = {"type": kind, "description": description}
        if kind == "array":
            properties[name]["items"] = {"type": "string"}
    return {"type": "function", "function": {
        "name": _TOOL_NAME, "description": _TOOL_DESCRIPTION,
        "parameters": {"type": "object", "properties": properties},
    }}


def _coerce_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Check and convert the arguments a model sent, or raise ``ValueError``.

    Small local models send numbers as strings, a single node id where a list was
    asked for, or a parameter that does not exist. The first two are converted;
    the last is refused rather than dropped, because dropping it would run a
    query other than the one the model meant, and the model can only correct a
    mistake it is told about.
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object")
    unknown = sorted(set(arguments) - set(_TOOL_PARAMETERS))
    if unknown:
        raise ValueError(f"unknown parameter(s) {', '.join(unknown)}; "
                         f"accepted: {', '.join(_TOOL_PARAMETERS)}")
    out: dict[str, Any] = {}
    for name, value in arguments.items():
        if value is None:
            continue
        _, convert, _ = _TOOL_PARAMETERS[name]
        if convert is list:
            out[name] = [str(v) for v in value] if isinstance(value, list) else [str(value)]
        elif convert is int:
            out[name] = int(float(value))
        else:
            out[name] = convert(value)
    return out


def _dispatch_tool_call(name: Any, arguments: Any, call_tool: Callable[..., str]) -> str:
    """Run one tool call a model asked for, or say why it could not be run."""
    if name != _TOOL_NAME:
        return json.dumps({"error": f"no tool named {name!r}; the only tool is {_TOOL_NAME}"})
    try:
        coerced = _coerce_tool_arguments(arguments)
    except (ValueError, TypeError) as exc:
        return json.dumps({"error": f"bad arguments: {exc}"})
    return call_tool(**coerced)


def _strip_thinking(text: str) -> str:
    # Reasoning models served by older Ollama versions leave their thinking inline.
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _shakemap_events(network: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Nearby events for the prompt, or none if the package or the catalogue cannot say.

    Best effort on purpose: without the list the model can still use an id the
    user gives, so a catalogue outage should cost that convenience, not the chat.
    """
    if "shakemap" not in network.get("hazard_models", ()):
        return []
    from . import shakemap

    try:
        return shakemap.search_events(dict(network), min_magnitude=5.0)
    except shakemap.ShakeMapError:
        return []


def ask(
    network: Mapping[str, Any],
    question: str,
    history: Iterable[Mapping[str, str]] = (),
    *,
    client: Any = None,
    model: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    scenario: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer one question by letting the configured model drive the engine.

    ``client`` is injectable so tests can run the loop without a network call:
    an ``anthropic.Anthropic``-shaped object for that backend, and for Ollama a
    callable ``(path, payload) -> dict`` standing in for its HTTP API.
    Returns the reply, every query that was run, and the full result of the last
    one so the page can redraw itself to match the answer.

    ``on_event`` receives the events ``run_query_for_chat`` emits, while the
    loop is still running, so a caller can show each query before its values
    and both before the reply. The return value is unchanged without it.

    ``scenario`` is the hazard currently set in the page. It becomes the default
    for any query that does not set its own, so "given the scenario" means what
    the user is looking at rather than the package's built-in default.
    """
    question = (question or "").strip()
    if not question:
        raise ChatUnavailable("Ask a question first.")

    ok, reason = available()
    if not ok and client is None:
        raise ChatUnavailable(reason)

    model = model or model_name()
    selected = _selected_scenario(network, scenario)
    if selected:
        # Merged, not replaced: a query switching to another model still finds
        # that model's package defaults.
        network = {**network, "default_hazard": {**network["default_hazard"], **selected}}
    runs: list[dict[str, Any]] = []

    known_events: list[str] = []

    def call_tool(**kwargs: Any) -> str:
        return run_query_for_chat(network, runs, on_event=on_event, known_events=known_events, **kwargs)

    messages = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in list(history)[-MAX_TURNS:]
        if turn.get("role") in {"user", "assistant"} and turn.get("content")
    ]
    messages.append({"role": "user", "content": question})

    nearby = _shakemap_events(network)
    known_events.extend(e["event_id"] for e in nearby)
    system = system_prompt(network, nearby, selected)
    if provider() == "anthropic":
        answer = _ask_anthropic(client, model, system, messages, call_tool)
    elif provider() == "gemini":
        answer = _ask_openai_compatible(client or _gemini_request, model, system, messages, call_tool)
    else:
        answer = _ask_ollama(client or _ollama_request, model, system, messages, call_tool)

    return {
        "answer": answer or "No answer was produced.",
        "model": model,
        "runs": [run["query"] for run in runs],
        "result": runs[-1]["result"] if runs else None,
        "ran_query": bool(runs),
    }


def _ask_ollama(
    post: Callable[[str, Mapping[str, Any]], dict[str, Any]],
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    call_tool: Callable[..., str],
) -> str:
    """The tool loop over Ollama's /api/chat: call, run any tools, repeat."""
    conversation: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
    think: bool | None = _ollama_think()
    for _ in range(MAX_TOOL_ROUNDS):
        payload = {
            "model": model,
            "messages": conversation,
            "tools": [_tool_schema()],
            "stream": False,
            "keep_alive": _ollama_keep_alive(),
            "options": {"num_predict": MAX_TOKENS},
        }
        if think is not None:
            payload["think"] = think
        try:
            reply = post("/api/chat", payload)
        except ChatUnavailable as exc:
            # A model without a thinking mode may reject the option outright;
            # it was only ever there to switch thinking off, so drop it and retry.
            if think is None or "think" not in str(exc).lower():
                raise
            think = None
            payload.pop("think")
            reply = post("/api/chat", payload)
        message = reply.get("message") or {}
        calls = message.get("tool_calls") or []
        conversation.append({"role": "assistant", "content": message.get("content") or "",
                             **({"tool_calls": calls} if calls else {})})
        if not calls:
            return _strip_thinking(message.get("content") or "")
        for call in calls:
            function = call.get("function") or {}
            result = _dispatch_tool_call(function.get("name"), function.get("arguments"), call_tool)
            conversation.append({"role": "tool", "tool_name": function.get("name"), "content": result})
    return (f"Stopped after {MAX_TOOL_ROUNDS} rounds of tool calls without a final answer. "
            "The queries that did run are shown above.")


def _ask_openai_compatible(
    post: Callable[[str, Mapping[str, Any]], dict[str, Any]],
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    call_tool: Callable[..., str],
) -> str:
    """The tool loop over an OpenAI-style /chat/completions API (Gemini here)."""
    conversation: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
    for _ in range(MAX_TOOL_ROUNDS):
        reply = post("/chat/completions", {
            "model": model,
            "messages": conversation,
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            # Choosing parameters needs little deliberation; keep latency and quota down.
            "reasoning_effort": "low",
        })
        choices = reply.get("choices") or [{}]
        message = choices[0].get("message") or {}
        calls = message.get("tool_calls") or []
        # Echo the assistant turn back as received: Gemini attaches thought
        # signatures to tool calls and expects to see them again.
        conversation.append({**message, "role": "assistant", "content": message.get("content") or ""})
        if not calls:
            return _strip_thinking(message.get("content") or "")
        for call in calls:
            function = call.get("function") or {}
            result = _dispatch_tool_call(function.get("name"), function.get("arguments"), call_tool)
            conversation.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})
    return (f"Stopped after {MAX_TOOL_ROUNDS} rounds of tool calls without a final answer. "
            "The queries that did run are shown above.")


def _ask_anthropic(
    client: Any,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    call_tool: Callable[..., str],
) -> str:
    """The tool loop through the Anthropic SDK's tool runner."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    from anthropic import beta_tool

    @beta_tool
    def run_forward_bn_query(
        hazard_model: str | None = None,
        magnitude: float | None = None,
        epicentre_x: float | None = None,
        epicentre_y: float | None = None,
        sa1_g: float | None = None,
        shakemap_event_id: str | None = None,
        origins: list[str] | None = None,
        destinations: list[str] | None = None,
        samples: int = DEFAULT_SAMPLES,
        top_n: int = 5,
    ) -> str:
        """Run the forward Bayesian-network query for one earthquake scenario.

        Returns the probability that each origin still reaches a destination, the
        per-bridge failure probabilities, and the expected population share with
        access. Call once per scenario; call again to compare scenarios.

        Args:
            hazard_model: "ema_reference_scenario", "uniform_sa1" or "shakemap". Omit for the package default.
            magnitude: Moment magnitude Mw. Only for ema_reference_scenario.
            epicentre_x: Epicentre x in the dataset's planar kilometres. Only for ema_reference_scenario.
            epicentre_y: Epicentre y in the dataset's planar kilometres. Only for ema_reference_scenario.
            sa1_g: Spectral acceleration at 1 s, in g, applied to every site. Only for uniform_sa1.
            shakemap_event_id: USGS event id whose ShakeMap supplies the ground motion. Only for shakemap.
            origins: Node ids to evaluate. Omit for every configured origin.
            destinations: Node ids that count as reaching service. Omit for the package default.
            samples: Monte Carlo samples, 100 to 20000. More samples narrow the interval.
            top_n: How many of the riskiest bridges and least reliable origins to list, 1 to 50. Default 5.
        """
        return call_tool(
            hazard_model=hazard_model, magnitude=magnitude,
            epicentre_x=epicentre_x, epicentre_y=epicentre_y, sa1_g=sa1_g,
            shakemap_event_id=shakemap_event_id,
            origins=origins, destinations=destinations, samples=samples, top_n=top_n,
        )

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=[run_forward_bn_query],
        messages=messages,
    )

    final = None
    for message in runner:
        final = message

    if final is None:
        return ""
    return "\n\n".join(
        block.text for block in final.content if getattr(block, "type", None) == "text"
    ).strip()
