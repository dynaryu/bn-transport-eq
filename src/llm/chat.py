"""Translate between natural-language questions and probabilistic queries.

The division of labour is the design, and it is not negotiable:

    The language model chooses the query.  The engine computes the answer.

The model reads a question and calls one tool, ``run_query``, whose argument is
a query in ``src/inference/query.schema.json``. The engine validates and
computes it, and the numbers in the reply are the engine's. The model is never
asked to estimate a probability itself: it would produce a plausible one, and
nothing downstream could catch it.

Two properties follow from routing through the engine rather than around it:

* The forward-only boundary holds. ``engine.answer`` refuses a query with
  evidence whoever sent it, and the refusal goes back to the model to explain.
* Every answer carries the queries that produced it, each written as a
  probability statement and reported *before* the engine runs, then its values,
  then the prose. A reader checks the translation while it can still be read as
  a claim, rather than as a footnote to an answer already accepted.

Backends, chosen by ``BN_LLM`` (or ``serve --llm``), with ``BN_MODEL`` (or
``serve --model``) for another model of that backend:

* ``ollama`` (the default): a local model served by Ollama, over its HTTP API
  with the standard library alone. ``OLLAMA_BASE_URL`` overrides the address.
* ``gemini``: Google Gemini through its OpenAI-compatible API. Needs ``GEMINI_API_KEY``.
* ``anthropic``: Claude through the ``anthropic`` SDK and a Console credential.

With no backend available the rest of the app works unchanged; ``available()``
says why, and the page hides the panel.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any, Callable, Iterable, Mapping

from src.inference import engine


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
ANTHROPIC_MODEL = "claude-opus-5-5"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 300    # seconds per model call; a CPU-only 8B model can be slow
OLLAMA_KEEP_ALIVE = "30m"   # how long Ollama keeps the model loaded after a request
MAX_TOOL_ROUNDS = 8     # a small model can loop on tool calls; stop it here
MAX_TOKENS = 4000
DEFAULT_SAMPLES = 2000
MAX_REPORTED = 60       # longest list of node results handed back to the model
MAX_TURNS = 12          # cap the stored history so a long session cannot grow unbounded


class ChatUnavailable(RuntimeError):
    """The natural-language layer is not configured or not installed."""


def provider() -> str:
    """Which backend drives the chat: ``BN_LLM``, Ollama by default."""
    return (os.environ.get("BN_LLM") or "ollama").strip().lower()


DEFAULT_MODELS = {"ollama": OLLAMA_MODEL, "gemini": GEMINI_MODEL, "anthropic": ANTHROPIC_MODEL}


def model_name() -> str:
    """``BN_MODEL``, else the backend's own variable (``OLLAMA_MODEL``,
    ``GEMINI_MODEL``, ``ANTHROPIC_MODEL``), else its default."""
    chosen = provider()
    return (os.environ.get("BN_MODEL") or os.environ.get(f"{chosen.upper()}_MODEL")
            or DEFAULT_MODELS.get(chosen, OLLAMA_MODEL))


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
            "(serve --model) or a GPU will help; the query form still works."
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
            "BN_LLM is gemini but GEMINI_API_KEY is not set. Create a key at "
            "aistudio.google.com/apikey, set it in the shell that starts the server, then restart it."
        )
    return False, (
        f"BN_LLM is {chosen!r}; it must be one of {', '.join(PROVIDERS)}. "
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
                "overloaded; ask again shortly, or switch to BN_LLM=ollama."
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
        404: f"Model {model_name()!r} was not found; start the server with --model and a current one.",
        429: "Rate limit or free-tier quota reached; wait and ask again.",
        503: ("Gemini is overloaded right now and stayed so after a retry. Ask again in a "
              "minute, or switch to BN_LLM=ollama."),
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
        f"{model}, or start the server with --model and a pulled model with tool support.{hint}"
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
        return False, ("The anthropic package is not installed. With conda, run conda env update "
                       "-f environment.yml; with uv, uv sync --extra anthropic. Then restart the server.")

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




# ---- what the model is told ------------------------------------------------


TOOL_NAME = "run_query"
_OPS = ["<", "<=", ">", ">=", "==", "!="]


def _quantity_line(q: Mapping[str, Any]) -> str:
    if q["kind"] == "discrete":
        kind = "states " + ", ".join(map(str, q["states"]))
    elif q["kind"] == "record":
        kind = "record of " + ", ".join(q["fields"] or [])
    else:
        kind = f"continuous, in {q['unit']}"
    return f"- {q['id']} on {q['entity']} ({q['nodes']} nodes, e.g. {q['example']}): {q['label']}; {kind}"


def _examples(menu: Mapping[str, Any]) -> list[str]:
    by_id = {q["id"]: q for q in menu["quantities"]}
    out = []
    if "TravelTime" in by_id:
        node = by_id["TravelTime"]["example"]
        out.append(f'{{"events": [{{"node": "{node}", "op": "==", "value": "inf"}}]}}  '
                   f"= P(no open route for {node.partition('::')[2]})")
        out.append(f'{{"targets": ["{node}"]}}  = the distribution of that travel time')
    if "MaxFlow" in by_id:
        out.append('{"targets": ["MaxFlow::*"]}  = max flow for every max-flow event')
    if "SystemState" in by_id:
        out.append('{"events": [{"node": "SystemState::*", "op": "==", "value": 0}]}  '
                   '= P(each system event fails, by its own definition)')
    if "EdgeOpen" in by_id:
        out.append('{"events": [{"node": "EdgeOpen::*", "op": "==", "value": 0}]}  = P(each edge closed)')
    return out


def system_prompt(menu: Mapping[str, Any], defaults: Mapping[str, Any] | None = None) -> str:
    settable = menu["settable"]
    lines = [
        "You translate questions about a transport network's earthquake risk into probabilistic "
        f"queries, run them with the {TOOL_NAME} tool, and translate the results back into plain language.",
        "",
        "Rules:",
        f"- Every number you state comes from a {TOOL_NAME} result. Never estimate, extrapolate or invent one.",
        "- Say in words which query you ran before giving its numbers.",
        "- The engine is forward-only: scenario in, distribution out. A question conditioned on an "
        "observation (which bridge failed, given that a route is closed) cannot be computed. Say so "
        "rather than answering a different question; the engine refuses evidence if you pass it.",
        "- Results are Monte Carlo estimates: give probabilities to two significant figures, with the "
        "95% interval when it matters.",
        "- Keep answers short.",
        "",
        f"The Bayesian network is {menu['assembly']}: {menu['label']}. {menu['description']}".strip(),
        f"Network {menu['network']['id']} ({menu['network']['title']}): "
        + ", ".join(f"{n} {k}" for k, n in menu["network"]["rows"].items()) + ".",
    ]
    if len(menu["alternatives"]) > 1:
        lines.append("Other BNs on this network, chosen with the assembly argument: "
                     + "; ".join(f"{a['id']} ({a['label']})" for a in menu["alternatives"]
                                 if a["id"] != menu["assembly"]) + ".")
    lines += ["", "Nodes are named <Quantity>::<row id>; * in a target or event node matches any characters.",
              *(_quantity_line(q) for q in menu["quantities"])]
    lines += ["", "Scenario settings (do-interventions):"]
    for r in settable["rows"]:
        lines.append(f'- rows: {{"{r["row"]}": {{"{r["attribute"]}": <{r["type"]}>}}}}, '
                     f'now {r["value"]} {r["unit"] or ""}'.rstrip())
    for p in settable["parameters"]:
        lines.append(f'- parameters: {{"{p["function"]}": {{"{p["name"]}": ...}}}}, now {json.dumps(p["value"])}')
    for c in settable["parameter_sets"]:
        lines.append(f'- parameter_sets: {{"{c["function"]}": one of {", ".join(s["id"] for s in c["choices"])}}}, '
                     f'now {c["current"]}')
    lines.append("- any other row attribute is set the same way, e.g. a Bridge's hazus_class.")
    if menu["names"]:
        lines += ["", "Node names: " + "; ".join(f"{k} {v}" for k, v in menu["names"].items())]
    lines += ["", "Examples of tool arguments:", *_examples(menu)]
    if defaults:
        lines += ["", "The page currently sets " + json.dumps(defaults)
                  + ". These apply to every query unless the question changes them."]
    return "\n".join(lines)


def tool_spec(menu: Mapping[str, Any]) -> dict[str, Any]:
    """The tool's JSON schema: deliberately plainer than the query schema, which the engine enforces."""
    obj = {"type": "object"}
    return {
        "name": TOOL_NAME,
        "description": ("Run one probabilistic query on the Bayesian network and return per-node summaries "
                        "(targets) and event probabilities with 95% intervals (events). Call again to "
                        "compare scenarios."),
        "parameters": {"type": "object", "properties": {
            "assembly": {"type": "string", "enum": [a["id"] for a in menu["alternatives"]],
                         "description": f"Which BN. Omit for {menu['assembly']}."},
            "targets": {"type": "array", "items": {"type": "string"},
                        "description": "Node ids or patterns to summarise, e.g. TravelTime::n1>*."},
            "events": {"type": "array", "description": "Threshold events whose probability to compute.",
                       "items": {"type": "object", "required": ["node", "op", "value"], "properties": {
                           "node": {"type": "string"}, "op": {"type": "string", "enum": _OPS},
                           "value": {"type": "string", "description": "A number, inf, or a state name."}}}},
            "rows": {**obj, "description": 'Set row attributes: {"<row id>": {"<attribute>": value}}.'},
            "parameters": {**obj, "description": 'Set inline parameters: {"<function id>": {"<name>": value}}.'},
            "parameter_sets": {**obj, "description": 'Swap parameter sets: {"<function id>": "<set id>"}.'},
            "evidence": {**obj, "description": "Observed node values. Not supported: the engine refuses them."},
            "samples": {"type": "integer",
                        "description": f"Monte Carlo samples, 100 to 20000. Default {DEFAULT_SAMPLES}."},
        }},
    }


# ---- one tool call ---------------------------------------------------------

_ARGUMENTS = {"assembly", "targets", "events", "rows", "parameters", "parameter_sets", "evidence", "samples"}


def coerce_arguments(arguments: Any, assembly: str, defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The query a model's arguments mean, or ``ValueError``.

    Small models send numbers as strings and one node where a list was asked
    for; those are converted. An unknown argument is refused rather than dropped,
    because dropping it would run a query other than the one meant.
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object")
    unknown = sorted(set(arguments) - _ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown argument(s) {', '.join(unknown)}; accepted: {', '.join(sorted(_ARGUMENTS))}")
    query = {k: v for k, v in arguments.items() if v not in (None, {}, [])}
    query.setdefault("assembly", assembly)
    if isinstance(query.get("targets"), str):
        query["targets"] = [query["targets"]]
    if isinstance(query.get("events"), dict):
        query["events"] = [query["events"]]
    for event in query.get("events", []):
        if isinstance(event, dict) and isinstance(event.get("value"), str) and event["value"].lower() != "inf":
            try:
                number = float(event["value"])
            except ValueError:
                continue                               # a state name stays a string
            event["value"] = int(number) if number.is_integer() else number
    query["samples"] = int(float(query["samples"])) if "samples" in query else DEFAULT_SAMPLES
    if defaults and query["assembly"] == assembly:
        for key in ("rows", "parameters", "parameter_sets"):
            if defaults.get(key) and key not in query:
                query[key] = defaults[key]
    return query


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.4g}")
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def compact(result: Mapping[str, Any]) -> dict[str, Any]:
    """What the model sees of a result: rounded, and capped at MAX_REPORTED entries per list."""
    out: dict[str, Any] = {"statement": result["statement"], "samples": result["samples"]}
    for key in ("events", "targets"):
        items = result[key]
        out[key] = _round([{k: v for k, v in item.items() if k not in ("quantity", "row")}
                           for item in items[:MAX_REPORTED]])
        if len(items) > MAX_REPORTED:
            out[f"{key}_omitted"] = len(items) - MAX_REPORTED
    if result["warnings"]:
        out["warnings"] = result["warnings"]
    return out


def run_tool(arguments: Any, *, assembly: str, defaults: Mapping[str, Any] | None,
             runs: list[dict[str, Any]], on_event: Callable[[dict[str, Any]], None] | None = None) -> str:
    """Run one tool call: state the query, compute it, report it. Returns what the model reads."""
    emit = on_event or (lambda event: None)
    try:
        query = coerce_arguments(arguments, assembly, defaults)
    except (ValueError, TypeError) as exc:
        return json.dumps({"error": f"bad arguments: {exc}"})
    stated = engine.statement(query)
    emit({"type": "query", "query": query, "statement": stated})
    try:
        result = engine.answer(query)
    except engine.QueryError as exc:
        runs.append({"query": query, "statement": stated, "error": str(exc)})
        emit({"type": "refused", "query": query, "statement": stated, "error": str(exc)})
        return json.dumps({"refused": True, "error": str(exc)})
    runs.append({"query": query, "statement": result["statement"], "result": result})
    emit({"type": "computed", "statement": result["statement"], "result": result})
    return json.dumps(compact(result))


def _dispatch(name: Any, arguments: Any, call: Callable[[Any], str]) -> str:
    if name != TOOL_NAME:
        return json.dumps({"error": f"no tool named {name!r}; the only tool is {TOOL_NAME}"})
    return call(arguments)


def _strip_thinking(text: str) -> str:
    # Reasoning models served by older Ollama versions leave their thinking inline.
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


# ---- the conversation ------------------------------------------------------


def ask(
    assembly: str,
    question: str,
    history: Iterable[Mapping[str, str]] = (),
    *,
    defaults: Mapping[str, Any] | None = None,
    client: Any = None,
    model: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Answer one question about ``assembly`` by letting the configured model drive the engine.

    ``defaults`` are the interventions currently set on the page, so "under this
    scenario" means what the user is looking at. ``client`` is injectable for
    tests: a callable ``(path, payload) -> dict`` standing in for Ollama's or
    Gemini's HTTP API, or an ``anthropic.Anthropic``-shaped object.
    """
    question = (question or "").strip()
    if not question:
        raise ChatUnavailable("Ask a question first.")
    ok, reason = available()
    if not ok and client is None:
        raise ChatUnavailable(reason)

    menu = engine.describe(assembly)
    model = model or model_name()
    system = system_prompt(menu, defaults)
    spec = tool_spec(menu)
    runs: list[dict[str, Any]] = []

    def call(arguments: Any) -> str:
        return run_tool(arguments, assembly=assembly, defaults=defaults, runs=runs, on_event=on_event)

    messages = [{"role": turn["role"], "content": turn["content"]}
                for turn in list(history)[-MAX_TURNS:]
                if turn.get("role") in {"user", "assistant"} and turn.get("content")]
    messages.append({"role": "user", "content": question})

    if provider() == "anthropic":
        answer = _ask_anthropic(client, model, system, messages, spec, call)
    elif provider() == "gemini":
        answer = _ask_openai_compatible(client or _gemini_request, model, system, messages, spec, call)
    else:
        answer = _ask_ollama(client or _ollama_request, model, system, messages, spec, call)

    computed = [r for r in runs if "result" in r]
    return {
        "answer": answer or "No answer was produced.",
        "model": model,
        "runs": [{k: v for k, v in r.items() if k != "result"} for r in runs],
        "result": computed[-1]["result"] if computed else None,
    }


def _stopped() -> str:
    return (f"Stopped after {MAX_TOOL_ROUNDS} rounds of tool calls without a final answer. "
            "The queries that did run are shown above.")


def _ask_ollama(post: Callable[[str, Mapping[str, Any]], dict[str, Any]], model: str, system: str,
                messages: list[dict[str, Any]], spec: dict[str, Any], call: Callable[[Any], str]) -> str:
    """The tool loop over Ollama's /api/chat: call, run any tools, repeat."""
    conversation: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
    think: bool | None = _ollama_think()
    for _ in range(MAX_TOOL_ROUNDS):
        payload = {"model": model, "messages": conversation, "tools": [{"type": "function", "function": spec}],
                   "stream": False, "keep_alive": _ollama_keep_alive(), "options": {"num_predict": MAX_TOKENS}}
        if think is not None:
            payload["think"] = think
        try:
            reply = post("/api/chat", payload)
        except ChatUnavailable as exc:
            # A model without a thinking mode may reject the option; it only ever switched thinking off.
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
        for tool_call in calls:
            function = tool_call.get("function") or {}
            result = _dispatch(function.get("name"), function.get("arguments"), call)
            conversation.append({"role": "tool", "tool_name": function.get("name"), "content": result})
    return _stopped()


def _ask_openai_compatible(post: Callable[[str, Mapping[str, Any]], dict[str, Any]], model: str, system: str,
                           messages: list[dict[str, Any]], spec: dict[str, Any], call: Callable[[Any], str]) -> str:
    """The tool loop over an OpenAI-style /chat/completions API (Gemini here)."""
    conversation: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
    for _ in range(MAX_TOOL_ROUNDS):
        reply = post("/chat/completions", {"model": model, "messages": conversation,
                                           "tools": [{"type": "function", "function": spec}],
                                           "tool_choice": "auto", "reasoning_effort": "low"})
        message = (reply.get("choices") or [{}])[0].get("message") or {}
        calls = message.get("tool_calls") or []
        # Echoed back as received: Gemini attaches thought signatures to tool calls and expects them again.
        conversation.append({**message, "role": "assistant", "content": message.get("content") or ""})
        if not calls:
            return _strip_thinking(message.get("content") or "")
        for tool_call in calls:
            function = tool_call.get("function") or {}
            result = _dispatch(function.get("name"), function.get("arguments"), call)
            conversation.append({"role": "tool", "tool_call_id": tool_call.get("id"), "content": result})
    return _stopped()


def _ask_anthropic(client: Any, model: str, system: str, messages: list[dict[str, Any]],
                   spec: dict[str, Any], call: Callable[[Any], str]) -> str:
    """The tool loop over the Messages API: answer, or run the tools asked for and continue."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    tools = [{"name": spec["name"], "description": spec["description"], "input_schema": spec["parameters"]}]
    conversation = list(messages)
    for _ in range(MAX_TOOL_ROUNDS):
        reply = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=system,
                                       tools=tools, messages=conversation)
        conversation.append({"role": "assistant", "content": reply.content})
        uses = [block for block in reply.content if getattr(block, "type", None) == "tool_use"]
        if not uses:
            return "\n\n".join(block.text for block in reply.content
                               if getattr(block, "type", None) == "text").strip()
        conversation.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": block.id, "content": _dispatch(block.name, block.input, call)}
            for block in uses]})
    return _stopped()
