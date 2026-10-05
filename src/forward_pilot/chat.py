"""Natural-language front door to the forward BN query.

The division of labour is the whole design, and it is not negotiable:

    Claude chooses the query.  The engine computes the answer.

Claude reads a question, picks the scenario parameters, and calls the same
``run_forward_query`` the UI calls. The numbers in its reply are the numbers the
engine returned. It is never asked to estimate a probability itself, because it
would produce a plausible one and nothing downstream could catch it -- the same
reason ``hazard/gmpe`` refuses to invent coefficients.

Two properties follow from routing through the real engine rather than around it:

* The forward-only boundary still holds. ``_validate_query`` raises
  ``ForwardQueryError`` for diagnostic and criticality queries whatever Claude
  asks for, and the tool hands that refusal back so it can be explained rather
  than silently worked around.
* Every answer carries the query that produced it, so a reader can check the
  translation from question to parameters instead of trusting it.

The order is part of that second property. Each query is written out as a
probability statement in the model's own variables and reported *before* the
engine is called, then the values, then the prose. A reader sees what is about
to be computed while it can still be read as a claim to check, rather than as a
footnote to an answer they have already accepted.

This module is optional. With no credentials configured the rest of the pilot
works exactly as before; ``available()`` reports why, and the UI hides the panel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .engine import ForwardQueryError, run_forward_query

MODEL = "claude-opus-5"
MAX_TOKENS = 4000
DEFAULT_SAMPLES = 4000
MAX_TURNS = 12          # cap the stored history so a long session cannot grow unbounded


class ChatUnavailable(RuntimeError):
    """The natural-language layer is not configured or not installed."""


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
    """Whether the chat layer can run, and why not when it cannot.

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


# ---- what Claude is told -------------------------------------------------


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


def system_prompt(network: Mapping[str, Any]) -> str:
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
    lines += [
        "",
        "Rules:",
        "1. Never state a probability, percentage or count that the tool did not return. "
        "If you need a number, call the tool. If the tool did not give it, say so.",
        "2. Report what the tool returns, with the scenario it was run under. Round "
        "sensibly and say what the numbers mean for access to the destinations.",
        "3. To compare scenarios, call the tool once per scenario and compare the results.",
        "4. This model answers FORWARD questions only: given an earthquake, what happens. "
        "Backward questions -- which component is most critical given an observed outage, "
        "diagnosing component states from evidence -- are outside its scope. The tool will "
        "refuse them. Explain why rather than estimating an answer: the network is built "
        "for forward simulation, and a criticality ranking needs inference running the "
        "other way, which this pilot does not do.",
        "5. Keep replies short -- a few sentences. Each query, written out as a "
        "probability statement, and its headline values are already on screen above "
        "your reply.",
        "6. Say when a result rests on an assumption the package makes, if it matters to "
        "the question. Bridge states are sampled independently given the scenario, so "
        "correlated ground motion is not represented.",
    ]
    return "\n".join(lines)


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
    if hazard.get("model") == "uniform_sa1":
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
    origins: Sequence[str] | None = None,
    destinations: Sequence[str] | None = None,
    samples: int = DEFAULT_SAMPLES,
) -> str:
    """Build a query, run the real engine, record it, and summarise for the model.

    Separate from the decorated tool so it can be tested without the SDK or a
    network call.

    The summary returned to Claude is deliberately small: the full result holds
    129 components and 72 origins, which would cost tokens without helping it
    answer. The complete result goes to ``runs`` for the UI to draw.

    ``on_event`` is told what is happening as it happens: a ``query`` event with
    the probability statement before the engine is called, then ``computed``
    with the headline values, or ``not_computed`` with the reason. One call
    emits exactly one such pair, in that order.
    """
    emit = on_event or (lambda event: None)
    hazard: dict[str, Any] = {"model": hazard_model or network["default_hazard"]["model"]}
    if hazard["model"] == "uniform_sa1":
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

    # Stated before the engine runs, not reconstructed from what it returned: a
    # statement written afterwards could only ever agree with the answer.
    emit({"type": "query", "query": query, "statement": query_statement(network, query)})

    try:
        result = run_forward_query(network, query)
    except ForwardQueryError as exc:
        # Hand the refusal back as a tool result so it can be explained. The
        # boundary is enforced here, in the engine, not by the prompt.
        emit({"type": "not_computed", "reason": str(exc)})
        return json.dumps({"refused": True, "reason": str(exc)})
    except (ValueError, KeyError) as exc:
        emit({"type": "not_computed", "reason": f"{type(exc).__name__}: {exc}"})
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    runs.append({"query": query, "result": result})
    emit({"type": "computed", "values": _computed_values(network, result)})

    worst = result["origin_service"][:5]
    riskiest = result["components"][:5]
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


def ask(
    network: Mapping[str, Any],
    question: str,
    history: Iterable[Mapping[str, str]] = (),
    *,
    client: Any = None,
    model: str = MODEL,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Answer one question by letting Claude drive the engine.

    ``client`` is injectable so tests can run the loop without a network call.
    Returns the reply, every query that was run, and the full result of the last
    one so the page can redraw itself to match the answer.

    ``on_event`` receives the events ``run_query_for_chat`` emits, while the
    loop is still running, so a caller can show each query before its values
    and both before the reply. The return value is unchanged without it.
    """
    question = (question or "").strip()
    if not question:
        raise ChatUnavailable("Ask a question first.")

    ok, reason = available()
    if not ok and client is None:
        raise ChatUnavailable(reason)

    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    from anthropic import beta_tool

    runs: list[dict[str, Any]] = []

    @beta_tool
    def run_forward_bn_query(
        hazard_model: str | None = None,
        magnitude: float | None = None,
        epicentre_x: float | None = None,
        epicentre_y: float | None = None,
        sa1_g: float | None = None,
        origins: list[str] | None = None,
        destinations: list[str] | None = None,
        samples: int = DEFAULT_SAMPLES,
    ) -> str:
        """Run the forward Bayesian-network query for one earthquake scenario.

        Returns the probability that each origin still reaches a destination, the
        per-bridge failure probabilities, and the expected population share with
        access. Call once per scenario; call again to compare scenarios.

        Args:
            hazard_model: "ema_reference_scenario" or "uniform_sa1". Omit for the package default.
            magnitude: Moment magnitude Mw. Only for ema_reference_scenario.
            epicentre_x: Epicentre x in the dataset's planar kilometres. Only for ema_reference_scenario.
            epicentre_y: Epicentre y in the dataset's planar kilometres. Only for ema_reference_scenario.
            sa1_g: Spectral acceleration at 1 s, in g, applied to every site. Only for uniform_sa1.
            origins: Node ids to evaluate. Omit for every configured origin.
            destinations: Node ids that count as reaching service. Omit for the package default.
            samples: Monte Carlo samples, 100 to 20000. More samples narrow the interval.
        """
        return run_query_for_chat(
            network, runs, on_event=on_event,
            hazard_model=hazard_model, magnitude=magnitude,
            epicentre_x=epicentre_x, epicentre_y=epicentre_y, sa1_g=sa1_g,
            origins=origins, destinations=destinations, samples=samples,
        )

    messages = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in list(history)[-MAX_TURNS:]
        if turn.get("role") in {"user", "assistant"} and turn.get("content")
    ]
    messages.append({"role": "user", "content": question})

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt(network),
        tools=[run_forward_bn_query],
        messages=messages,
    )

    final = None
    for message in runner:
        final = message

    answer = ""
    if final is not None:
        answer = "\n\n".join(
            block.text for block in final.content if getattr(block, "type", None) == "text"
        ).strip()

    return {
        "answer": answer or "No answer was produced.",
        "model": model,
        "runs": [run["query"] for run in runs],
        "result": runs[-1]["result"] if runs else None,
        "ran_query": bool(runs),
    }
