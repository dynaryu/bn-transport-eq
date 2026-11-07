"""A session's decision query or observation, drafted from what the questioner writes, by the configured
language model: the Decision query and Evidence panels' agent.

The model gets the session as background (the BN's observable and decidable variables, the towns and
edges, the evidence and the decision queries so far), the panel's conversation so far, and two tools,
draft_query or draft_evidence, and clarify. The tool checks the draft's names against the session and hands the
problems back, so the model corrects them; or, when the sentence leaves the meaning open, the model
asks instead (clarification, with choices), as Ask did. The draft it settles on is shown in the panel,
where the questioner reviews it before adding it. Nothing is added to the session here, and the
model computes nothing: every number comes from the session, through src/inference/cliques.py.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from . import chat

QUERY_TOOL = {
    "name": "draft_query",
    "description": "Draft a decision query: what the decision is about (its target) and what kind of decision it is.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "A short name for the decision, in the questioner's terms."},
            "target": {"type": "object", "properties": {
                "kind": {"type": "string", "enum": ["cut_off", "cut_off_at_least", "fail", "closed"]},
                "origin": {"type": "string", "description": "the town, for kind fail"},
                "edge": {"type": "string", "description": "the edge, for kind closed"},
                "at_least": {"type": "number", "description": "for kind cut_off_at_least"}},
                "required": ["kind"]},
            "decision": {"type": "string", "enum": ["inspect", "track"],
                         "description": "inspect: which bridge to inspect first (and repair if damaged); track: the target itself"},
        },
    },
}
CLARIFY_TOOL = {
    "name": "clarify",
    "description": "Instead of a draft, ask what the sentence leaves open (which town, bridge or reading is meant), "
                   "with two to four short choices.",
    "parameters": {"type": "object", "required": ["question", "choices"], "properties": {
        "question": {"type": "string"},
        "choices": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string"}}}},
}


def _clarification(args: Mapping[str, Any]) -> dict[str, Any] | None:
    """The page's question with choices, from what the model sent: choices as plain text, or as objects naming
    a label (or name) and perhaps a definition. None if there is no question or fewer than two choices."""
    choices = []
    for c in args.get("choices") or []:
        if isinstance(c, Mapping):
            label = c.get("label") or c.get("name") or c.get("definition")
            choices.append({"label": str(label), "definition": str(c.get("definition") or "")} if label else None)
        else:
            choices.append({"label": str(c), "definition": ""})
    choices = [c for c in choices if c]
    question = str(args.get("question") or "").strip()
    return {"question": question, "choices": choices[:4]} if question and len(choices) >= 2 else None
EVIDENCE_TOOL = {
    "name": "draft_evidence",
    "description": "Draft one observation: the variable observed, its value, and who reported it.",
    "parameters": {
        "type": "object",
        "properties": {
            "variable": {"type": "string", "description": "the observed variable's full name, as listed"},
            "value": {"type": ["string", "number"], "description": "a state as listed, or a reading in g"},
            "source": {"type": "string", "description": "who or what reported it, if said"},
        },
    },
}


def _background(state: Mapping[str, Any], network: Mapping[str, Any]) -> str:
    kinds: dict[str, list[str]] = {}
    for name, v in state["variables"].items():
        kinds.setdefault(v["kind"], []).append(name)
    states = {k: next((v["states"] for n, v in state["variables"].items() if v["kind"] == k), None) for k in kinds}
    lines = [f"Network: {network.get('title', '')}. {network.get('description', '')}",
             *(f"- {k}: {v}" for k, v in (network.get("assumptions") or {}).items() if k in ("key_destination", "system_event", "functionality")),
             f"Towns (origins whose access is a system state): {', '.join(state['origins'])}",
             "Edges (end nodes; bridge): " + "; ".join(
                 f"{e['id']} {e['ends'][0]}-{e['ends'][1]}" + (f" ({', '.join(e['bridges'])})" if e["bridges"] else "")
                 for e in network.get("edges", [])) if network.get("edges") else f"Edges: {', '.join(state['edges'])}",
             f"L counts {state['unit']} cut off from a key destination."]
    for kind, names in kinds.items():
        lines.append(f"{kind} variables ({'reading in g' if states[kind] is None else 'states: ' + ', '.join(states[kind])}): "
                     + ", ".join(names))
    lines.append("Decision query targets (kind: meaning): " + "; ".join(f"{k}: {v}" for k, v in state["targets"].items())
                 + ". fail takes origin (a town); closed takes edge; cut_off_at_least takes at_least. Decisions: "
                 + "; ".join(f"{k}: {v}" for k, v in state["decisions"].items()) + ".")
    lines.append("Evidence so far: " + (", ".join(f"{v} = {x['label']}" for v, x in state["evidence"].items()) or "none"))
    lines.append("Decision queries so far: " + ("; ".join(
        f"{q['name']} (target {q['target']}; decision {q['decision']})" for q in state["queries"]) or "none"))
    return "\n".join(lines)


def _check(kind: str, args: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """The draft in the page's shape, or the problems with it."""
    variables, problems = state["variables"], []

    def state_of(var, value):
        if var not in variables:
            problems.append(f"no variable {var!r}; use a full name from the list")
            return None
        st = variables[var]["states"]
        if st is None:
            try:
                return float(value)
            except (TypeError, ValueError):
                problems.append(f"{var} is a reading in g, not {value!r}")
                return None
        if str(value) not in st:
            problems.append(f"{var} has states {st}, not {value!r}")
        return str(value)

    if kind == "evidence":
        var = args.get("variable", "")
        if var in state["evidence"]:
            problems.append(f"{var} is already observed")
        value = state_of(var, args.get("value"))
        return ({"variable": var, "value": value, "source": args.get("source")} if not problems else None), problems
    target = dict(args.get("target") or {})
    kinds = list(state["targets"])
    fields = {"cut_off": set(), "cut_off_at_least": {"at_least"}, "fail": {"origin"}, "closed": {"edge"}}
    if target.get("kind") not in kinds:
        problems.append(f"target needs kind, one of {kinds} (cut_off: the {state['unit']} cut off), not {target!r}")
    elif extra := set(target) - {"kind"} - fields.get(target["kind"], set()):
        problems.append(f"target kind {target['kind']!r} takes {sorted(fields[target['kind']]) or 'no other field'}, not "
                        f"{sorted(extra)}: for one town's chance of being cut off use kind 'fail' with origin; for "
                        f"'at least x {state['unit']} cut off' use kind 'cut_off_at_least' with at_least")
    if target.get("kind") == "fail" and target.get("origin") not in state["origins"]:
        problems.append(f"no town {target.get('origin')!r}; have {state['origins']}")
    if target.get("kind") == "closed" and target.get("edge") not in state["edges"]:
        problems.append(f"no edge {target.get('edge')!r}")
    decision = args.get("decision") or "track"
    if decision not in state["decisions"]:
        problems.append(f"decision is one of {list(state['decisions'])}, not {decision!r}")
    return ({"name": args.get("name"), "target": target, "decision": decision} if not problems else None), problems


def draft(kind: str, text: str, state: Mapping[str, Any], network: Mapping[str, Any], history=(), *,
          client: Any = None) -> dict[str, Any]:
    """{"draft": the drafted entry or None, "clarification": a question with choices or None, "note": the
    model's one-line remark} for the sentence, after the panel's conversation so far (history: role and
    content turns)."""
    if kind not in ("query", "evidence"):
        raise ValueError("kind is query or evidence")
    ok, reason = chat.available()
    if not ok and client is None:
        raise chat.ChatUnavailable(reason)
    spec = QUERY_TOOL if kind == "query" else EVIDENCE_TOOL
    system = (f"You turn an emergency operator's sentence into a {'decision query' if kind == 'query' else 'single observation'} "
              f"for a Bayesian network session, by calling {spec['name']} once with names exactly as listed. "
              + ("A decision query says what the decision is about: its target, the quantity the operator wants to "
                 "improve, and its decision: 'inspect' when the operator wants to know which bridge to inspect (or repair) "
                 "first, 'track' when they want to follow the target itself. For example, 'I would like to prioritise "
                 "inspection to maximise the number of towns with access to the key destinations' is draft_query with name "
                 "'Inspection priority for towns' access', target {kind: 'cut_off'}, decision 'inspect'. Before drafting "
                 "an 'inspect' query, unless the conversation already says, call clarify: 'If an inspected bridge is "
                 "found closed, what will you do?' with choices 'repair it' and 'leave it closed'; draft 'inspect' after "
                 "'repair it', 'track' after 'leave it closed'. "
                 if kind == "query" else
                 "A station reading is an SAObs variable in g; a report that a town cannot reach the key destinations is its "
                 "SystemState Fail; an edge reported closed is its EdgeOpen closed; an inspection gives a bridge's DamageState. ")
              + "If the sentence leaves something open that changes the draft (which town, bridge or reading is meant), "
                "call clarify instead, a short question with two to four choices; never ask in plain text. Use the "
                "edges' end nodes to tell which bridges are at or near a town. If the draft tool reports "
                "problems, correct them and call it again. Then reply in one short sentence saying what you drafted and "
                "any assumption you made.\n\nThe session:\n" + _background(state, network))
    drafted: dict[str, Any] = {}

    def call(name: str, arguments: Any) -> str:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        if name == CLARIFY_TOOL["name"]:
            asked = _clarification(args)
            if asked is None:
                return json.dumps({"problems": ["clarify needs a question and two to four choices, as short text"]})
            raise chat._NeedsClarification(asked)
        out, problems = _check(kind, args, state)
        if problems:
            return json.dumps({"problems": problems})
        drafted.clear()
        drafted.update(out)
        return json.dumps({"ok": True})

    messages = [{"role": t["role"], "content": t["content"]} for t in list(history)[-chat.MAX_TURNS:]
                if t.get("role") in ("user", "assistant") and t.get("content")]
    messages.append({"role": "user", "content": text})
    model = chat.model_name()
    try:
        if chat.provider() == "anthropic":
            note = chat._ask_anthropic(client, model, system, messages, [spec, CLARIFY_TOOL], call)
        elif chat.provider() == "gemini":
            note = chat._ask_openai_compatible(client or chat._gemini_request, model, system, messages, [spec, CLARIFY_TOOL], call)
        else:
            note = chat._ask_ollama(client or chat._ollama_request, model, system, messages, [spec, CLARIFY_TOOL], call)
    except chat._NeedsClarification as exc:
        return {"draft": None, "clarification": exc.value, "note": exc.value["question"]}
    if not drafted:
        return {"draft": None, "clarification": None, "note": note or "No draft was produced; try saying it another way."}
    return {"draft": drafted, "clarification": None, "note": note}
