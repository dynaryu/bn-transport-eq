"""Expandable translation contracts, few-shot selection and held-out evaluation."""
from __future__ import annotations
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from src.builder.load import ROOT
from src.inference import engine

CASE_SCHEMA = {
    "type": "object", "required": ["id", "split", "assembly", "context", "questions", "expected", "must_not"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "minLength": 1}, "split": {"enum": ["few_shot", "held_out"]},
        "assembly": {"type": "string"}, "context": {"type": "object"},
        "questions": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "expected": {"type": "object", "required": ["outcome", "tool_arguments"], "properties": {
            "outcome": {"enum": ["compute", "clarify", "unsupported"]},
            "tool_arguments": {"type": "array", "minItems": 1, "items": {"type": "object"}}}},
        "must_not": {"type": "array", "items": {"type": "string"}},
    },
}


def load_cases(folder=None):
    paths = sorted(Path(folder).glob("*.json")) if folder else sorted((ROOT / "examples").glob("*/answer-cases/*.json"))
    cases, ids = [], set()
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        errors = list(Draft202012Validator(CASE_SCHEMA).iter_errors(case))
        if errors:
            raise engine.QueryError(f"{path.name}: {errors[0].message}")
        if case["id"] in ids or path.stem != case["id"]:
            raise engine.QueryError(f"Duplicate id or filename mismatch: {case['id']}")
        ids.add(case["id"])
        cases.append(case)
    return cases


def few_shots(assembly):
    return ["\nReviewed example: " + json.dumps({"context": c["context"], "question": c["questions"][0],
            "tool_arguments": c["expected"]["tool_arguments"]})
            for c in load_cases() if c["assembly"] == assembly and c["split"] == "few_shot"][:4]


def canonical(arguments, assembly, defaults):
    """Ignore JSON order, conjunction order and sampling controls, retaining causal meaning."""
    from .chat import coerce_arguments
    from src.inference.comparisons import merge
    if "analysis" in arguments:
        value = arguments["analysis"]
        arguments = {"analysis": {**value, "query": merge(defaults, {"assembly": assembly, **value["query"]})}}
    elif "clarification" not in arguments:
        arguments = coerce_arguments(arguments, assembly, defaults)
    def tidy(value):
        if isinstance(value, dict):
            out = {k: tidy(v) for k, v in value.items() if k not in ("samples", "seed")}
            for key in ("all", "any", "events", "targets"):
                if key in out:
                    out[key] = sorted(out[key], key=lambda v: json.dumps(v, sort_keys=True))
            return out
        return [tidy(v) for v in value] if isinstance(value, list) else value
    return tidy(_labelled(arguments, assembly))


def _labelled(arguments, assembly):
    """Each discrete node's value as its state label, so "value": 0 and "value": "Fail" are one plan."""
    from src.inference import engine

    try:
        quantities = {q["id"]: q for q in engine.describe(assembly)["quantities"] if q["kind"] == "discrete"}
    except Exception:                       # an unknown assembly is reported elsewhere
        return arguments

    def label(node, value):
        q = quantities.get(str(node).split("::")[0])
        index = engine.state_position(q, value) if q else None
        return value if index is None else q["states"][index]

    def walk(value):
        if isinstance(value, list):
            return [walk(v) for v in value]
        if not isinstance(value, dict):
            return value
        out = {k: walk(v) for k, v in value.items()}
        if "node" in out and "value" in out:
            out["value"] = label(out["node"], out["value"])
        for key in ("do", "evidence"):
            if isinstance(out.get(key), dict):
                out[key] = {node: label(node, v) for node, v in out[key].items()}
        return out

    return walk(arguments)


def check_contracts():
    from src.inference.comparisons import execute
    cases = load_cases()
    def validate_query(query):
        engine.scope(query)
        events = query.get("joint_events", query.get("events", []))
        return {"events": [{"p": 1 / len(events), "ci95": [0, 1]} for _ in events],
                "inference": {"method": "contract validation only"}}
    for case in cases:
        for args in case["expected"]["tool_arguments"]:
            if "analysis" in args:
                # Expand generated queries and validate names/types without drawing samples.
                execute(canonical(args, case["assembly"], case["context"].get("defaults", {}))["analysis"], validate_query)
            elif "clarification" in args:
                from .chat import CLARIFICATION_SCHEMA
                errors = list(Draft202012Validator(CLARIFICATION_SCHEMA).iter_errors(args["clarification"]))
                if errors:
                    raise engine.QueryError("Invalid clarification: " + errors[0].message)
            else:
                validate_query(canonical(args, case["assembly"], case["context"].get("defaults", {})))
    return {"contracts": len(cases), "few_shot": sum(c["split"] == "few_shot" for c in cases),
            "held_out": sum(c["split"] == "held_out" for c in cases), "live_model_evaluated": False}


def evaluate(client=None, model=None):
    """Translation only: no numerical runs. Uses held-out cases only.

    Clarifications compare required missing-input labels, not exact prose. Computations
    compare canonical query plans; this deliberately does not claim general equivalence.
    """
    from . import chat
    rows = []
    for case in load_cases():
        if case["split"] != "held_out":
            continue
        for question in case["questions"]:
            plans = []
            chat.ask(case["assembly"], question, history=case["context"].get("history", []),
                     client=client, model=model, defaults=case["context"].get("defaults"),
                     translate_only=True, on_event=lambda e: plans.append(e["arguments"]) if e["type"] == "plan" else None)
            expected = case["expected"]["tool_arguments"]
            defaults = case["context"].get("defaults", {})
            if case["expected"]["outcome"] == "clarify":
                wanted = set(expected[0]["clarification"]["missing"])
                actual = plans[0].get("clarification") if len(plans) == 1 else None
                valid = isinstance(actual, dict) and not list(Draft202012Validator(chat.CLARIFICATION_SCHEMA).iter_errors(actual))
                passed = (valid and wanted <= set(actual["missing"]) and
                          (not expected[0]["clarification"].get("choices") or bool(actual.get("choices"))))
            else:
                passed = ([canonical(a, case["assembly"], defaults) for a in plans] ==
                          [canonical(a, case["assembly"], defaults) for a in expected])
            rows.append({"case": case["id"], "question": question, "passed": passed, "actual": plans})
    from src.inference import records
    report = {"kind": "evaluation", "id": records.new_id(), "model": model or chat.model_name(),
              "provider": chat.provider(), "passed": sum(r["passed"] for r in rows), "total": len(rows),
              "rows": rows, "note": "Canonical plan agreement; must_not entries also need human review."}
    records._append(report)
    return report
