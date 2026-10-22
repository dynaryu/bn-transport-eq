"""Reviewable data proposals. Drafts never enter the active instance loader."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from datetime import datetime, timezone

from jsonschema import Draft202012Validator

from src.builder.load import ROOT, INSTANCES, SCHEMA, RegistryError, load_library
from src.inference import engine

_LOCK = threading.Lock()
SPEC = {
    "type": "object", "required": ["kind", "title", "reason", "source", "payload"], "additionalProperties": False,
    "properties": {
        "kind": {"enum": ["parameter_set", "scenario"]}, "title": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1}, "source": {"type": "object", "minProperties": 1},
        "payload": {"type": "object"},
    },
}


def directory():
    return Path(os.environ.get("QUAKENET_PROPOSALS") or ROOT / "proposals")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _path(identifier):
    if not re.fullmatch(r"[A-Za-z0-9-]+", identifier):
        raise engine.QueryError("Invalid proposal id")
    return directory() / f"{identifier}.json"


def read(identifier):
    path = _path(identifier)
    if not path.is_file():
        raise engine.QueryError("No such proposal")
    return json.loads(path.read_text(encoding="utf-8"))


def list_proposals():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory().glob("*.json"))]


def validate(spec):
    errors = [e.message for e in Draft202012Validator(SPEC).iter_errors(spec)]
    if errors:
        return errors
    try:
        payload = spec["payload"]
        if spec["kind"] == "parameter_set":
            lib = engine.library()
            fn = lib.functions.get(payload.get("function"))
            if fn is None:
                raise engine.QueryError("A new mathematical function needs code and tests, not only a data proposal")
            # Run the real library checks in isolation, including duplicates and typed parameters.
            with tempfile.TemporaryDirectory() as temp:
                staged = Path(temp) / "instances"
                shutil.copytree(INSTANCES, staged)
                target = staged / fn["role"] / "proposal.json"
                if target.exists():
                    raise engine.QueryError("Reserved staging filename already exists")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(payload), encoding="utf-8")
                load_library(SCHEMA, staged)
        else:
            if set(payload) != {"id", "query"} or not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", payload["id"]):
                raise engine.QueryError("Scenario payload needs id '<network>/<name>' and a complete query")
            engine.scope(payload["query"])
            compiled = engine.compile_query(payload["query"])
            if payload["id"].split("/")[0] != compiled.network.id:
                raise engine.QueryError("Scenario id must belong to the query's network")
    except (engine.QueryError, RegistryError, ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    return errors


def draft(spec):
    from src.inference.records import new_id
    if not isinstance(spec, dict):
        raise engine.QueryError("Proposal must be a JSON object")
    identifier = new_id()
    errors = validate(spec)
    proposal = {"id": identifier, "status": "draft", "spec": spec, "sha256": fingerprint(spec),
                "created_at": datetime.now(timezone.utc).isoformat(), "validation": errors}
    directory().mkdir(parents=True, exist_ok=True)
    _path(identifier).write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    return proposal


def activate(identifier, reviewer, expected_sha256):
    """Explicit user activation of the exact reviewed draft, revalidated against current data."""
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise engine.QueryError("A reviewer name is required")
    with _LOCK:
        proposal = read(identifier)
        spec = proposal["spec"]
        if proposal["status"] != "draft" or fingerprint(spec) != expected_sha256:
            raise engine.QueryError("Proposal changed or is already active; review its current version")
        errors = validate(spec)
        if errors:
            raise engine.QueryError("; ".join(errors))
        payload = spec["payload"]
        if spec["kind"] == "parameter_set":
            role = engine.library().functions[payload["function"]]["role"]
            target = INSTANCES / role / f"proposal-{identifier}.json"
            content = {**payload, "source": {**payload["source"], "proposal": identifier,
                       "reviewer": reviewer.strip(), "approved_at": datetime.now(timezone.utc).isoformat()}}
        else:
            network, name = payload["id"].split("/")
            target = ROOT / "examples" / network / f"{name}.scenario.json"
            content = {**payload, "title": spec["title"], "source": spec["source"], "proposal": identifier}
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents replacing any active file.
        with target.open("x", encoding="utf-8") as out:
            out.write(json.dumps(content, indent=2) + "\n")
        proposal.update(status="active", reviewer=reviewer.strip(), activated_at=datetime.now(timezone.utc).isoformat(),
                        file=target.relative_to(ROOT).as_posix())
        _path(identifier).write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
        engine.library.cache_clear()
        return proposal


def scenarios(assembly):
    return [data for p in sorted((ROOT / "examples").glob("*/*.scenario.json"))
            if (data := json.loads(p.read_text(encoding="utf-8")))["query"]["assembly"] == assembly]
