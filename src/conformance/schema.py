"""Derive the single-file asset schema from the capability ontology.

Run:  python -m src.conformance.schema [--check]

A single-file asset (``<name>.asset.json``) is one JSON object whose keys are the
local names of the properties ``ontology/capabilities.ttl`` gives ``cap:Asset``.
This module turns the ontology into the JSON Schema those files are validated
against, so the vocabulary is written once:

  which keys exist      every property whose rdfs:domain is cap:Asset
  the type of a value   its rdfs:range -- a datatype, or a class whose named
                        individuals become the permitted values
  one value or many,    the owl:cardinality restrictions on cap:Asset
  required or optional

OWL alone rejects nothing: a restriction is an inference rule under an open world.
The derived schema is the closed-world reading, and it refuses unknown keys, which
is what stops a contributed file from carrying a contract field no compiler reads.

``ontology/asset.schema.json`` is the committed copy, there for editors and for
contributors who do not run Python. The harness validates against a fresh
derivation, and a test holds the committed copy to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY = ROOT / "ontology" / "capabilities.ttl"
SCHEMA = ROOT / "ontology" / "asset.schema.json"

CAP = Namespace("https://example.org/dra/cap#")
RDF_JSON = URIRef(str(RDF) + "JSON")


def _local(iri: Any) -> str:
    return str(iri).replace("#", "/").rsplit("/", 1)[-1]


def _comment(g: Graph, node: Any) -> str:
    return " ".join(str(c) for c in g.objects(node, RDFS.comment))


def _cardinality(g: Graph) -> dict[Any, tuple[int, int | None]]:
    """(min, max) per property, from the restrictions cap:Asset is a subclass of."""
    out: dict[Any, tuple[int, int | None]] = {}
    for node in g.objects(CAP.Asset, RDFS.subClassOf):
        prop = g.value(node, OWL.onProperty)
        if prop is None:
            continue
        exact = g.value(node, OWL.cardinality)
        lo = exact if exact is not None else g.value(node, OWL.minCardinality)
        hi = exact if exact is not None else g.value(node, OWL.maxCardinality)
        out[prop] = (int(lo) if lo is not None else 0, int(hi) if hi is not None else None)
    return out


def _value_schema(g: Graph, prop: Any) -> dict[str, Any]:
    """JSON type of one value of ``prop``, read off its rdfs:range."""
    rng = g.value(prop, RDFS.range)
    if rng in (None, XSD.string):
        return {"type": "string", "minLength": 1}
    if rng == XSD.boolean:
        return {"type": "boolean"}
    if rng == RDF_JSON:
        return {"type": "object"}
    if rng == RDF.List:
        return {"type": "array", "items": {"type": "string", "minLength": 1},
                "minItems": 1, "uniqueItems": True}
    members = sorted(g.subjects(RDF.type, rng), key=_local)
    if not members:
        raise ValueError(
            f"cannot derive a JSON type for {prop}: its range {rng} is neither a known "
            "datatype nor a class with named individuals"
        )
    schema: dict[str, Any] = {"enum": [_local(m) for m in members]}
    notes = [f"{_local(m)}: {_comment(g, m)}" for m in members if _comment(g, m)]
    if notes:
        schema["description"] = " | ".join(notes)
    return schema


def derive_schema(ontology: Path | str = ONTOLOGY) -> dict[str, Any]:
    """The JSON Schema a single-file asset must observe."""
    g = Graph().parse(ontology, format="turtle")
    cardinality = _cardinality(g)
    version = g.value(URIRef(str(CAP).rstrip("#")), OWL.versionInfo)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for prop in sorted(set(g.subjects(RDFS.domain, CAP.Asset)) | set(cardinality), key=_local):
        name = _local(prop)
        if name in properties:
            raise ValueError(f"two properties of cap:Asset share the local name {name!r}")
        lo, hi = cardinality.get(prop, (0, None))
        value = _value_schema(g, prop)
        if hi == 1:
            entry = value
        else:
            entry = {"type": "array", "items": value, "uniqueItems": True}
            if lo:
                entry["minItems"] = lo
            if hi is not None:
                entry["maxItems"] = hi
        comment = _comment(g, prop)
        if comment:
            entry = {"description": comment, **entry}
        properties[name] = entry
        if lo >= 1:
            required.append(name)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://example.org/dra/cap/asset.schema.json",
        "title": str(g.value(CAP.Asset, RDFS.label) or "Knowledge asset"),
        "description": (
            f"One knowledge asset as a single file. GENERATED from ontology/capabilities.ttl "
            f"(version {version}) by `python -m src.conformance.schema`. Edit the ontology, "
            "not this file."
        ),
        "type": "object",
        "properties": {
            "$schema": {"description": "Path or URL of this schema, for editors.", "type": "string"},
            **properties,
        },
        "required": sorted(required),
        "additionalProperties": False,
    }


def schema_text(ontology: Path | str = ONTOLOGY) -> str:
    return json.dumps(derive_schema(ontology), indent=2, ensure_ascii=False) + "\n"


@lru_cache(maxsize=1)
def _validator():
    try:
        from jsonschema import Draft7Validator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "jsonschema is required to validate single-file assets.\n"
            "  conda env create -f environment.yml   (then: conda activate dra)\n"
            "or install it into the env you are already using."
        ) from exc
    return Draft7Validator(derive_schema())


def validation_errors(record: Any) -> list[str]:
    """Every way ``record`` departs from the derived schema; empty when it observes it."""
    errors = sorted(_validator().iter_errors(record), key=lambda e: [str(p) for p in e.absolute_path])
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<record>'}: {e.message}" for e in errors]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the committed schema is stale, without rewriting it")
    args = parser.parse_args(argv)
    text = schema_text()
    if args.check:
        current = SCHEMA.read_text(encoding="utf-8") if SCHEMA.exists() else ""
        if current != text:
            print(f"{SCHEMA.relative_to(ROOT)} is stale; run: python -m src.conformance.schema")
            return 1
        print(f"{SCHEMA.relative_to(ROOT)} matches the ontology")
        return 0
    SCHEMA.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {SCHEMA.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
