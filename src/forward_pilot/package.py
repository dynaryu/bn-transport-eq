"""Build portable, reviewable BN packages from the local network datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import CATALOG, load_case
from .engine import run_forward_query


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "forward-bn-pilot" / "packages"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _bn_spec(network: dict[str, Any]) -> dict[str, Any]:
    edge_ids = list(network["edges"])
    origins = network["default_origins"]
    return {
        "schema_version": "0.1.0",
        "model_type": "parametric_bayesian_network",
        "inference_scope": "forward_only",
        "factor_templates": [
            {
                "node": "EarthquakeScenario",
                "type": "input",
                "states": "continuous parameters",
            },
            {
                "nodes": [f"SA1::{edge_id}" for edge_id in edge_ids],
                "type": "deterministic_or_marginal_hazard",
                "parents": ["EarthquakeScenario"],
                "unit": "g",
            },
            {
                "nodes": [f"Damage::{edge_id}" for edge_id in edge_ids],
                "type": "Hazus61_extensive_limit_state_fragility",
                "parents_template": ["SA1::{edge_id}"],
                "states": ["BelowExtensive", "ExtensiveOrComplete"],
            },
            {
                "nodes": [f"Usable::{edge_id}" for edge_id in edge_ids],
                "type": "deterministic_policy",
                "parents_template": ["Damage::{edge_id}"],
                "states": [0, 1],
                "rule": "1 below Extensive; 0 for Extensive or Complete",
            },
            {
                "nodes": [f"Service::{origin}" for origin in origins],
                "type": "deterministic_network_connectivity",
                "parents": [f"Usable::{edge_id}" for edge_id in edge_ids],
                "destinations": network["destinations"],
                "directed": network["directed"],
            },
        ],
        "conditional_independence": (
            "Component damage variables are conditionally independent given the supplied hazard "
            "marginals in this pilot."
        ),
        "supported_queries": [
            "P(origin reaches any selected destination | earthquake scenario)",
            "Expected population fraction reaching any selected destination | earthquake scenario",
            "P(each bridge unusable | earthquake scenario)",
        ],
        "rejected_queries": [
            "P(component state | observed system state)",
            "most critical components given service failure",
            "diagnostic or backward inference",
        ],
        "extension_points": [
            {
                "name": "Intervention",
                "future_arc": "Intervention -> FragilityParameters -> Damage",
                "examples": ["retrofit", "replacement", "maintenance"],
            },
            {
                "name": "Deterioration",
                "future_arc": "Age/Exposure/Condition -> FragilityParameters -> Damage",
            },
            {
                "name": "Spatial ground-motion dependence",
                "future_arc": "EventResidual/SpatialResidual -> SA1",
            },
        ],
    }


def assemble_package(case_id: str, output_root: Path | str = DEFAULT_OUTPUT) -> Path:
    network = load_case(case_id)
    output = Path(output_root) / case_id
    output.mkdir(parents=True, exist_ok=True)
    source_records = [
        {"path": str(path), "sha256": _sha256(path)} for path in network.pop("source_files")
    ]
    manifest = {
        "package_id": case_id,
        "title": network["title"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "0.1.0",
        "inference_scope": "forward_only",
        "network": {"nodes": len(network["nodes"]), "edges": len(network["edges"]), "directed": network["directed"]},
        "source_files": source_records,
        "assumptions": network["assumptions"],
        "files": {
            "network": "network.json",
            "bn_spec": "bn_spec.json",
            "query_scope": "query_scope.json",
            "example_query": "example_query.json",
            "example_result": "example_result.json",
        },
    }
    query_scope = {
        "mode": "forward_only",
        "allowed": ["connectivity", "population_accessibility"],
        "disallowed": ["critical_components", "diagnosis", "backward_inference", "queries_with_evidence"],
        "reason": "The first pilot validates automated assembly and scenario-to-consequence inference.",
    }
    example_query = {
        "query_type": "population_accessibility" if case_id == "ema-highway" else "connectivity",
        "origins": network["default_origins"],
        "destinations": network["destinations"],
        "hazard": network["default_hazard"],
        "samples": 5000,
        "seed": 20261004,
    }
    example_result = run_forward_query(network, example_query)
    _write(output / "manifest.json", manifest)
    _write(output / "network.json", network)
    _write(output / "bn_spec.json", _bn_spec(network))
    _write(output / "query_scope.json", query_scope)
    _write(output / "example_query.json", example_query)
    _write(output / "example_result.json", example_result)
    return output


def assemble_all(output_root: Path | str = DEFAULT_OUTPUT) -> list[Path]:
    return [assemble_package(case_id, output_root) for case_id in CATALOG]


def load_package(path: Path | str) -> dict[str, Any]:
    package = Path(path)
    return json.loads((package / "network.json").read_text(encoding="utf-8"))
