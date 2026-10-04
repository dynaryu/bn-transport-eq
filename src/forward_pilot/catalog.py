"""Dataset catalogue and adapters for the stakeholder pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# Only needed to BUILD packages from the source datasets. A built package is
# self-contained, so anyone working from the committed packages never needs
# this checkout. Override with NETWORK_DATASETS_ROOT.
DATASETS_ROOT = Path(
    os.environ.get("NETWORK_DATASETS_ROOT", r"C:\Users\bjieu\git\network-datasets\datasets")
)

CATALOG: dict[str, dict[str, Any]] = {
    "ema-highway": {
        "title": "Eastern Massachusetts highway network",
        "description": "Earthquake accessibility to either airport node n22 or n66.",
        "source_dir": DATASETS_ROOT / "ema_highway" / "v1" / "data",
        "directed": False,
        "destinations": ["n22", "n66"],
        "default_origins": "all_except_destinations",
        "hazard_models": ["ema_reference_scenario", "uniform_sa1"],
        "default_hazard": {
            "model": "ema_reference_scenario",
            "magnitude": 8.0,
            "epicentre_x": 1.0,
            "epicentre_y": 10.0,
        },
        "bridge_class_policy": "Use each edge's fragility_type; use HWB11 only if absent.",
        "coordinate_note": "Schematic planar coordinates interpreted as kilometres by the source scenario script.",
    },
    "hebei-transport": {
        "title": "Hebei directed transport network",
        "description": "Earthquake connectivity through directed arcs to n22 (Handan).",
        "source_dir": DATASETS_ROOT / "hebei_transport_liu2021" / "v1" / "data",
        "directed": True,
        "destinations": ["n22"],
        "default_origins": ["n1"],
        "hazard_models": ["uniform_sa1"],
        "default_hazard": {"model": "uniform_sa1", "sa1_g": 0.70},
        "bridge_class_policy": "Pilot assumption: each arc is one ordinary HWB11 bridge.",
        "coordinate_note": "Digitised figure coordinates are unitless; no spatial GMPE is applied.",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_case(case_id: str) -> dict[str, Any]:
    """The case definition, from the source datasets or from a built package.

    A built package is self-contained by design, so where the source datasets are
    not checked out the package stands in for them: everything downstream sees the
    same dictionary apart from `source_files`, which records where it came from.
    That is what lets this repository be cloned and run without the datasets.
    """
    if case_id not in CATALOG:
        raise ValueError(f"unknown case {case_id!r}; choose from {sorted(CATALOG)}")
    cfg = CATALOG[case_id]
    source_dir = Path(cfg["source_dir"])

    if not source_dir.is_dir():
        from .package import DEFAULT_OUTPUT, load_package  # local: package imports this module

        built = DEFAULT_OUTPUT / case_id
        if not built.is_dir():
            raise FileNotFoundError(
                f"neither the source dataset ({source_dir}) nor a built package "
                f"({built}) is available for {case_id!r}. Set NETWORK_DATASETS_ROOT "
                "to a network-datasets checkout, or run `python -m src.forward_pilot build`."
            )
        network = load_package(built)
        network["source_files"] = [f"built package: {built}"]
        return network

    nodes = _read_json(source_dir / "nodes.json")
    edges = _read_json(source_dir / "edges.json")

    clean_edges: dict[str, dict[str, Any]] = {}
    for edge_id, edge in edges.items():
        item = dict(edge)
        item["bridge_class"] = str(item.get("fragility_type", "HWB11"))
        clean_edges[edge_id] = item

    destinations = list(cfg["destinations"])
    origins_cfg = cfg["default_origins"]
    origins = (
        [node_id for node_id in nodes if node_id not in destinations]
        if origins_cfg == "all_except_destinations"
        else list(origins_cfg)
    )
    return {
        "case_id": case_id,
        "title": cfg["title"],
        "description": cfg["description"],
        "directed": bool(cfg["directed"]),
        "nodes": nodes,
        "edges": clean_edges,
        "destinations": destinations,
        "default_origins": origins,
        "hazard_models": list(cfg["hazard_models"]),
        "default_hazard": dict(cfg["default_hazard"]),
        "assumptions": {
            "bridge_representation": "Each network edge or arc is represented by one highway bridge.",
            "bridge_class": cfg["bridge_class_policy"],
            "coordinates": cfg["coordinate_note"],
            "functionality": "A bridge is unusable when damage reaches Extensive or Complete.",
            "dependence": "Bridge states are sampled independently conditional on the hazard scenario.",
        },
        "source_files": [source_dir / "nodes.json", source_dir / "edges.json"],
    }
