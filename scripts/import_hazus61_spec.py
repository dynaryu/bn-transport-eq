"""Reproduce JSON tables from the user-supplied Hazus 6.1 specification.

Run from the repository root; no YAML dependency or embedded prompt execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/sources/hazus61_roadway_bridge_fragility_prompt.md"


def table(text, heading):
    block = text.split(heading, 1)[1].split("\n## ", 1)[0]
    lines = [line for line in block.splitlines() if line.startswith("| ")]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def extract(path=DEFAULT):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig")
    source = {
        "document": "Hazus Earthquake Model Technical Manual", "version": "6.1",
        "date": "July 2024", "section": "7.1",
        "url": "https://www.fema.gov/sites/default/files/documents/fema_hazus-earthquake-model-technical-manual-6-1.pdf",
        "transcription": "docs/sources/hazus61_roadway_bridge_fragility_prompt.md",
        "transcription_sha256": hashlib.sha256(raw).hexdigest(),
        "verification": "Reproduced from user-supplied specification; full FEMA PDF retrieval returned HTTP 403",
        "manual_page": None, "pdf_page": None,
    }
    road = {"source": source, "curves": []}
    bridge = {"source": source, "curves": [], "classification_rules": [], "k3d": {}, "pgd_modifiers": {}}
    for cls, state, median, beta in table(text, "## 6."):
        code = cls.split("(")[1].split(")")[0]
        state = state.lower().replace("/", "_")
        road["curves"].append(curve(code, state, "ground_failure", float(median), float(beta), "7-5", source))
    for row in table(text, "## 11."):
        cls, *values = row
        for mode, numbers, beta in [("ground_shaking", values[:4], 0.6), ("ground_failure", values[4:], 0.2)]:
            for state, median in zip(["slight", "moderate", "extensive", "complete"], numbers):
                bridge["curves"].append(curve(cls, state, mode, float(median), beta, "7-6", source))
    keys = ["asset_class", "nbi_class", "region", "year", "spans", "max_span", "length_lt20", "k3d", "ishape", "design", "description"]
    for index, row in enumerate(table(text, "## 8.")):
        rule = dict(zip(keys, row))
        rule.update(priority=index, source_table="7-1")
        bridge["classification_rules"].append(rule)
    for eq, a, b, expression in table(text, "## 10."):
        bridge["k3d"][eq] = {"A": float(a), "B": float(b), "expression": expression, "source_table": "7-2"}
    for cls, f1, f2 in table(text, "## 13."):
        bridge["pgd_modifiers"][cls] = {"f1": f1, "f2": f2, "source_table": "7-7"}
    assert len(road["curves"]) == 6
    assert len(bridge["curves"]) == 224
    assert len(bridge["pgd_modifiers"]) == 28
    assert len(bridge["k3d"]) == 7
    return road, bridge


def curve(cls, state, mode, median, beta, table_id, source):
    return {
        "curve_id": f"HAZUS61_{cls}_{'SA1' if mode == 'ground_shaking' else 'PGD'}_{state.upper()}",
        "model_family": "Hazus 6.1", "asset_family": "roadway" if cls.startswith("HRD") else "highway_bridge",
        "asset_class": cls, "hazard_mode": mode,
        "im_type": "SA" if mode == "ground_shaking" else "PGD",
        "im_period_s": 1.0 if mode == "ground_shaking" else None,
        "im_unit": "g" if mode == "ground_shaking" else "in", "damage_state": state,
        "median": median, "beta_ln": beta,
        "modifier_rule_id": None if cls.startswith("HRD") else f"{cls}_{mode}",
        "source": {**source, "table": table_id},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT)
    args = parser.parse_args()
    for slug, data in zip(["hazus-roadway", "hazus-bridge"], extract(args.source)):
        target = ROOT / "assets/fragility" / slug / "params/database.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
