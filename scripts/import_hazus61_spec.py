"""Reproduce the Hazus 6.1 asset files from the user-supplied specification.

Run from the repository root; no YAML dependency or embedded prompt execution.

One file is written per Hazus class -- ``assets/fragility/hazus-bridge/HWB17.asset.json``
and so on -- each a complete asset on its own: the contract in the ``cap:``
vocabulary, the provenance, and that class's rows of every source table. The
contract fields are fixed here rather than read from the specification, which
supplies numbers and not capabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/sources/hazus61_roadway_bridge_fragility_prompt.md"
ASSETS = ROOT / "assets/fragility"
SCHEMA_REF = "../../../ontology/asset.schema.json"

CONTRACT = {
    "version": "0.1.0",
    "targetVariable": "AssetDamageState",
    "givenVariable": ["LocalIntensity"],
    "deterministic": False,
    "support": "Discrete",
    "drawKind": "Marginal",
    "batchable": True,
    "reproducible": True,
    "hasCapability": ["SampleForward", "EvaluateDensity", "EvaluateLikelihood"],
}
NOT_MODELLED = ("No deterioration/AssetHealth dependence, multi-asset joint draws, "
                "restoration, or damage-to-capacity mapping.")
# Classes whose modified shaking curve the source cannot supply. Their assets say so
# and offer the tabulated curves, instead of declaring a capability that would raise.
BASE_ONLY = {
    "HWB3": "the Table 7-2 K3D equation divides by N - 1, which is zero for a single-span bridge",
    "HWB4": "the Table 7-2 K3D equation divides by N - 1, which is zero for a single-span bridge",
    "HWB28": "Table 7-1 defines no K3D equation or Ishape for it",
}


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


HOISTED = ("model_family", "asset_family", "asset_class")


def _class_parameters(tables, code, **stated):
    """One class's share of the tables. What every curve repeats is stated once.

    ``src/fragility/hazus61/data.py`` puts it back, so the engine still sees the
    full record per curve that ``extract`` returns.
    """
    curves = [c for c in tables["curves"] if c["asset_class"] == code]
    common = {key: curves[0][key] for key in HOISTED}
    slim = [{**{k: v for k, v in c.items() if k not in HOISTED}, "source": {"table": c["source"]["table"]}}
            for c in curves]
    return {**common, **stated, "curves": slim}


def _record(folder, code, label, contract, scope, source, parameters):
    return {
        "$schema": SCHEMA_REF,
        "assetId": f"fragility/{folder}/{code}",
        "label": label,
        **{k: contract[k] for k in ("version", "implementedBy", "targetVariable", "givenVariable",
                                    "stateValues", "inputEncoding", "deterministic", "support", "units",
                                    "drawKind", "batchable", "reproducible", "hasCapability")},
        "validityScope": scope,
        "source": source,
        "parameters": parameters,
    }


def records(path=DEFAULT):
    """Every Hazus asset file, as ``{path: content}``."""
    road, bridge = extract(path)
    out = {}

    contract = {**CONTRACT, "implementedBy": "model.py:HazusRoadway",
                "stateValues": ["none", "slight", "moderate", "extensive_complete"],
                "inputEncoding": "hazus61-roadway-pgd-v1",
                "units": "categorical damage; parent PGD normalised to inches"}
    names = {cls.split("(")[1].split(")")[0]: cls.split("(")[0].strip()
             for cls, *_ in table(Path(path).read_bytes().decode("utf-8-sig"), "## 6.")}
    for code, name in names.items():
        scope = (f"Hazus 6.1 class {code} ({name}) only. Earthquake PGD fragility, reproduced from the "
                 "user-supplied Hazus 6.1 specification (Table 7-5). No ground-shaking curve, PGD "
                 f"generation, or separate Extensive/Complete states. {NOT_MODELLED} Explicit units required.")
        out[ASSETS / "hazus-roadway" / f"{code}.asset.json"] = _record(
            "hazus-roadway", code, f"Hazus 6.1 roadway ground-failure fragility, {code}: {name}",
            contract, scope, road["source"], _class_parameters(road, code))

    contract = {**CONTRACT, "implementedBy": "model.py:HazusBridge",
                "stateValues": ["none", "slight", "moderate", "extensive", "complete"],
                "inputEncoding": "hazus61-bridge-v1",
                "units": "categorical damage; parent Sa(1.0 s) in g or PGD in inches"}
    for code in bridge["pgd_modifiers"]:
        rules = [r for r in bridge["classification_rules"] if r["asset_class"] == code]
        equations = {r["k3d"] for r in rules}
        assert len(equations) == 1, f"{code} rows disagree on the K3D equation"
        equation = equations.pop()
        description, design = rules[0]["description"], rules[0]["design"]
        label = f"Hazus 6.1 highway-bridge fragility, {code}: {description}"
        if design != "-":
            label += f" ({design.lower()} design)"
        scope = (f"Hazus 6.1 class {code} only; the Table 7-1 rows that select it are under "
                 "parameters.classification_rules. Earthquake fragility for Sa(1.0 s) or PGD, reproduced "
                 "from the user-supplied Hazus 6.1 specification (Tables 7-1, 7-2, 7-6, 7-7). ")
        if code in BASE_ONLY:
            scope += (f"Supplies BASE shaking curves only: {BASE_ONLY[code]}, so the modified curve "
                      "is refused rather than repaired. ")
            parameters = _class_parameters(bridge, code, application="base", application_note=BASE_ONLY[code])
        else:
            shapes = {r["ishape"] for r in rules}
            assert len(shapes) == 1, f"{code} rows disagree on Ishape"
            scope += ("Shaking curves are modified for skew and span count"
                      + (", and the slight state for spectral shape" if shapes == {"1"} else "")
                      + "; a missing modifier input is an error, not a fallback to the base curve. ")
            parameters = _class_parameters(bridge, code, application="modified")
        scope += ("Crossed thresholds and source formula singularities are errors. PGD requires an "
                  "explicit interpretation policy. US/NBI classification or a named project mapping. "
                  + NOT_MODELLED)
        parameters.update(
            classification_rules=rules,
            k3d={"equation": equation, **bridge["k3d"][equation]} if equation in bridge["k3d"] else None,
            pgd_modifiers=bridge["pgd_modifiers"][code],
        )
        out[ASSETS / "hazus-bridge" / f"{code}.asset.json"] = _record(
            "hazus-bridge", code, label, contract, scope, bridge["source"], parameters)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT)
    args = parser.parse_args()
    for target, data in records(args.source).items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
