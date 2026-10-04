"""Conservative matching of explicit metadata; unknown fields remain unknown."""
from __future__ import annotations

import itertools
import re

from .data import database

US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR".split())
WARNINGS = [
    "vintage_source_conflict: Table 7-1 uses CA>=1975 / non-CA>=1990; section prose reverses these",
    "major_bridge_length_ambiguity: matching uses maximum span >150 m, not total bridge length",
    "classification_precedence_inferred: major, single-span, then table-order material rules; narrower overlapping ranges come first",
]


def normalize_asset(asset, family):
    if isinstance(asset, dict):
        result = dict(asset)
        for key in ("bridge_length", "span_width"):
            if result.get(key) is not None:
                factors = {"m": 1, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "in": 0.0254}
                unit = result.get("length_unit")
                if unit not in factors:
                    raise ValueError("bridge_length/span_width require length_unit (m, cm, mm, ft, in), or use explicit _m fields")
                metres = positive_number(result[key], key) * factors[unit]
                if result.get(key + "_m") is not None and abs(float(result[key + "_m"]) - metres) > 1e-10:
                    raise ValueError(f"conflicting {key} unit representations")
                result[key + "_m"] = metres
        return result
    if not isinstance(asset, str):
        raise TypeError("asset must be structured metadata or a short description")
    # Deliberately bounded vocabulary, not an unrestricted language interpreter.
    text = asset.lower()
    result = {"description": asset, "normalization": "bounded-explicit-aliases"}
    codes = set(re.findall(r"\b(?:hwb\d+|hrd\d+)\b", text))
    if len(codes) > 1:
        raise ValueError("description contains conflicting class codes; use structured metadata")
    if codes:
        result["hazus_bridge_class" if family == "highway_bridge" else "hazus_roadway_class"] = codes.pop().upper()
    if family == "roadway":
        major = bool(re.search(r"\b(major road|interstate(?: highway)?|state highway|parkway)\b", text))
        urban = bool(re.search(r"\b(urban road|city road|local road|two-lane road)\b", text))
        if major and urban:
            raise ValueError("conflicting roadway aliases; use structured metadata")
        if major or urban:
            result["class_hint"] = "major_road" if major else "urban_road"
        lanes = re.findall(r"\b(\d+)[ -]lanes?\b", text)
        if lanes:
            result["lanes"] = int(lanes[0])
    else:
        spans = re.findall(r"\b(\d+)[ -]spans?\b", text)
        if spans:
            result["num_spans"] = int(spans[0])
        if "single span" in text or "single-span" in text:
            result["num_spans"] = 1
        nbi = re.search(r"\bnbi(?: class)?[ :]*(\d{3})\b", text)
        if nbi:
            result["nbi_class"] = nbi.group(1)
        year = re.search(r"\bbuilt(?: in)? (\d{4})\b", text)
        if year:
            result["year_built"] = int(year.group(1))
        if re.search(r"\bnon-california\b", text):
            result["is_california"] = False
        elif re.search(r"\bcalifornia\b", text):
            result["is_california"] = True
    return result


def positive_number(value, field, integer=False):
    import math
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean")
    number = float(value)
    if not math.isfinite(number) or number <= 0 or (integer and int(number) != number):
        raise ValueError(f"{field} must be a positive {'integer' if integer else 'number'}")
    return int(number) if integer else number


def classify_roadway(asset):
    asset = normalize_asset(asset, "roadway")
    code = asset.get("hazus_roadway_class")
    if code is not None and code not in {"HRD1", "HRD2"}:
        raise ValueError("invalid roadway class")
    hint = asset.get("class_hint")
    if hint is not None and hint not in {"major_road", "urban_road"}:
        raise ValueError("invalid roadway class_hint")
    lanes = asset.get("lanes")
    if lanes is not None:
        lanes = positive_number(lanes, "lanes", integer=True)
    basis = "explicit_class"
    if code is None and hint:
        code, basis = ("HRD1" if hint == "major_road" else "HRD2"), "class_hint"
    if code is None and asset.get("route_type") in {"interstate", "state_highway", "parkway"}:
        code, basis = "HRD1", "route_type"
    if code is None and lanes is not None:
        code = "HRD1" if lanes >= 4 else "HRD2" if lanes == 2 else None
        basis = "lanes"
    return {"match_status": "exact" if code else "insufficient_information", "asset_class": code,
            "candidate_classes": [code] if code else ["HRD1", "HRD2"],
            "missing_fields": [] if code else ["class_hint or route_type or lanes"],
            "classification_basis": basis if code else None, "asset": asset, "warnings": []}


def _class_for(values, rules, policy):
    ca, year, spans, max_span, length, nbi = values
    threshold = (1975 if ca else 1990) if policy == "table_7_1" else (1990 if ca else 1975)
    modern = year >= threshold
    if max_span > 150:
        return "HWB2" if modern else "HWB1"
    if spans == 1:
        return "HWB4" if modern else "HWB3"
    for rule in rules:
        cls = rule["asset_class"]
        if cls in {"HWB1", "HWB2", "HWB3", "HWB4", "HWB28"}:
            continue
        low, high = map(int, rule["nbi_class"].split("-"))
        if not low <= nbi <= high or (rule["region"] == "CA") != ca:
            continue
        if (rule["design"] == "Seismic") != modern:
            continue
        if rule["length_lt20"] != "N/A" and (rule["length_lt20"] == "Yes") != (length < 20):
            continue
        return cls
    return "HWB28"


def classify_bridge(asset, vintage_policy="table_7_1"):
    if vintage_policy not in {"table_7_1", "section_7_1_3_prose"}:
        raise ValueError("unknown vintage_policy")
    asset = normalize_asset(asset, "highway_bridge")
    warnings = list(WARNINGS)
    code = asset.get("hazus_bridge_class")
    if code is not None and code not in {f"HWB{i}" for i in range(1, 29)}:
        raise ValueError("invalid hazus_bridge_class")
    for field in ("num_spans", "year_built", "max_span_length_m", "bridge_length_m"):
        if asset.get(field) is not None:
            asset[field] = positive_number(asset[field], field, integer=field in {"num_spans", "year_built"})
    if asset.get("nbi_class") is not None:
        raw = str(asset["nbi_class"])
        if not re.fullmatch(r"\d{3}", raw):
            raise ValueError("nbi_class must be an explicit three-digit code")
        asset["nbi_class"] = int(raw)
    ca = asset.get("is_california")
    if ca is not None and type(ca) is not bool:
        raise ValueError("is_california must be boolean")
    state = asset.get("state")
    if state is not None:
        state = state.upper()
        if state == "CALIFORNIA":
            state = "CA"
        if state not in US_STATES:
            raise ValueError("state must be a US postal code; foreign assets need country and project_mapping")
        if ca is not None and ca != (state == "CA"):
            raise ValueError("state and is_california conflict")
        ca = state == "CA"
    country = str(asset.get("country", "US")).upper()
    if country not in {"US", "USA", "UNITED STATES"}:
        if not isinstance(asset.get("project_mapping"), str) or not asset["project_mapping"].strip():
            return {"match_status": "out_of_native_taxonomy", "asset_class": None,
                    "candidate_classes": [code] if code else [], "missing_fields": ["project_mapping"],
                    "asset": asset, "warnings": warnings, "vintage_policy": vintage_policy}
        warnings.append("project_mapping_applied: " + asset["project_mapping"])
    if code:
        return {"match_status": "exact", "asset_class": code, "candidate_classes": [code],
                "missing_fields": [], "asset": asset, "classification_basis": "explicit_class",
                "warnings": warnings, "vintage_policy": vintage_policy}
    names = ["is_california", "year_built", "num_spans", "max_span_length_m", "bridge_length_m", "nbi_class"]
    known = [ca] + [asset.get(k) for k in names[1:]]
    # Representatives of every rule partition, not guessed metadata values.
    representatives = [[False, True], [1974, 1975, 1989, 1990], [1, 2], [150, 151], [19, 20],
                       [101, 201, 205, 301, 402, 501, 601, 605, 999]]
    options = [[v] if v is not None else alternatives for v, alternatives in zip(known, representatives)]
    rules = database("highway_bridge")["classification_rules"]
    candidates = sorted({_class_for(values, rules, vintage_policy) for values in itertools.product(*options)}, key=lambda x: int(x[3:]))
    exact = len(candidates) == 1
    return {"match_status": "exact" if exact else "ambiguous" if any(v is not None for v in known) else "insufficient_information",
            "asset_class": candidates[0] if exact else None, "candidate_classes": candidates,
            "missing_fields": [] if exact else [k for k, v in zip(names, known) if v is None],
            "asset": asset, "classification_basis": "table_7_1_rule_partitions",
            "warnings": warnings, "vintage_policy": vintage_policy}
