"""Auditable fragility evaluation and conservative class lookup."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .classify import classify_bridge, classify_roadway, normalize_asset
from .data import curves, database
from .modifiers import class_modifiers, modified_medians, required_fields
from .probability import combine_independent_exceedance, cumulative_to_discrete, lognormal_exceedance, nonnegative

BRIDGE_STATES = ("none", "slight", "moderate", "extensive", "complete")
ROAD_STATES = ("none", "slight", "moderate", "extensive_complete")


def convert(value, unit, kind):
    unit = str(unit).lower()
    factors = ({"in": 1, "inch": 1, "inches": 1, "mm": 1 / 25.4, "cm": 1 / 2.54,
                "m": 1 / 0.0254, "ft": 12} if kind == "PGD" else
               {"g": 1, "m/s2": 1 / 9.80665, "m/s^2": 1 / 9.80665,
                "cm/s2": 1 / 980.665, "cm/s^2": 1 / 980.665, "gal": 1 / 980.665})
    if unit not in factors:
        raise ValueError(f"unsupported {kind} unit: {unit}")
    return nonnegative(value, kind) * factors[unit]


def _vector(value):
    value = np.atleast_1d(value)
    if value.ndim != 1 or len(value) == 0:
        raise ValueError("intensity must be a scalar or nonempty one-dimensional batch")
    return value


def normalize_intensity(hazard, mode):
    kind = "SA" if mode == "ground_shaking" else "PGD"
    key = "sa_1p0_g" if kind == "SA" else "pgd_in"
    canonical_unit = "g" if kind == "SA" else "in"
    if hazard.get("type") is not None and hazard["type"] != kind:
        raise ValueError("intensity type conflicts with requested mode; PGA is not SA or PGD")
    if kind == "SA" and hazard.get("period_s") is not None and hazard["period_s"] != 1.0:
        raise ValueError("bridge shaking requires SA at exactly 1.0 s")
    if "value" in hazard:
        if hazard.get("type") != kind or (kind == "SA" and hazard.get("period_s") != 1.0):
            raise ValueError("ground shaking requires SA at exactly 1.0 s; ground failure requires PGD (PGA is not SA)")
        value = convert(hazard["value"], hazard["unit"], kind)
        if key in hazard and not np.array_equal(value, np.asarray(hazard[key])):
            raise ValueError("conflicting generic and explicit-unit intensity fields")
    elif key in hazard:
        value = convert(hazard[key], canonical_unit, kind)
    else:
        raise ValueError(f"missing {key}; no conversion from PGA or another hazard is inferred")
    return _vector(value)


def _context(asset, hazard, size):
    asset = dict(asset)
    if "sa_0p3" in hazard:
        short = hazard["sa_0p3"]
        sa03 = convert(short["value"], short["unit"], "SA")
        if "sa_0p3_g" in hazard and not np.array_equal(sa03, np.asarray(hazard["sa_0p3_g"])):
            raise ValueError("conflicting Sa(0.3s) fields")
    elif "sa_0p3_g" in hazard:
        sa03 = convert(hazard["sa_0p3_g"], "g", "SA")
    else:
        sa03 = None
    if sa03 is not None:
        sa03 = _vector(sa03)
        if len(sa03) not in (1, size):
            raise ValueError("Sa(0.3s) must be scalar or match the Sa(1.0s) batch")
        asset["sa_0p3_g"] = sa03
    return asset


def match(family, asset, hazard_mode, hazard=None, vintage_policy="table_7_1"):
    if family == "roadway":
        if hazard_mode != "ground_failure":
            raise ValueError("roadways have PGD curves only")
        result = classify_roadway(asset)
    elif family == "highway_bridge":
        if hazard_mode not in {"ground_shaking", "ground_failure"}:
            raise ValueError("invalid hazard_mode")
        result = classify_bridge(asset, vintage_policy)
    else:
        raise ValueError("unsupported family")
    code = result["asset_class"]
    result["hazard_mode"] = hazard_mode
    if code is None:
        return result
    selected = curves(family, code, hazard_mode)
    result.update(curves=selected, curve_ids=[c["curve_id"] for c in selected], can_compute_base_fragility=True)
    missing = []
    if family == "highway_bridge":
        values = {**result["asset"], **(hazard or {})}
        if "sa_0p3" in values:
            values["sa_0p3_g"] = values["sa_0p3"]
        missing = [key for key in required_fields(code, hazard_mode) if values.get(key) is None]
        if hazard_mode == "ground_shaking" and class_modifiers(code)[0] == "-":
            missing.append("sourced_HWB28_shaking_modifier_rule")
    result.update(missing_fields=missing, can_compute_modified_fragility=not missing)
    if missing:
        result["match_status"] = "base_curve_only"
    return result


def evaluate(family, given, *, hazard_mode, application="modified", bridge_pgd_policy=None, vintage_policy="table_7_1"):
    """Evaluate one asset across intensity scenarios; returns only JSON-native values."""
    if "AssetHealth" in given:
        raise ValueError("Hazus tables do not define dependence on AssetHealth; no deterioration adjustment is available")
    if application not in {"base", "modified"}:
        raise ValueError("application must be base or modified")
    if family == "highway_bridge" and hazard_mode == "ground_failure" and bridge_pgd_policy not in {"literal_table_and_steps", "extensive_complete_only"}:
        raise ValueError("bridge ground failure requires an explicit bridge_pgd_policy")
    if given.get("class_probabilities") is not None:
        return _mixture(family, given, hazard_mode=hazard_mode, application=application,
                        bridge_pgd_policy=bridge_pgd_policy, vintage_policy=vintage_policy)
    hazard = given["LocalIntensity"]
    matched = match(family, given["asset"], hazard_mode, hazard, vintage_policy)
    if not matched["asset_class"]:
        return matched
    if application == "modified" and matched["missing_fields"]:
        raise ValueError("missing modifier inputs: " + ", ".join(matched["missing_fields"]) + "; inspect match() or explicitly select application='base'")
    intensity = normalize_intensity(hazard, hazard_mode)
    context = _context(matched["asset"], hazard, len(intensity))
    selected = matched["curves"]
    base = np.asarray([c["median"] for c in selected])
    beta = np.asarray([c["beta_ln"] for c in selected])
    final = np.broadcast_to(base, (len(intensity), len(base))).copy()
    applied = {}
    if application == "modified" and family == "highway_bridge":
        final, applied = modified_medians(base, matched["asset_class"], hazard_mode, context, intensity)
    warnings = list(matched["warnings"])
    if family == "highway_bridge" and application == "base":
        warnings.append("base_curve_only: asset-specific modifiers were explicitly bypassed")
        matched["match_status"] = "base_curve_only"
    if family == "highway_bridge" and hazard_mode == "ground_failure":
        warnings.append("bridge_pgd_source_conflict: full Table 7-6 thresholds coexist with a narrative limiting ground-failure damage to extensive/complete")
        if bridge_pgd_policy == "extensive_complete_only":
            # Severe-only damage implies identical exceedance of all lower thresholds.
            final[:, :2] = final[:, 2:3]
            warnings.append("extensive_complete_only: lower exceedances equal extensive; slight/moderate point masses are zero")
    # Check threshold ordering even at IM=0, where all probabilities coincide.
    if (np.diff(final, axis=1) < 0).any():
        raise ValueError("non_monotone_exceedance: modified thresholds cross; source values have not been repaired")
    q = np.zeros_like(final)
    active = intensity > 0
    if active.any():
        q[active] = lognormal_exceedance(intensity[active, None], final[active], beta)
    p = cumulative_to_discrete(q)
    states = ROAD_STATES if family == "roadway" else BRIDGE_STATES
    data = database(family)
    return {**matched, "states": list(states), "intensity": intensity.tolist(),
            "im_type": selected[0]["im_type"], "im_unit": selected[0]["im_unit"], "im_period_s": selected[0]["im_period_s"],
            "base_medians": base.tolist(), "final_medians": final.tolist(), "beta_ln": beta.tolist(),
            "modifiers": applied, "exceedance": q.tolist(), "probabilities": p.tolist(),
            "warnings": warnings, "application": application, "bridge_pgd_policy": bridge_pgd_policy,
            "provenance": {"source": data["source"], "tables": sorted({c["source"]["table"] for c in selected} | ({"7-1", "7-2" if hazard_mode == "ground_shaking" else "7-7"} if family == "highway_bridge" else set())),
                           "database_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                           "classification_basis": matched["classification_basis"], "vintage_policy": vintage_policy,
                           "assumptions": ["single-asset marginal fragility", "no deterioration/inspection-health adjustment"]}}


def _mixture(family, given, **options):
    weights = given["class_probabilities"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError("class_probabilities must be a nonempty class-to-weight mapping")
    raw = nonnegative(list(weights.values()), "class weights")
    if not np.isclose(raw.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("class weights must sum to one")
    context = normalize_asset(given["asset"], family)
    key = "hazus_roadway_class" if family == "roadway" else "hazus_bridge_class"
    if context.get(key) is not None:
        raise ValueError("supply either an explicit class or class_probabilities")
    components = []
    for cls, weight in zip(weights, raw / raw.sum()):
        sub = {**given, "asset": {**context, key: cls}}
        sub.pop("class_probabilities")
        result = evaluate(family, sub, **options)
        if "probabilities" not in result:
            raise ValueError(f"class {cls} cannot be evaluated: {result['match_status']}")
        components.append({"weight": float(weight), "result": result})
    q = sum(c["weight"] * np.asarray(c["result"]["exceedance"]) for c in components)
    return {"match_status": "class_mixture", "states": components[0]["result"]["states"],
            "exceedance": q.tolist(), "probabilities": cumulative_to_discrete(q).tolist(),
            "components": components, "provenance": {"mixture_policy": "caller_supplied_class_probabilities",
                                                        "supplied_class_probabilities": weights,
                                                        "weight_sum_roundoff_normalized": float(raw.sum()) != 1.0}}


def evaluate_bridge_modes(given, *, bridge_pgd_policy, combination_policy=None, application="modified", vintage_policy="table_7_1"):
    """Always retain separate modes; an explicit policy enables independent union."""
    if combination_policy not in {None, "general_hazus_combination_rule"}:
        raise ValueError("unsupported combination_policy")
    result = {mode: evaluate("highway_bridge", given, hazard_mode=mode, application=application,
                             bridge_pgd_policy=bridge_pgd_policy, vintage_policy=vintage_policy)
              for mode in ("ground_shaking", "ground_failure")}
    if combination_policy:
        gs, gf = result["ground_shaking"], result["ground_failure"]
        if "probabilities" not in gs or "probabilities" not in gf:
            raise ValueError("both hazard-mode classifications must resolve before combination")
        if "components" in gs:
            # Class is a shared latent variable: combine within class, then marginalise.
            q = sum(a["weight"] * combine_independent_exceedance(a["result"]["exceedance"], b["result"]["exceedance"])
                    for a, b in zip(gs["components"], gf["components"]))
        else:
            q = combine_independent_exceedance(gs["exceedance"], gf["exceedance"])
        result["combined"] = {"states": list(BRIDGE_STATES), "exceedance": q.tolist(),
                              "probabilities": cumulative_to_discrete(q).tolist(),
                              "provenance": {"combination_policy": combination_policy,
                                             "assumption": "GS and GF damage independent conditional on class and supplied intensities; not an equation printed in the bridge subsection"}}
    return result
