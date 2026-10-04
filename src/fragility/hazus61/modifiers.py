"""Literal source modifiers; singularities and crossed thresholds are not repaired."""
import math
import numpy as np

from .classify import positive_number
from .data import database
from .probability import nonnegative


def angle(alpha):
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha < 90:
        raise ValueError("skew_angle_deg must be finite and in [0, 90)")
    return math.radians(alpha)


def k_skew(alpha):
    return math.sqrt(math.cos(angle(alpha)))


def k_shape(sa03, sa10):
    sa03, sa10 = nonnegative(sa03, "Sa(0.3s)"), nonnegative(sa10, "Sa(1.0s)")
    if (sa03 <= 0).any():
        raise ValueError("Sa(0.3s) must be positive for Ishape=1")
    return 2.5 * sa10 / sa03


def k_3d(eq, spans):
    spans = positive_number(spans, "num_spans", integer=True)
    data = database("highway_bridge")["k3d"]
    if eq not in data:
        raise ValueError("source_modifier_undefined: no K3D equation for this class")
    a, b = data[eq]["A"], data[eq]["B"]
    if spans - b <= 0:
        raise ValueError("source_formula_singularity: K3D requires N-B>0")
    return 1 + a / (spans - b)


def class_modifiers(code):
    rows = [r for r in database("highway_bridge")["classification_rules"] if r["asset_class"] == code]
    if not rows:
        raise ValueError("unknown bridge class")
    return rows[0]["k3d"], rows[0]["ishape"]


def pgd_factors(code, asset):
    rules = database("highway_bridge")["pgd_modifiers"][code]
    if rules["f1"] == rules["f2"] == "1":
        return 1.0, 1.0
    sine = math.sin(angle(asset["skew_angle_deg"]))
    if abs(sine) < 1e-12:
        raise ValueError("source_formula_singularity: Table 7-7 has zero/singular sin(alpha)")
    if rules["f1"] == "1":
        return 1.0, sine
    n = positive_number(asset["num_spans"], "num_spans", integer=True)
    # Explicit metre fields eliminate silent mismatches between width and length units.
    length = positive_number(asset["bridge_length_m"], "bridge_length_m")
    width = positive_number(asset["span_width_m"], "span_width_m")
    value = 0.5 * length / (n * width * sine)
    return value, value


def required_fields(code, mode):
    if mode == "ground_shaking":
        _, shape = class_modifiers(code)
        return ["skew_angle_deg", "num_spans"] + (["sa_0p3_g"] if shape == "1" else [])
    rules = database("highway_bridge")["pgd_modifiers"][code]
    if rules["f1"] == rules["f2"] == "1":
        return []
    return ["skew_angle_deg"] + (["num_spans", "bridge_length_m", "span_width_m"] if rules["f1"] != "1" else [])


def modified_medians(base, code, mode, asset, intensity):
    final = np.broadcast_to(np.asarray(base, dtype=float), (len(intensity), 4)).copy()
    if mode == "ground_shaking":
        eq, shape = class_modifiers(code)
        skew, three_d = k_skew(asset["skew_angle_deg"]), k_3d(eq, asset["num_spans"])
        shape_factor = np.ones(len(intensity))
        if shape == "1":
            shape_factor = np.minimum(1, k_shape(asset["sa_0p3_g"], intensity))
        final[:, 0] *= shape_factor
        # At zero Sa1 the shape-adjusted slight median is also zero. Preserve
        # that diagnostic; the engine handles IM=0 without evaluating log(0/0).
        final[:, 1:] *= skew * three_d
        info = {"k_skew": skew, "k_3d": three_d, "k3d_equation": eq, "slight_shape_factor": shape_factor.tolist()}
    else:
        f1, f2 = pgd_factors(code, asset)
        final[:, 1:3] *= f1
        final[:, 3] *= f2
        info = {"f1": f1, "f2": f2, "source_table": "7-7"}
    return final, info
