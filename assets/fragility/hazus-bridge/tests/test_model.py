from pathlib import Path

import numpy as np
import pytest

from src.wrappers.loader import load_asset_class
from src.fragility.hazus61.classify import classify_bridge
from src.fragility.hazus61.engine import evaluate_bridge_modes
from src.fragility.hazus61.modifiers import k_3d, k_shape, k_skew, pgd_factors

HazusBridge = load_asset_class("model.py:HazusBridge", Path(__file__).resolve().parents[1])


def inventory(**overrides):
    return dict(state="TN", year_built=1968, nbi_class="501", num_spans=3,
                max_span_length_m=30, bridge_length_m=90, **overrides)


def given(code="HWB17", skew=30):
    return {"asset": {"hazus_bridge_class": code, "num_spans": 3, "skew_angle_deg": skew,
                      "bridge_length_m": 120, "span_width_m": 10},
            "LocalIntensity": {"sa_1p0_g": 0.43, "sa_0p3_g": 0.8, "pgd_in": 4}}


def test_example_class_and_modified_shaking():
    assert classify_bridge(inventory())["asset_class"] == "HWB17"
    result = HazusBridge().evaluate(given())
    factor = np.sqrt(np.cos(np.deg2rad(30))) * 1.125
    np.testing.assert_allclose(result["final_medians"][0], np.array([0.25, 0.35 * factor, 0.45 * factor, 0.7 * factor]))
    assert sum(result["probabilities"][0]) == pytest.approx(1)
    assert result["provenance"]["vintage_policy"] == "table_7_1"


@pytest.mark.parametrize("ca,year,expected", [(True, 1974, "HWB18"), (True, 1975, "HWB19"),
                                            (False, 1989, "HWB17"), (False, 1990, "HWB19")])
def test_vintage_boundaries(ca, year, expected):
    asset = inventory()
    asset.update(state="CA" if ca else "TN", year_built=year)
    result = classify_bridge(asset)
    assert result["asset_class"] == expected
    assert any("vintage_source_conflict" in w for w in result["warnings"])


def test_named_vintage_override():
    asset = inventory()
    asset.update(state="CA", year_built=1980)
    assert classify_bridge(asset)["asset_class"] == "HWB19"
    assert classify_bridge(asset, "section_7_1_3_prose")["asset_class"] == "HWB18"


def test_classification_precedence_and_short_bridge_boundary():
    asset = inventory()
    asset.update(max_span_length_m=151, num_spans=1)
    assert classify_bridge(asset)["asset_class"] == "HWB1"
    asset["max_span_length_m"] = 150
    assert classify_bridge(asset)["asset_class"] == "HWB3"
    asset.update(nbi_class="301", num_spans=2, bridge_length_m=19)
    assert classify_bridge(asset)["asset_class"] == "HWB24"
    asset["bridge_length_m"] = 20
    assert classify_bridge(asset)["asset_class"] == "HWB12"


def test_missing_data_does_not_become_hwb28():
    result = classify_bridge({"nbi_class": "501", "num_spans": 3, "max_span_length_m": 30})
    assert result["match_status"] == "ambiguous"
    assert set(result["candidate_classes"]) == {"HWB17", "HWB18", "HWB19"}
    assert classify_bridge({})["asset_class"] is None
    asset = inventory()
    asset["nbi_class"] = "999"
    assert classify_bridge(asset)["asset_class"] == "HWB28"


def test_foreign_taxonomy_requires_named_mapping():
    asset = {"country": "UK", "hazus_bridge_class": "HWB17"}
    assert classify_bridge(asset)["match_status"] == "out_of_native_taxonomy"
    asset["project_mapping"] = "test-project structural analogue, version 1"
    assert classify_bridge(asset)["asset_class"] == "HWB17"


def test_length_normalisation():
    asset = inventory()
    asset.pop("bridge_length_m")
    asset.update(bridge_length=9000, span_width=1000, length_unit="cm")
    result = classify_bridge(asset)
    assert result["asset_class"] == "HWB17"
    assert result["asset"]["bridge_length_m"] == 90
    assert result["asset"]["span_width_m"] == 10
    del asset["length_unit"]
    with pytest.raises(ValueError, match="length_unit"):
        classify_bridge(asset)


def test_base_curve_opt_in_and_missing_hwb28_rule():
    request = {"asset": {"hazus_bridge_class": "HWB17"}, "LocalIntensity": {"sa_1p0_g": 0.25}}
    model = HazusBridge()
    assert model.match(request["asset"])["match_status"] == "base_curve_only"
    with pytest.raises(ValueError, match="missing modifier"):
        model.evaluate(request)
    result = HazusBridge(application="base").evaluate(request)
    assert result["match_status"] == "base_curve_only"
    assert result["exceedance"][0][0] == 0.5
    with pytest.raises(ValueError, match="HWB28"):
        model.evaluate(given("HWB28"))


def test_modifier_known_values_and_singularities():
    assert k_skew(60) == pytest.approx(np.sqrt(0.5))
    assert k_shape(1.0, 0.2) == pytest.approx(0.5)
    assert k_3d("EQ1", 3) == 1.125
    assert pgd_factors("HWB17", given()["asset"]) == pytest.approx((4, 4))
    request = given("HWB3")
    request["asset"]["num_spans"] = 1
    with pytest.raises(ValueError, match="source_formula_singularity"):
        HazusBridge().evaluate(request)
    with pytest.raises(ValueError, match="source_formula_singularity"):
        pgd_factors("HWB17", given(skew=0)["asset"])
    with pytest.raises(ValueError, match="source_formula_singularity"):
        pgd_factors("HWB10", given(skew=0)["asset"])


def test_shape_zero_intensity_and_acceleration_units():
    request = given("HWB10")
    request["LocalIntensity"] = {"type": "SA", "period_s": 1.0, "unit": "m/s2", "value": [0, 1.96133],
                                 "sa_0p3": {"value": 1, "unit": "g"}}
    result = HazusBridge().evaluate(request)
    np.testing.assert_array_equal(result["probabilities"][0], [1, 0, 0, 0, 0])
    assert result["final_medians"][1][0] == pytest.approx(0.3)
    assert result["intensity"][1] == pytest.approx(0.2)


def test_pgd_interpretation_is_explicit_and_zero_lower_damage_mass():
    with pytest.raises(ValueError, match="bridge_pgd_policy"):
        HazusBridge(hazard_mode="ground_failure").evaluate(given())
    literal = HazusBridge(hazard_mode="ground_failure", bridge_pgd_policy="literal_table_and_steps").evaluate(given())
    np.testing.assert_allclose(literal["final_medians"][0], [3.9, 15.6, 15.6, 55.2])
    severe = HazusBridge(hazard_mode="ground_failure", bridge_pgd_policy="extensive_complete_only").evaluate(given())
    assert severe["probabilities"][0][1:3] == [0, 0]
    assert severe["exceedance"][0][0] == severe["exceedance"][0][2]


def test_crossed_thresholds_are_not_silently_repaired():
    request = given("HWB17", skew=85)
    with pytest.raises(ValueError, match="non_monotone"):
        HazusBridge().evaluate(request)
    request = given("HWB17")
    request["asset"]["bridge_length_m"] = 3
    with pytest.raises(ValueError, match="non_monotone"):
        HazusBridge(hazard_mode="ground_failure", bridge_pgd_policy="literal_table_and_steps").evaluate(request)


def test_modes_separate_and_explicit_union():
    options = {"bridge_pgd_policy": "extensive_complete_only"}
    result = evaluate_bridge_modes(given(), **options)
    assert set(result) == {"ground_shaking", "ground_failure"}
    result = evaluate_bridge_modes(given(), combination_policy="general_hazus_combination_rule", **options)
    gs, gf = (np.asarray(result[m]["exceedance"]) for m in ["ground_shaking", "ground_failure"])
    np.testing.assert_allclose(result["combined"]["exceedance"], 1 - (1 - gs) * (1 - gf))


def test_class_mixture_preserves_shared_class_when_combining_modes():
    request = given()
    del request["asset"]["hazus_bridge_class"]
    request["class_probabilities"] = {"HWB1": 0.4, "HWB17": 0.6}
    result = evaluate_bridge_modes(request, application="base", bridge_pgd_policy="extensive_complete_only", combination_policy="general_hazus_combination_rule")
    expected = 0
    for cls, weight in request["class_probabilities"].items():
        individual = given(cls)
        one = evaluate_bridge_modes(individual, application="base", bridge_pgd_policy="extensive_complete_only", combination_policy="general_hazus_combination_rule")
        expected += weight * np.asarray(one["combined"]["probabilities"])
    np.testing.assert_allclose(result["combined"]["probabilities"], expected)
