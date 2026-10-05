from pathlib import Path

import numpy as np
import pytest

from src.wrappers.loader import load_asset_class
from src.fragility.hazus61.classify import classify_roadway

HazusRoadway = load_asset_class("model.py:HazusRoadway", Path(__file__).resolve().parents[1])


def given(code="HRD1", value=12):
    return {"asset": {"hazus_roadway_class": code}, "LocalIntensity": {"pgd_in": value}}


@pytest.mark.parametrize("asset,code", [
    ({"class_hint": "urban_road", "lanes": 4}, "HRD2"),
    ({"route_type": "state_highway", "lanes": 2}, "HRD1"),
    ({"lanes": 4}, "HRD1"), ({"lanes": 2}, "HRD2"),
    ("two-lane road", "HRD2"), ("interstate highway", "HRD1"),
])
def test_classification(asset, code):
    assert classify_roadway(asset)["asset_class"] == code


def test_unknown_road_returns_candidates():
    result = classify_roadway({"lanes": 3})
    assert result["asset_class"] is None
    assert result["candidate_classes"] == ["HRD1", "HRD2"]


def test_combined_state_and_known_median():
    result = HazusRoadway().evaluate(given())
    assert result["states"] == ["none", "slight", "moderate", "extensive_complete"]
    assert result["exceedance"][0][0] == pytest.approx(0.5)
    assert sum(result["probabilities"][0]) == pytest.approx(1)
    assert "complete" not in result["states"]


def test_units_and_batch():
    model = HazusRoadway()
    request = given(value=[0, 12, 24])
    expected = model.distribution(request)
    request["LocalIntensity"] = {"type": "PGD", "unit": "mm", "value": [0, 304.8, 609.6]}
    np.testing.assert_allclose(model.distribution(request), expected)
    np.testing.assert_array_equal(expected[0], [1, 0, 0, 0])


def test_class_mixture():
    model = HazusRoadway()
    request = {"asset": {}, "LocalIntensity": {"pgd_in": 12}, "class_probabilities": {"HRD1": 0.25, "HRD2": 0.75}}
    expected = 0.25 * model.distribution(given("HRD1")) + 0.75 * model.distribution(given("HRD2"))
    np.testing.assert_allclose(model.distribution(request), expected)
    request["class_probabilities"]["HRD1"] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        model.distribution(request)


def test_sampling_and_likelihood():
    model = HazusRoadway()
    a = model.sample_forward(given(), 20000, np.random.default_rng(91))["AssetDamageState"]
    b = model.sample_forward(given(), 20000, np.random.default_rng(91))["AssetDamageState"]
    np.testing.assert_array_equal(a, b)
    expected = model.distribution(given())[0]
    np.testing.assert_allclose([(a == s).mean() for s in model.states], expected, atol=0.012)
    for index, state in enumerate(model.states):
        assert model.evaluate_likelihood({"AssetDamageState": state}, given())[0] == pytest.approx(expected[index])


def test_no_health_adjustment_or_road_shaking():
    request = given()
    request["AssetHealth"] = "poor"
    with pytest.raises(ValueError, match="AssetHealth"):
        HazusRoadway().evaluate(request)
    with pytest.raises(ValueError, match="hazard_mode"):
        HazusRoadway(hazard_mode="ground_shaking")


def test_one_file_answers_for_one_class():
    bound = HazusRoadway.from_asset_file(Path(__file__).resolve().parents[1] / "HRD2.asset.json")
    assert bound.asset_id == "fragility/hazus-roadway/HRD2"
    np.testing.assert_allclose(bound.distribution({"LocalIntensity": {"pgd_in": 12}}),
                               HazusRoadway().distribution(given("HRD2")))
    with pytest.raises(ValueError, match="HRD2 only"):
        bound.evaluate(given("HRD1"))
