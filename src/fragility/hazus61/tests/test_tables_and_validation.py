import json
from pathlib import Path

import numpy as np
import pytest

from scripts.import_hazus61_spec import extract
from src.conformance.check import check_asset
from src.fragility.hazus61.classify import classify_bridge
from src.fragility.hazus61.data import database
from src.fragility.hazus61.engine import evaluate
from src.fragility.hazus61.modifiers import k_3d, pgd_factors
from src.fragility.hazus61.probability import cumulative_to_discrete, lognormal_exceedance

ROOT = Path(__file__).resolve().parents[4]


def test_source_transcription_reproduces_every_record():
    roadway, bridge = extract()
    assert roadway == database("roadway")
    assert bridge == database("highway_bridge")
    assert len(roadway["curves"]) == 6
    assert len(bridge["curves"]) == 224
    assert len({c["curve_id"] for c in roadway["curves"] + bridge["curves"]}) == 230


@pytest.mark.parametrize("family", ["roadway", "highway_bridge"])
def test_every_curve_has_median_property_and_source(family):
    for curve in database(family)["curves"]:
        assert lognormal_exceedance(curve["median"], curve["median"], curve["beta_ln"]) == 0.5
        assert lognormal_exceedance(0, curve["median"], curve["beta_ln"]) == 0
        assert curve["source"]["table"] in {"7-5", "7-6"}
        assert len(curve["source"]["transcription_sha256"]) == 64


@pytest.mark.parametrize("index,expected", list(enumerate([
    [0.4, 0.5, 0.7, 0.9], [0.6, 0.9, 1.1, 1.7], [0.8, 1, 1.2, 1.7], [0.8, 1, 1.2, 1.7],
    [0.25, 0.35, 0.45, 0.7], [0.3, 0.5, 0.6, 0.9], [0.5, 0.8, 1.1, 1.7],
    [0.35, 0.45, 0.55, 0.8], [0.6, 0.9, 1.3, 1.6], [0.6, 0.9, 1.1, 1.5], [0.9, 0.9, 1.1, 1.5],
    [0.25, 0.35, 0.45, 0.7], [0.3, 0.5, 0.6, 0.9], [0.5, 0.8, 1.1, 1.7],
    [0.75, 0.75, 0.75, 1.1], [0.9, 0.9, 1.1, 1.5], [0.25, 0.35, 0.45, 0.7], [0.3, 0.5, 0.6, 0.9],
    [0.5, 0.8, 1.1, 1.7], [0.35, 0.45, 0.55, 0.8], [0.6, 0.9, 1.3, 1.6], [0.6, 0.9, 1.1, 1.5],
    [0.9, 0.9, 1.1, 1.5], [0.25, 0.35, 0.45, 0.7], [0.3, 0.5, 0.6, 0.9],
    [0.75, 0.75, 0.75, 1.1], [0.75, 0.75, 0.75, 1.1], [0.8, 1, 1.2, 1.7],
], start=1)))
def test_all_bridge_base_rows(index, expected):
    selected = [c for c in database("highway_bridge")["curves"] if c["asset_class"] == f"HWB{index}"]
    assert [c["median"] for c in selected[:4]] == expected
    assert [c["median"] for c in selected[4:]] == [3.9, 3.9, 3.9, 13.8]
    assert [c["beta_ln"] for c in selected] == [0.6] * 4 + [0.2] * 4


def test_all_k3d_and_pgd_rows():
    expected = [(0.25, 1), (0.33, 0), (0.33, 1), (0.09, 1), (0.05, 0), (0.2, 1), (0.1, 0)]
    for i, (a, b) in enumerate(expected, 1):
        assert k_3d(f"EQ{i}", 3) == pytest.approx(1 + a / (3 - b))
    for i in range(1, 29):
        expected_factors = (1, 1) if i in {1, 2, 3, 4, 28} else (1, 0.5) if i in {8, 9, 10, 11, 15, 16, 20, 21, 26, 27} else (4, 4)
        assert pgd_factors(f"HWB{i}", {"skew_angle_deg": 30, "num_spans": 3, "bridge_length_m": 120, "span_width_m": 10}) == pytest.approx(expected_factors)


def test_all_roadway_rows():
    rows = database("roadway")["curves"]
    assert [c["median"] for c in rows] == [12, 24, 60, 6, 12, 24]
    assert [c["beta_ln"] for c in rows] == [0.7] * 6


@pytest.mark.parametrize("rule", database("highway_bridge")["classification_rules"])
def test_every_classification_source_row(rule):
    major = rule["asset_class"] in {"HWB1", "HWB2"}
    single = rule["asset_class"] in {"HWB3", "HWB4"}
    ca = rule["region"] == "CA"
    threshold = 1975 if ca else 1990
    asset = {"is_california": ca, "year_built": threshold if rule["design"] == "Seismic" else threshold - 1,
             "num_spans": 1 if single else 3, "max_span_length_m": 151 if major else 30,
             "bridge_length_m": 19 if rule["length_lt20"] == "Yes" else 90,
             "nbi_class": rule["nbi_class"].split("-")[0] if rule["nbi_class"] not in {"All", "-"} else "999"}
    assert classify_bridge(asset)["asset_class"] == rule["asset_class"]


@pytest.mark.parametrize("im,median,beta", [(-1, 1, 1), (np.nan, 1, 1), (np.inf, 1, 1),
                                           (0, 0, 1), (0, 1, 0), (1, np.inf, 1), (1, 1, np.nan)])
def test_invalid_distribution_parameters(im, median, beta):
    with pytest.raises(ValueError):
        lognormal_exceedance(im, median, beta)


def test_ordering_and_pga_rejection():
    with pytest.raises(ValueError, match="non_monotone"):
        cumulative_to_discrete([0.2, 0.8, 0.1, 0])
    with pytest.raises(ValueError, match="PGA"):
        evaluate("highway_bridge", {"asset": {"hazus_bridge_class": "HWB17"},
                                   "LocalIntensity": {"type": "PGA", "unit": "g", "value": 0.2}},
                 hazard_mode="ground_shaking", application="base")


@pytest.mark.parametrize("slug", ["hazus-roadway", "hazus-bridge"])
def test_registry_contract(slug):
    result = check_asset(ROOT / "assets/fragility" / slug / "asset.ttl")
    assert result.ok, result
    assert not result.underclaimed
