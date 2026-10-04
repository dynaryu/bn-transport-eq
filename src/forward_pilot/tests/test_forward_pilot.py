import json
from pathlib import Path

import pytest

from src.forward_pilot.catalog import CATALOG, load_case
from src.forward_pilot.engine import ForwardQueryError, run_forward_query
from src.forward_pilot.hazard import component_probabilities
from src.forward_pilot.package import assemble_package, load_package


def test_ema_reference_scenario_reproduces_dataset_probabilities():
    network = load_case("ema-highway")
    calculated, _, warnings = component_probabilities(network, network["default_hazard"])
    source_path = Path(
        r"C:\Users\bjieu\git\network-datasets\datasets\ema_highway\v1\data\probs_eq.json"
    )
    # Compares our hazard code against the source dataset's own precomputed
    # file, so it is skipped where that checkout is absent.
    if not source_path.is_file():
        pytest.skip(f"source dataset not available at {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert calculated.keys() == source.keys()
    assert max(abs(calculated[e] - source[e]["0"]["p"]) for e in calculated) < 1e-12
    assert any("placeholder" in warning for warning in warnings)


def test_forward_query_rejects_backward_evidence():
    network = load_case("ema-highway")
    with pytest.raises(ForwardQueryError, match="forward inference only"):
        run_forward_query(
            network,
            {
                "query_type": "critical_components",
                "evidence": {"Service": 0},
                "samples": 100,
            },
        )


def test_higher_uniform_intensity_reduces_hebei_connectivity():
    network = load_case("hebei-transport")
    base = {"query_type": "connectivity", "samples": 2000, "seed": 17}
    low = run_forward_query(
        network, {**base, "hazard": {"model": "uniform_sa1", "sa1_g": 0.3}}
    )
    high = run_forward_query(
        network, {**base, "hazard": {"model": "uniform_sa1", "sa1_g": 1.1}}
    )
    assert low["origin_service"][0]["probability"] > high["origin_service"][0]["probability"]
    assert low["components"][0]["p_unusable"] < high["components"][0]["p_unusable"]


def test_package_is_self_contained_and_queryable(tmp_path):
    """Needs the source datasets: it tests BUILDING a package, not using one.

    Everything else in the pilot falls back to the committed packages, so this is
    the only test that a clone without the datasets cannot run.
    """
    if not Path(CATALOG["hebei-transport"]["source_dir"]).is_dir():
        pytest.skip("source datasets not available; package building cannot be tested")
    path = assemble_package("hebei-transport", tmp_path)
    expected = {
        "manifest.json",
        "network.json",
        "bn_spec.json",
        "query_scope.json",
        "example_query.json",
        "example_result.json",
    }
    assert {item.name for item in path.iterdir()} == expected
    network = load_package(path)
    result = run_forward_query(network, {"query_type": "connectivity", "samples": 100})
    assert result["case_id"] == "hebei-transport"
    # Which backend answers depends on whether tbnpy and torch are installed, so
    # the test pins the pair rather than one string. Backend equivalence is
    # covered in test_tbn.py.
    assert (result["backend"], result["inference"]) in {
        ("tbn", "forward_tbnpy_sampling"),
        ("analytic", "forward_monte_carlo"),
    }
