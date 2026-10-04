"""Known-value tests. These are the regression floor for the asset."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.wrappers.loader import load_asset_module  # noqa: E402

# Loaded by path, not as `model`: every asset's wrapper is called model.py, so a bare
# `from model import ...` picks up whichever asset pytest collected first.
StConnectivity = load_asset_module(
    Path(__file__).resolve().parents[1] / "model.py"
).StConnectivity

# s --0-- a --1-- t
#  \--2-- b --3--/
EDGES = [("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")]
UP = 1.0
DOWN = 0.0


def given(functionality):
    return {
        "edges": EDGES,
        "source": "s",
        "target": "t",
        "AssetFunctionality": np.atleast_2d(functionality),
    }


@pytest.fixture
def asset():
    return StConnectivity()


@pytest.mark.parametrize(
    "states, expected, why",
    [
        ([UP, UP, UP, UP], 1, "all links up"),
        ([UP, UP, DOWN, DOWN], 1, "upper path alone suffices"),
        ([DOWN, DOWN, UP, UP], 1, "lower path alone suffices"),
        ([UP, DOWN, DOWN, UP], 0, "one link of each path down, no complete path"),
        ([DOWN, DOWN, DOWN, DOWN], 0, "all links down"),
    ],
)
def test_connectivity(asset, states, expected, why):
    assert asset.sample_forward(given(states))["Service"][0] == expected, why


def test_threshold_is_a_parameter(asset):
    """Partial functionality below threshold severs the link."""
    partial = StConnectivity(threshold=0.8)
    states = [0.6, 0.6, DOWN, DOWN]
    assert asset.sample_forward(given(states))["Service"][0] == 1
    assert partial.sample_forward(given(states))["Service"][0] == 0


def test_density_is_degenerate(asset):
    g = given([UP, UP, DOWN, DOWN])
    assert asset.evaluate_density({"Service": np.array([1])}, g)[0] == 1.0
    assert asset.evaluate_density({"Service": np.array([0])}, g)[0] == 0.0


def test_batch_matches_single(asset):
    batch = np.array([[UP, UP, UP, UP], [UP, DOWN, DOWN, UP], [DOWN, DOWN, DOWN, DOWN]])
    out = asset.batch_classify(given(batch))
    assert list(out) == [1, 0, 0]


def test_likelihood_is_unnormalised_over_parents(asset):
    """Both all-up and upper-path-only give S=1, so L does not sum to one over parents."""
    total = sum(
        asset.evaluate_likelihood({"Service": np.array([1])}, given(s))[0]
        for s in ([UP, UP, UP, UP], [UP, UP, DOWN, DOWN], [DOWN, DOWN, UP, UP])
    )
    assert total == 3.0
