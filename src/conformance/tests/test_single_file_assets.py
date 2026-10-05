"""The asset schema is derived from the ontology, and every single-file asset observes it.

Two things are held here that no asset's own tests can hold: that the committed
schema is the one the ontology currently derives, and that the harness treats a
contributed file it has never seen the way it treats the ones in the repository.
"""

import json
from pathlib import Path

import pytest

from src.conformance.check import check_asset, find_declarations
from src.conformance.schema import SCHEMA, derive_schema, schema_text, validation_errors
from src.wrappers.base import CAPABILITY_METHOD
from src.wrappers.record import expected_asset_id, is_asset_file, read_asset_file

ROOT = Path(__file__).resolve().parents[3]
ASSET_FILES = [p for p in find_declarations(ROOT / "assets") if is_asset_file(p)]


# ---- the schema -----------------------------------------------------------


def test_committed_schema_is_the_one_the_ontology_derives():
    """Fails when capabilities.ttl changes and `python -m src.conformance.schema` was not run."""
    assert SCHEMA.read_text(encoding="utf-8") == schema_text()


def test_schema_takes_its_keys_types_and_cardinalities_from_the_ontology():
    schema = derive_schema()
    properties = schema["properties"]
    assert schema["additionalProperties"] is False
    # Enumerations are the ontology's named individuals, not a list kept here.
    assert properties["drawKind"]["enum"] == ["Joint", "Marginal"]
    assert properties["support"]["enum"] == ["Continuous", "Discrete", "Mixed"]
    assert set(properties["hasCapability"]["items"]["enum"]) == set(CAPABILITY_METHOD)
    # owl:cardinality 1 -> required scalar; owl:maxCardinality 1 -> optional scalar;
    # unrestricted -> optional array.
    assert properties["deterministic"]["type"] == "boolean"
    assert {"assetId", "implementedBy", "validityScope", "source"} <= set(schema["required"])
    assert "parameters" not in schema["required"] and properties["parameters"]["type"] == "object"
    assert "hasCapability" not in schema["required"] and properties["hasCapability"]["type"] == "array"
    assert properties["givenVariable"]["type"] == "array"


def test_the_template_a_contributor_copies_observes_the_schema():
    """Not discovered as an asset, but it must not teach a shape the harness refuses."""
    template = ROOT / "assets" / "_template" / "template.asset.json"
    assert template not in ASSET_FILES
    assert validation_errors(read_asset_file(template)) == []


# ---- the files in the repository ------------------------------------------


def test_the_registry_holds_one_file_per_asset():
    by_folder = {}
    for path in ASSET_FILES:
        by_folder.setdefault(path.parent.relative_to(ROOT / "assets").as_posix(), []).append(path)
    assert {k: len(v) for k, v in by_folder.items()} == {
        "fragility/hazus-bridge": 28, "fragility/hazus-roadway": 2, "hazard/gmpe": 30,
    }
    ids = [read_asset_file(p)["assetId"] for p in ASSET_FILES]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("path", ASSET_FILES, ids=lambda p: p.name)
def test_every_single_file_asset_observes_the_schema_and_conforms(path):
    record = read_asset_file(path)
    assert validation_errors(record) == []
    assert record["assetId"] == expected_asset_id(path)
    assert check_asset(path).ok


# ---- a contributed file the harness has never seen ------------------------

STUB = '''
import numpy as np

from src.wrappers.base import Asset, DataUnavailable


class Working(Asset):
    @classmethod
    def from_asset_file(cls, path, **options):
        return cls()

    def sample_forward(self, given, n=1, rng=None):
        return {"Y": np.zeros(n)}

    def smoke_inputs(self):
        return {"given": {}, "n": 2}


class Empty(Asset):
    @classmethod
    def from_asset_file(cls, path, **options):
        raise DataUnavailable("no numbers for this one.\\nand a second line of detail")


class Unbound(Asset):
    pass
'''


def contribute(tmp_path, folder, wrapper="Working", **changes):
    """Write a wrapper and one asset file, as a contributor would."""
    home = tmp_path / "demo" / folder
    home.mkdir(parents=True)
    (home / "model.py").write_text(STUB, encoding="utf-8")
    record = {
        "assetId": f"demo/{folder}/thing", "label": "A contributed asset", "version": "0.1.0",
        "implementedBy": f"model.py:{wrapper}", "targetVariable": "Y", "givenVariable": [],
        "deterministic": False, "support": "Continuous", "units": "none", "drawKind": "Marginal",
        "batchable": True, "reproducible": True, "hasCapability": ["SampleForward"],
        "validityScope": "A test fixture.", "source": {"citation": "none"},
    }
    record.update(changes)
    record = {k: v for k, v in record.items() if v is not None}
    path = home / "thing.asset.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_a_contributed_file_with_a_working_wrapper_conforms(tmp_path):
    result = check_asset(contribute(tmp_path, "works"))
    assert result.ok and not result.underclaimed and not result.unavailable
    assert result.asset_id == "demo/works/thing"


def test_a_capability_the_wrapper_lacks_is_an_overclaim(tmp_path):
    result = check_asset(contribute(tmp_path, "overclaims", hasCapability=["SampleForward", "Gradient"]))
    assert not result.ok
    assert result.overclaimed == {"Gradient"}


@pytest.mark.parametrize("changes,complaint", [
    ({"capabilities": ["SampleForward"]}, "Additional properties are not allowed"),
    ({"validityScope": None}, "'validityScope' is a required property"),
    ({"drawKind": "Independent"}, "drawKind"),
    ({"hasCapability": ["Sample"]}, "hasCapability"),
    ({"deterministic": "false"}, "deterministic"),
    ({"parameters": [1, 2]}, "parameters"),
], ids=["unknown-key", "missing-required", "bad-enum", "bad-capability", "wrong-type", "parameters-not-object"])
def test_a_file_that_departs_from_the_schema_is_refused_before_anything_runs(tmp_path, changes, complaint):
    result = check_asset(contribute(tmp_path, "malformed", **changes))
    assert not result.ok
    assert "does not observe ontology/asset.schema.json" in result.errors[0]
    assert complaint in result.errors[0]


def test_an_id_that_disagrees_with_the_files_location_is_refused(tmp_path):
    result = check_asset(contribute(tmp_path, "misfiled", assetId="demo/elsewhere/thing"))
    assert not result.ok
    assert "does not match its location" in result.errors[0]


def test_an_asset_without_its_numbers_conforms_only_if_it_claims_nothing(tmp_path):
    honest = check_asset(contribute(tmp_path, "honest", wrapper="Empty", hasCapability=[]))
    assert honest.ok
    assert honest.unavailable == "no numbers for this one."

    hopeful = check_asset(contribute(tmp_path, "hopeful", wrapper="Empty"))
    assert not hopeful.ok
    assert hopeful.overclaimed == {"SampleForward"}


def test_a_wrapper_that_cannot_bind_to_one_file_is_reported(tmp_path):
    result = check_asset(contribute(tmp_path, "unbound", wrapper="Unbound"))
    assert not result.ok
    assert "does not bind to a single-file asset" in result.errors[0]
