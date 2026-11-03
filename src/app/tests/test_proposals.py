"""Draft isolation, content-bound approval, validation and live refresh."""
import copy
import json
import pytest

from src import proposals
from src.inference import engine


def spec():
    return {"kind": "parameter_set", "title": "Test closure policy", "reason": "Exercise reviewed addition",
            "source": {"basis": "user-defined test assumption"}, "payload": {
                "function": "edge.open", "source": {"basis": "test policy"},
                "sets": [{"id": "test.closed-at-slight", "label": "Test closure", "values": {
                    "damage_scale": "DamageState.HazusBridge", "usable_states": ["none"]}}]}}


def test_draft_is_isolated_and_activation_requires_exact_reviewed_content(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAKENET_PROPOSALS", str(tmp_path / "proposals"))
    # Activation goes only into this scratch copy, never the repository.
    import shutil
    scratch = tmp_path / "instances"
    shutil.copytree(proposals.INSTANCES, scratch)
    monkeypatch.setattr(proposals, "INSTANCES", scratch)
    monkeypatch.setattr(proposals, "ROOT", tmp_path)
    p = proposals.draft(spec())
    assert p["validation"] == []
    assert "test.closed-at-slight" not in engine.library().parameter_sets
    with pytest.raises(engine.QueryError, match="reviewer"):
        proposals.activate(p["id"], "", p["sha256"])
    with pytest.raises(engine.QueryError, match="changed"):
        proposals.activate(p["id"], "Colleague", "incorrect digest")
    active = proposals.activate(p["id"], "Colleague", p["sha256"])
    assert active["reviewer"] == "Colleague" and (tmp_path / active["file"]).is_file()
    with pytest.raises(engine.QueryError, match="already active"):
        proposals.activate(p["id"], "Colleague", p["sha256"])


def test_invalid_drafts_cannot_be_activated(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAKENET_PROPOSALS", str(tmp_path))
    data = spec()
    data["payload"]["sets"][0]["values"]["usable_states"] = ["invented-state"]
    p = proposals.draft(data)
    assert p["validation"]
    with pytest.raises(engine.QueryError):
        proposals.activate(p["id"], "Colleague", p["sha256"])


def test_scenario_validates_the_actual_network_and_system_event(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAKENET_PROPOSALS", str(tmp_path))
    p = proposals.draft({"kind": "scenario", "title": "N1 only", "reason": "Destination-specific access",
        "source": {"basis": "user-defined requirement"}, "payload": {"id": "toy-ca/n1-only", "query": {
            "assembly": "toy-ca/demo", "rows": {"n2": {"key_destination": False}},
            "events": [{"node": "SystemState::n4", "op": "==", "value": 0}]}}})
    assert p["validation"] == []
