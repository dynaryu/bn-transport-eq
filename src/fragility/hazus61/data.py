"""Fresh copies of source tables so a caller cannot mutate another evaluation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def database(family):
    if family not in {"roadway", "highway_bridge"}:
        raise ValueError("family must be roadway or highway_bridge")
    slug = "hazus-roadway" if family == "roadway" else "hazus-bridge"
    return json.loads((ROOT / "assets/fragility" / slug / "params/database.json").read_text(encoding="utf-8"))


def curves(family, asset_class, hazard_mode):
    selected = [r for r in database(family)["curves"] if r["asset_class"] == asset_class and r["hazard_mode"] == hazard_mode]
    if not selected:
        raise ValueError(f"no curves for {family}/{asset_class}/{hazard_mode}")
    return selected
