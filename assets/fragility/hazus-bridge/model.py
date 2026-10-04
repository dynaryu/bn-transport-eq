"""Hazus 6.1 highway-bridge shaking or PGD fragility."""
from src.fragility.hazus61.asset import HazusAsset
from src.fragility.hazus61.engine import BRIDGE_STATES


class HazusBridge(HazusAsset):
    asset_id = "fragility/hazus-bridge"
    family = "highway_bridge"
    states = BRIDGE_STATES
