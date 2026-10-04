"""Hazus 6.1 roadway PGD fragility with a combined extensive/complete state."""
from src.fragility.hazus61.asset import HazusAsset
from src.fragility.hazus61.engine import ROAD_STATES


class HazusRoadway(HazusAsset):
    asset_id = "fragility/hazus-roadway"
    family = "roadway"
    states = ROAD_STATES
