# Hazus 6.1 roadway PGD fragility

`HazusRoadway` maps `LocalIntensity` (PGD) to a categorical `AssetDamageState` for
HRD1/HRD2. The four states are `none`, `slight`, `moderate`, `extensive_complete`.
The source provides no partition of Extensive from Complete for roads.

All six Table 7-5 curves are in `params/database.json`. This asset implements
forward sampling, probability mass and likelihood. It does not implement
deterioration, restoration, ground shaking, PGD generation, or capacity mapping.

See the [API and examples](../../../docs/hazus61-fragility.md),
[source record](SOURCE.md), and `tests/test_model.py`. Shared numerical and table
tests are in `src/fragility/hazus61/tests/`.
