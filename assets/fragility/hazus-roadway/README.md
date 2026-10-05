# Hazus 6.1 roadway PGD fragility

`HazusRoadway` maps `LocalIntensity` (PGD) to a categorical `AssetDamageState` for
HRD1/HRD2. The four states are `none`, `slight`, `moderate`, `extensive_complete`.
The source provides no partition of Extensive from Complete for roads.

Each class is its own asset file, `HRD1.asset.json` and `HRD2.asset.json`, holding
the contract, the source record and that class's three Table 7-5 curves, and
observing the [schema derived from the ontology](../../README.md#single-file-assets).
`HazusRoadway()` reads both; `HazusRoadway.from_asset_file("HRD1.asset.json")`
answers for one. They implement
forward sampling, probability mass and likelihood. They do not implement
deterioration, restoration, ground shaking, PGD generation, or capacity mapping.

See the [API and examples](../../../docs/hazus61-fragility.md),
[source record](SOURCE.md), and `tests/test_model.py`. Shared numerical and table
tests are in `src/fragility/hazus61/tests/`.
