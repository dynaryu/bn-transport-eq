# Hazus 6.1 highway-bridge fragility

`HazusBridge` maps supplied Sa(1.0 s) or PGD to `AssetDamageState` for HWB1–HWB28.
The five states are `none`, `slight`, `moderate`, `extensive`, `complete`.

**Each class is its own asset file**, `HWB1.asset.json` … `HWB28.asset.json`: the
contract, the source record, and that class's share of every table — its eight
curves (224 in all), its Table 7-1 classification rows, its K3D equation and its PGD
modifier rule. Each observes the [schema derived from the ontology](../../README.md#single-file-assets).
`HazusBridge()` is the family: it reads every file here and classifies an inventory
record among them. `HazusBridge.from_asset_file("HWB17.asset.json")` answers for
that one class and refuses metadata that states another.

HWB3, HWB4 and HWB28 declare `"application": "base"` in their own files, because the
source cannot supply their modified shaking curve (single-span K3D singularity; no
modifier defined for HWB28). The family wrapper still refuses those cases unless
base curves are requested explicitly.

Modified shaking is the default. Missing metadata never silently selects a class
or bypasses modifiers. Base curves are available through an explicit application
policy. PGD requires a named interpretation policy because the supplied source
contains conflicting statements about its damage states. Literal source
singularities, missing HWB28 shaking modifiers and crossed thresholds are errors.

Capabilities are forward sampling, probability mass and likelihood. There is no
AssetHealth dependence, multi-asset joint model, restoration or capacity mapping.
See the [API and policies](../../../docs/hazus61-fragility.md),
[source record](SOURCE.md), and `tests/test_model.py`. Shared table and numerical
tests are in `src/fragility/hazus61/tests/`.
