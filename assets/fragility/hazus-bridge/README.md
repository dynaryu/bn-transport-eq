# Hazus 6.1 highway-bridge fragility

`HazusBridge` maps supplied Sa(1.0 s) or PGD to `AssetDamageState` for HWB1–HWB28.
The five states are `none`, `slight`, `moderate`, `extensive`, `complete`.
All 224 base curve records, classification rules, K3D coefficients and PGD modifier
rules are in `params/database.json`.

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
