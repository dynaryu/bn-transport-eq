# Hazus 6.1 roadway and bridge assets

Thirty registry assets, one per Hazus class, reproduce the supplied specification's
earthquake fragility tables. The numeric data are complete for these tables; that
does not resolve the source ambiguities or supply a deterioration model.

| Collection | Assets | Hazard modes | Damage states |
|---|---|---|---|
| `fragility/hazus-roadway/` | `HRD1`, `HRD2` | PGD | none, slight, moderate, extensive_complete |
| `fragility/hazus-bridge/` | `HWB1` … `HWB28` | Sa(1.0 s); PGD under an explicit policy | none, slight, moderate, extensive, complete |

Each class is a single file, `<class>.asset.json`, with `cap:assetId`
`fragility/<collection>/<class>`. It carries the contract, the source record and
that class's share of every table, and observes the schema derived from the
ontology — see [single-file assets](../assets/README.md#single-file-assets).

There are 230 individual curve records: 6 roadway PGD records and 224 bridge
records (28 classes × 4 damage thresholds × 2 hazard modes). A bridge file also
holds the class's Table 7-1 classifier rows, its K3D equation and its PGD modifier
expressions. Shared code is in `src/fragility/hazus61/`, which reassembles the
whole tables from the files: classification has to walk every Table 7-1 row in
source order, so that order is carried in each row's `priority`.

## Load and evaluate

```python
from pathlib import Path
from src.wrappers.loader import load_asset_class

Bridge = load_asset_class("model.py:HazusBridge", Path("assets/fragility/hazus-bridge"))
model = Bridge()  # modified ground-shaking curves
given = {
    "asset": {"hazus_bridge_class": "HWB17", "num_spans": 3, "skew_angle_deg": 20},
    "LocalIntensity": {"type": "SA", "period_s": 1.0, "unit": "g", "value": [0, 0.25, 0.43]},
}
result = model.evaluate(given)
print(result["states"])
print(result["probabilities"])  # rows align with the supplied intensity scenarios
mass = model.evaluate_density({"AssetDamageState": "extensive"}, given)
```

`evaluate()` includes curve IDs, the class-selection basis, raw/final medians,
modifier factors, canonical intensity and units, exceedances, discrete
probabilities, assumptions, source warnings, and source/database hashes.
`distribution()` returns only the probability matrix and refuses unresolved matches.

`Bridge()` above is the family: it classifies among all 28. To use one class as the
asset it is declared to be, bind the wrapper to its file:

```python
hwb17 = Bridge.from_asset_file("assets/fragility/hazus-bridge/HWB17.asset.json")
result = hwb17.evaluate({
    "asset": {"num_spans": 3, "skew_angle_deg": 20},   # no class: the file supplies it
    "LocalIntensity": {"sa_1p0_g": 0.43},
})
```

A bound wrapper refuses metadata naming another class, and refuses
`class_probabilities`, which is a statement about the family. Constructor options
(`hazard_mode`, `application`, `bridge_pgd_policy`, `vintage_policy`) pass through
`from_asset_file` unchanged.

`SampleForward`, `EvaluateDensity` (categorical probability mass) and
`EvaluateLikelihood` are declared. Likelihood is not normalised over the intensity
parents. BatchClassify, Gradient and NormalisingConstant are not declared.
Seed a NumPy Generator for reproducible sampling. A scalar intensity gives n draws;
a batch gives one draw per row, with n=1 or n equal to the batch size.
Draws are marginal for one asset, not a spatially correlated multi-asset sample.

Run the checked examples from the project root:

```powershell
conda activate dra
python -m src.fragility.hazus61 experiments/hazus61-fragility/bridge-query.json --output experiments/hazus61-fragility/bridge-result.json
python -m src.fragility.hazus61 experiments/hazus61-fragility/roadway-query.json --output experiments/hazus61-fragility/roadway-result.json
python -m src.conformance.check assets/fragility
python -m pytest -q assets/fragility src/fragility
```

## Input and matching contract

Asset metadata are fixed context; the declared stochastic parent is `LocalIntensity`.
For automatic bridge classification provide `state` (US postal code) or
`is_california`, `year_built`, `nbi_class`, `num_spans`, `max_span_length_m`, and
`bridge_length_m` as needed to distinguish the applicable rules. Explicit
`hazus_bridge_class` / `hazus_roadway_class` takes precedence over automatic matching.

`model.match(asset, hazard=None)` returns base curves when a class is known but
modifier inputs are missing. It returns candidate classes and missing fields when
classification is unresolved. It does not silently classify missing metadata as
HWB28. Candidate enumeration spans the rule partitions; it does not fill missing
inventory values. Major-bridge and single-span rules take priority; narrower
material rows precede overlapping broad rows, matching source table order.

Roadway matching uses `class_hint`, then major `route_type`, then `lanes`.
Short descriptions support explicit class codes and a bounded set of aliases,
lane/span counts, NBI codes, construction years and California labels. This is
not a general natural-language interpreter. Unparsed information stays unknown;
prefer structured records for production calls.

Foreign bridges need `country` and a nonempty, named `project_mapping`; the code
does not invent an NBI mapping. `vintage_policy="table_7_1"` uses the supplied table.
The named alternative `section_7_1_3_prose` uses the reversed prose thresholds.
Every bridge classification carries the vintage and maximum-span/total-length
source warnings. A construction-year classification is not an ageing model.

Use explicit metre fields for bridge length/width. Alternatively, `bridge_length`
and `span_width` require a common `length_unit` (m, cm, mm, ft, in); these are
converted before classification or modifier evaluation.

For hazard input, use the generic `{type, value, unit}` object (plus `period_s=1.0`
for SA), or explicit `sa_1p0_g` / `pgd_in` fields. PGD accepts in, inch, inches, mm,
cm, m and ft. SA accepts g, m/s2, m/s^2, cm/s2, cm/s^2 and Gal. SI acceleration
conversion uses 9.80665 m/s² per g. Site amplification is upstream of this asset.
No PGA→SA or acceleration→PGD conversion is inferred.

Shape-sensitive bridge classes additionally need `sa_0p3_g`, or
`sa_0p3={"value": ..., "unit": "g"}`, in `LocalIntensity`. Short-period SA must
be positive and scalar or match the Sa1 batch. Skew uses degrees in [0,90).
At zero Sa1 all exceedances are zero by the specification's convention, even
when the shape formula also gives a zero slight median; no log(0/0) is evaluated.

## Explicit policies and source limitations

`application="modified"` is the default and refuses missing modifiers. Choose
`application="base"` to evaluate tabulated curves intentionally; output is labelled
`base_curve_only`. No automatic fallback to unmodified curves occurs.

Three asset files state `"application": "base"` themselves — HWB3, HWB4 and HWB28 —
because the source cannot supply their modified shaking curve at all (see the errors
below). That is a declaration in the file, visible in its `validityScope`, and not a
fallback: the result is still labelled `base_curve_only`, an explicit
`application="modified"` still raises, and the family wrapper is unchanged.

For bridge PGD instantiate with `hazard_mode="ground_failure"` and explicitly select:

- `bridge_pgd_policy="literal_table_and_steps"`: retain Slight; multiply Moderate
  and Extensive medians by f1, Complete by f2.
- `bridge_pgd_policy="extensive_complete_only"`: use Extensive/Complete thresholds;
  set lower exceedances equal to Extensive, giving zero Slight/Moderate point mass.
  This is a named implementation of the narrative restriction, not an extra
  parameter table supplied by FEMA.

The following are errors, not silent corrections:

- Literal single-span HWB3/HWB4 K3D division by zero.
- Table 7-7 zero/singular sine factors for affected zero-skew bridges.
- Missing HWB28 shaking modifier definitions.
- Crossed final thresholds, invalid/nonfinite intensities or parameters, unknown
  units, unsupported classes, conflicting unit representations or mixed batch shapes.

No source-repair policy is implemented. A model requiring a different angle
convention or singularity treatment needs a separately sourced implementation.

`class_probabilities={"HWB17": 0.4, "HWB18": 0.6}` can be supplied in `given`
instead of a single class. Weights must be nonnegative and sum to one; only
floating-point roundoff within 1e-12 is normalised, and this is recorded. Every
component retains its own provenance and must be evaluable under the chosen policy.

`evaluate_bridge_modes()` returns separate GS and GF evaluations. Only the explicit
`combination_policy="general_hazus_combination_rule"` adds their independent union.
This is labelled as the general Hazus rule, not a bridge-subsection equation.
For uncertain classes, combination occurs within each class before marginalising;
otherwise latent-class dependence would be lost.

## Source reproduction and remaining integration

The input document is retained byte-for-byte in
`docs/sources/hazus61_roadway_bridge_fragility_prompt.md`. Regenerate the 30 asset
files with:

```powershell
python scripts/import_hazus61_spec.py
```

The importer reads Markdown data tables. It does not execute the coding prompt
embedded in that file. The contract fields of each file are fixed in the importer,
since the specification supplies numbers and not capabilities. A test holds every
file on disk to what the importer produces and fails on a file it does not, because
any `*.asset.json` in these two folders is read into the tables.
Tests cover all curve medians/betas, each classifier row,
all K3D/PGD rows, probability identities, conversions and refusal cases. This
verifies reproduction of the supplied transcription, not agreement with Hazus
software. The [FEMA source PDF](https://www.fema.gov/sites/default/files/documents/fema_hazus-earthquake-model-technical-manual-6-1.pdf)
was located, but its download returned HTTP 403. PDF/manual page coordinates
are null where not independently verified; table numbers are retained.

Neither asset accepts `AssetHealth`: the supplied tables do not parameterise
deterioration, corrosion, or inspection condition. Those candidates remain in the
separate pending source review. Neither asset supplies restoration or a
damage-to-functionality rule. Roadway `extensive_complete` must remain a separate
state-space identifier when adding such a rule.

The existing `src.compiler.pilot` assembles a binary deterministic system factor.
It does not yet automatically chain these stochastic fragilities into a full BN.
That requires a supported damage-to-functionality mapping and compiler support
for intermediate categorical factors. No arbitrary capacity ratios have been added.
