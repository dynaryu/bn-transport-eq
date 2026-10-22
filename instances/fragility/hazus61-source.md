# Hazus 6.1 Roadway and Bridge Fragility Database - Coding Prompt and Reproduction Specification

## 1. Purpose

Use this file as a **self-contained coding prompt and data specification** for building a roadway/bridge earthquake fragility database and a matching engine. The intended workflow is:

1. receive a user description or structured asset record;
2. normalize it to a canonical roadway or bridge representation;
3. match the asset to the applicable Hazus 6.1 class;
4. select the corresponding fragility curve(s);
5. apply bridge-specific modifiers when sufficient metadata are available;
6. compute cumulative exceedance probabilities and discrete damage-state probabilities; and
7. return both the result and complete provenance explaining why the curve was selected.

The numerical values below are extracted from **FEMA, Hazus Earthquake Model Technical Manual, Hazus 6.1, July 2024**, principally Section 7.1 (Highway Transportation System), Tables 7-1, 7-2, 7-5, 7-6, and 7-7. The generic lognormal fragility form follows Section 5.2 / 5.4.2.

**Scope:** roadways and highway bridges only. Tunnels, rail, ports, airports, utilities, buildings, liquefaction probability, and the generation of PGD itself are outside this database.

---

## 2. Core modeling conventions

Hazus models these highway fragilities as **lognormal exceedance functions**. For a damage-state threshold `ds`, intensity measure `IM`, median threshold `theta_ds`, and lognormal standard deviation `beta_ds`:

```text
P(DS >= ds | IM) = Phi( ln(IM / theta_ds) / beta_ds )
```

where `Phi` is the standard normal CDF.

For ordered cumulative exceedance probabilities

```text
qS = P(DS >= Slight)
qM = P(DS >= Moderate)
qE = P(DS >= Extensive)
qC = P(DS >= Complete)
```

the corresponding **discrete** probabilities are:

```text
P(None)      = 1 - qS
P(Slight)    = qS - qM
P(Moderate)  = qM - qE
P(Extensive) = qE - qC
P(Complete)  = qC
```

Hazus demonstrates this differencing explicitly in the highway bridge example on manual page 7-12.

### Unit contract

Never compute before normalizing units.

- roadway PGD: **inches**
- bridge PGD: **inches**
- bridge spectral acceleration: **g**
- bridge length variables `L` and `W`: use the **same length unit** because Table 7-7 uses their ratio
- skew angle `alpha`: source values are in **degrees**; convert to radians before calling programming-language `sin`/`cos` functions

### Probability guardrails

The implementation must:

- reject `IM < 0`;
- return zero exceedance probability at `IM = 0`;
- require `theta_ds > 0` and `beta_ds > 0`;
- check `1 >= qS >= qM >= qE >= qC >= 0` after all modifiers;
- report, rather than silently repair, non-monotone exceedance curves unless an explicit repair policy is enabled.

---

## 3. Canonical database schema

Use one fragility record per `(asset_class, hazard_mode, damage_state)` and separate classification/modifier tables.

```yaml
FragilityCurve:
  curve_id: string
  model_family: "Hazus 6.1"
  asset_family: roadway | highway_bridge
  asset_class: HRD1 | HRD2 | HWB1..HWB28
  hazard_mode: ground_shaking | ground_failure
  im_type: PGD | SA
  im_period_s: null | 1.0
  im_unit: in | g
  damage_state: slight | moderate | extensive | complete | extensive_complete
  median: float
  beta_ln: float
  modifier_rule_id: null | string
  source:
    document: "Hazus Earthquake Model Technical Manual"
    version: "6.1"
    date: "July 2024"
    section: string
    table: string
    manual_page: string
    pdf_page: int
  notes: [string]
```

Recommended companion tables:

```yaml
AssetClassRule:
  asset_family: roadway | highway_bridge
  asset_class: string
  required_fields: [string]
  rule: string
  priority: int
  source_table: string

ModifierRule:
  modifier_rule_id: string
  expression: string
  required_fields: [string]
  applies_to: [string]
  source_table_or_equation: string
```

Do **not** bury classification logic inside fragility records. This keeps curve data reusable when a new classifier or a probabilistic class-matching layer is added later.

---

# Part A - Roadways

## 4. Roadway asset matching

Hazus uses two roadway classes:

- **HRD1 - Major Road:** interstate and state highways, other roads with four lanes or more, and parkways.
- **HRD2 - Urban Road:** intercity roads and other roads with two lanes.

The minimum roadway input is:

```yaml
roadway:
  geometry: polyline
  class_hint: null | major_road | urban_road
  route_type: null | interstate | state_highway | parkway | other
  lanes: null | integer
  pgd_in: float
```

### Matching policy

Use this deterministic hierarchy when the information is explicit:

```text
if class_hint == major_road -> HRD1
else if class_hint == urban_road -> HRD2
else if route_type in [interstate, state_highway, parkway] -> HRD1
else if lanes >= 4 -> HRD1
else if lanes == 2 -> HRD2
else -> insufficient_information
```

The final three rules are a coding normalization of the manual's descriptions. If the description is ambiguous, return candidate classes instead of inventing a class.

## 5. Roadway damage-state semantics

From Hazus Section 7.1.4:

- **Slight:** slight settlement (a few inches) or offset of the ground.
- **Moderate:** moderate settlement (several inches) or offset of the ground.
- **Extensive:** major settlement of the ground (a few feet).
- **Complete:** major settlement of the ground; Hazus treats the roadway Complete definition as the same as Extensive.

Accordingly, Table 7-5 supplies only three distinct roadway limit-state curves: Slight, Moderate, and Extensive/Complete.

## 6. Roadway fragility parameters - Table 7-5

| Asset class | Damage state | PGD median (in) | beta |
| --- | --- | --- | --- |
| Major Road (HRD1) | Slight | 12 | 0.7 |
| Major Road (HRD1) | Moderate | 24 | 0.7 |
| Major Road (HRD1) | Extensive/Complete | 60 | 0.7 |
| Urban Roads (HRD2) | Slight | 6 | 0.7 |
| Urban Roads (HRD2) | Moderate | 12 | 0.7 |
| Urban Roads (HRD2) | Extensive/Complete | 24 | 0.7 |

Machine-readable block:

```yaml
roadway_fragilities:
  - curve_id: HAZUS61_HRD1_PGD_SLIGHT
    asset_class: HRD1
    hazard_mode: ground_failure
    im: PGD
    im_unit: in
    damage_state: "Slight"
    median: 12
    beta_ln: 0.7
  - curve_id: HAZUS61_HRD1_PGD_MODERATE
    asset_class: HRD1
    hazard_mode: ground_failure
    im: PGD
    im_unit: in
    damage_state: "Moderate"
    median: 24
    beta_ln: 0.7
  - curve_id: HAZUS61_HRD1_PGD_EXTENSIVE_COMPLETE
    asset_class: HRD1
    hazard_mode: ground_failure
    im: PGD
    im_unit: in
    damage_state: "Extensive/Complete"
    median: 60
    beta_ln: 0.7
  - curve_id: HAZUS61_HRD2_PGD_SLIGHT
    asset_class: HRD2
    hazard_mode: ground_failure
    im: PGD
    im_unit: in
    damage_state: "Slight"
    median: 6
    beta_ln: 0.7
  - curve_id: HAZUS61_HRD2_PGD_MODERATE
    asset_class: HRD2
    hazard_mode: ground_failure
    im: PGD
    im_unit: in
    damage_state: "Moderate"
    median: 12
    beta_ln: 0.7
  - curve_id: HAZUS61_HRD2_PGD_EXTENSIVE_COMPLETE
    asset_class: HRD2
    hazard_mode: ground_failure
    im: PGD
    im_unit: in
    damage_state: "Extensive/Complete"
    median: 24
    beta_ln: 0.7
```

### Roadway computation

```python
q_slight = lognormal_exceedance(PGD_in, median_slight, 0.7)
q_moderate = lognormal_exceedance(PGD_in, median_moderate, 0.7)
q_extensive_complete = lognormal_exceedance(PGD_in, median_extensive_complete, 0.7)
```

For applications that require five mutually exclusive states, Hazus does not provide a separate roadway curve that partitions Extensive from Complete. Preserve `Extensive/Complete` as a combined state unless a separate, explicitly sourced rule is introduced.

---

# Part B - Highway bridges

## 7. Bridge inputs required for exact Hazus matching

For damage evaluation Hazus requests:

```yaml
bridge:
  longitude: float | null
  latitude: float | null
  hazus_bridge_class: null | HWB1..HWB28

  # Classification / NBI-type fields
  state: string | null
  is_california: bool | null
  year_built: int | null
  nbi_class: string | null
  num_spans: int | null
  max_span_length_m: float | null

  # Fragility modifiers
  skew_angle_deg: float | null
  span_width: float | null
  bridge_length: float | null

  # Hazard inputs at bridge site
  pga_g: float | null
  sa_0p3_g: float | null
  sa_1p0_g: float | null
  pgd_in: float | null
```

The manual's worked example identifies the following NBI fields:

- NBI 27: year built
- NBI 34: angle of skew
- NBI 43: structural/NBI class code (example: 501 = prestressed concrete, simple span)
- NBI 45: number of spans
- NBI 48: maximum span length (m)
- NBI 49: total bridge length (m)

`PGA` is listed by Hazus as a bridge input, but in the bridge fragility algorithm reproduced below, the fragility calculation itself uses `Sa(0.3 s)`, `Sa(1.0 s)`, and `PGD`; PGA does not appear in Equations 7-1 through 7-4.

The worked example on manual page 7-11 also contains an apparent numerical typo: the input is `Sa(1.0 s)=0.24g`, and the site-amplified result is `0.43g`, but the printed multiplication line shows `1.8 * 2.4g = 0.43g`. The result is consistent with `1.8 * 0.24g`, not `2.4g`. Preserve the original input/result and flag the printed multiplier if using the example as a regression test.

## 8. Hazus bridge classification - Table 7-1

The table below preserves the source rows. Blank cells in the source are shown as `-`.

| Class | NBI class | State | Year built | # spans | Max span length (m) | Length <20 m | K3D eq | Ishape | Design | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HWB1 | All | Non-CA | <1990 | - | > 150 | N/A | EQ1 | 0 | Conventional | Major Bridge - Length >150 m |
| HWB1 | All | CA | <1975 | - | > 150 | N/A | EQ1 | 0 | Conventional | Major Bridge - Length >150 m |
| HWB2 | All | Non-CA | >=1990 | - | > 150 | N/A | EQ1 | 0 | Seismic | Major Bridge - Length > 150 m |
| HWB2 | All | CA | >=1975 | - | > 150 | N/A | EQ1 | 0 | Seismic | Major Bridge - Length >150 m |
| HWB3 | All | Non-CA | <1990 | 1 | - | N/A | EQ1 | 1 | Conventional | Single Span |
| HWB3 | All | CA | <1975 | 1 | - | N/A | EQ1 | 1 | Conventional | Single Span |
| HWB4 | All | Non-CA | >=1990 | 1 | - | N/A | EQ1 | 1 | Seismic | Single Span |
| HWB4 | All | CA | >=1975 | 1 | - | N/A | EQ1 | 1 | Seismic | Single Span |
| HWB5 | 101-106 | Non-CA | <1990 | - | - | N/A | EQ1 | 0 | Conventional | Multi-Col. Bent, Simple Support - Concrete |
| HWB6 | 101-106 | CA | <1975 | - | - | N/A | EQ1 | 0 | Conventional | Multi-Col. Bent, Simple Support - Concrete |
| HWB7 | 101-106 | Non-CA | >=1990 | - | - | N/A | EQ1 | 0 | Seismic | Multi-Col. Bent, Simple Support - Concrete |
| HWB7 | 101-106 | CA | >=1975 | - | - | N/A | EQ1 | 0 | Seismic | Multi-Col. Bent, Simple Support - Concrete |
| HWB8 | 205-206 | CA | <1975 | - | - | N/A | EQ2 | 0 | Conventional | Single Col., Box Girder - Continuous Concrete |
| HWB9 | 205-206 | CA | >=1975 | - | - | N/A | EQ3 | 0 | Seismic | Single Col., Box Girder - Continuous Concrete |
| HWB10 | 201-206 | Non-CA | <1990 | - | - | N/A | EQ2 | 1 | Conventional | Continuous Concrete |
| HWB10 | 201-206 | CA | <1975 | - | - | N/A | EQ2 | 1 | Conventional | Continuous Concrete |
| HWB11 | 201-206 | Non-CA | >=1990 | - | - | N/A | EQ3 | 1 | Seismic | Continuous Concrete |
| HWB11 | 201-206 | CA | >=1975 | - | - | N/A | EQ3 | 1 | Seismic | Continuous Concrete |
| HWB12 | 301-306 | Non-CA | <1990 | - | - | No | EQ4 | 0 | Conventional | Multi-Col. Bent, Simple Support - Steel |
| HWB13 | 301-306 | CA | <1975 | - | - | No | EQ4 | 0 | Conventional | Multi-Col. Bent, Simple Support - Steel |
| HWB14 | 301-306 | Non-CA | >=1990 | - | - | N/A | EQ1 | 0 | Seismic | Multi-Col. Bent, Simple Support - Steel |
| HWB14 | 301-306 | CA | >=1975 | - | - | N/A | EQ1 | 0 | Seismic | Multi-Col. Bent, Simple Support - Steel |
| HWB15 | 402-410 | Non-CA | <1990 | - | - | No | EQ5 | 1 | Conventional | Continuous Steel |
| HWB15 | 402-410 | CA | <1975 | - | - | No | EQ5 | 1 | Conventional | Continuous Steel |
| HWB16 | 402-410 | Non-CA | >=1990 | - | - | N/A | EQ3 | 1 | Seismic | Continuous Steel |
| HWB16 | 402-410 | CA | >=1975 | - | - | N/A | EQ3 | 1 | Seismic | Continuous Steel |
| HWB17 | 501-506 | Non-CA | <1990 | - | - | N/A | EQ1 | 0 | Conventional | Multi-Col. Bent, Simple Support - Prestressed Concrete |
| HWB18 | 501-506 | CA | <1975 | - | - | N/A | EQ1 | 0 | Conventional | Multi-Col. Bent, Simple Support - Prestressed Concrete |
| HWB19 | 501-506 | Non-CA | >=1990 | - | - | N/A | EQ1 | 0 | Seismic | Multi-Col. Bent, Simple Support - Prestressed Concrete |
| HWB19 | 501-506 | CA | >=1975 | - | - | N/A | EQ1 | 0 | Seismic | Multi-Col. Bent, Simple Support - Prestressed Concrete |
| HWB20 | 605-606 | CA | <1975 | - | - | N/A | EQ2 | 0 | Conventional | Single Col., Box Girder - Prestressed Continuous Concrete |
| HWB21 | 605-606 | CA | >=1975 | - | - | N/A | EQ3 | 0 | Seismic | Single Col., Box Girder - Prestressed Continuous Concrete |
| HWB22 | 601-607 | Non-CA | <1990 | - | - | N/A | EQ2 | 1 | Conventional | Continuous Concrete |
| HWB22 | 601-607 | CA | <1975 | - | - | N/A | EQ2 | 1 | Conventional | Continuous Concrete |
| HWB23 | 601-607 | Non-CA | >=1990 | - | - | N/A | EQ3 | 1 | Seismic | Continuous Concrete |
| HWB23 | 601-607 | CA | >=1975 | - | - | N/A | EQ3 | 1 | Seismic | Continuous Concrete |
| HWB24 | 301-306 | Non-CA | <1990 | - | - | Yes | EQ6 | 0 | Conventional | Multi-Col. Bent, Simple Support - Steel |
| HWB25 | 301-306 | CA | <1975 | - | - | Yes | EQ6 | 0 | Conventional | Multi-Col. Bent, Simple Support - Steel |
| HWB26 | 402-410 | Non-CA | <1990 | - | - | Yes | EQ7 | 1 | Conventional | Continuous Steel |
| HWB27 | 402-410 | CA | <1975 | - | - | Yes | EQ7 | 1 | Conventional | Continuous Steel |
| HWB28 | - | - | - | - | - | - | - | - | - | All other bridges that are not classified |

### Important classification precedence

Table 7-1 contains overlapping rules because HWB1-HWB4 use `NBI Class = All`. To create a deterministic matcher, apply the broad special cases before material-specific classes:

```text
1. major-bridge rule (HWB1/HWB2)
2. single-span rule (HWB3/HWB4)
3. NBI-class/material/continuity rules (HWB5-HWB27)
4. HWB28 only when the bridge genuinely falls outside the defined classes
```

This precedence is an implementation inference needed to make the table mutually exclusive; the manual does not present it as pseudocode.

### Do not use HWB28 as a missing-data default

`HWB28` means "all other bridges that are not classified" in Table 7-1. It should **not** be used merely because user metadata are missing. If the input is incomplete, return `insufficient_information` plus plausible candidate classes, or a user-supplied probability mixture over classes.

### Native applicability of the taxonomy

The bridge classification is explicitly tied to U.S. National Bridge Inventory (NBI) classes and a California/non-California design-vintage split. For a bridge outside the native NBI taxonomy, do not fabricate an NBI code. Either (a) require an explicit project mapping from local structural attributes to Hazus/NBI classes, or (b) return `out_of_native_taxonomy` with candidate structural analogues and a reduced confidence level. This rule makes the database extensible to additional fragility sources later without pretending Hazus is universally native.

## 9. Source inconsistency that the code must preserve and flag

There is a visible inconsistency in Section 7.1.3:

- the prose states that seismic vintage is "1990 or later in California, and 1975 or later outside of California";
- Table 7-1 explicitly uses the opposite assignment: **California >=1975**, **Non-California >=1990**.

For literal reproduction of Table 7-1, this specification uses the **table values**. Store a provenance warning in every automated classification result and allow a caller to override the vintage policy.

There is a second ambiguity: Table 7-1 puts `>150` under the column headed `Length of Max. Span (meter)`, whereas the description says `Major Bridge - Length >150 m`. Preserve both source statements in provenance; do not silently reinterpret one as total bridge length.

---

## 10. K3D equations - Table 7-2

Hazus defines:

```text
K3D = 1 + A / (N - B)
```

where `N` is the number of spans and `(A,B)` depend on the equation assigned in Table 7-1.

| Equation | A | B | Expression |
| --- | --- | --- | --- |
| EQ1 | 0.25 | 1 | 1 + 0.25 / (N - 1) |
| EQ2 | 0.33 | 0 | 1 + 0.33 / (N - 0) |
| EQ3 | 0.33 | 1 | 1 + 0.33 / (N - 1) |
| EQ4 | 0.09 | 1 | 1 + 0.09 / (N - 1) |
| EQ5 | 0.05 | 0 | 1 + 0.05 / (N - 0) |
| EQ6 | 0.2 | 1 | 1 + 0.2 / (N - 1) |
| EQ7 | 0.1 | 0 | 1 + 0.1 / (N - 0) |

```yaml
k3d_equations:
  EQ1: {A: 0.25, B: 1, expression: "1 + A / (N - B)"}
  EQ2: {A: 0.33, B: 0, expression: "1 + A / (N - B)"}
  EQ3: {A: 0.33, B: 1, expression: "1 + A / (N - B)"}
  EQ4: {A: 0.09, B: 1, expression: "1 + A / (N - B)"}
  EQ5: {A: 0.05, B: 0, expression: "1 + A / (N - B)"}
  EQ6: {A: 0.2, B: 1, expression: "1 + A / (N - B)"}
  EQ7: {A: 0.1, B: 0, expression: "1 + A / (N - B)"}
```

Guardrail: if `N - B <= 0`, do not evaluate. **This is not merely hypothetical:** Table 7-1 assigns `EQ1` to the single-span classes HWB3/HWB4 while also specifying `N = 1`, and Table 7-2 gives `EQ1 = 1 + 0.25/(N-1)`. Taken literally, the manual therefore produces a singular `K3D` for those classes. Do not invent a numerical repair. Return `source_formula_singularity` unless a separately documented Hazus-software/original-reference rule is supplied.

---

## 11. Bridge base fragility medians - Table 7-6

### Ground shaking

- base IM: `Sa(1.0 s)` in g
- base beta: `0.6`
- medians are subsequently modified using skew, shape, and 3-D effects

### Ground failure

- IM: `PGD` in inches
- beta: `0.2`
- PGD medians are subsequently modified using Table 7-7 factors

**Applicability ambiguity to preserve:** the narrative immediately before Table 7-6 says that only incipient unseating and collapse (Extensive and Complete) are considered possible bridge damage types due to ground failure, and that initial bearing damage corresponding to Slight/Moderate is not considered. Nevertheless, Table 7-6 contains PGD medians under all four damage-state columns, and Step 6 explicitly modifies Moderate, Extensive, and Complete medians. Store both the tabulated values and the narrative restriction. Do not silently choose one interpretation; expose a named implementation policy.

| Class | GS Slight Sa1 (g) | GS Moderate | GS Extensive | GS Complete | PGD Slight (in) | PGD Moderate | PGD Extensive | PGD Complete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HWB1 | 0.40 | 0.50 | 0.70 | 0.90 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB2 | 0.60 | 0.90 | 1.10 | 1.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB3 | 0.80 | 1.00 | 1.20 | 1.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB4 | 0.80 | 1.00 | 1.20 | 1.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB5 | 0.25 | 0.35 | 0.45 | 0.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB6 | 0.30 | 0.50 | 0.60 | 0.90 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB7 | 0.50 | 0.80 | 1.10 | 1.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB8 | 0.35 | 0.45 | 0.55 | 0.80 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB9 | 0.60 | 0.90 | 1.30 | 1.60 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB10 | 0.60 | 0.90 | 1.10 | 1.50 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB11 | 0.90 | 0.90 | 1.10 | 1.50 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB12 | 0.25 | 0.35 | 0.45 | 0.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB13 | 0.30 | 0.50 | 0.60 | 0.90 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB14 | 0.50 | 0.80 | 1.10 | 1.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB15 | 0.75 | 0.75 | 0.75 | 1.10 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB16 | 0.90 | 0.90 | 1.10 | 1.50 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB17 | 0.25 | 0.35 | 0.45 | 0.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB18 | 0.30 | 0.50 | 0.60 | 0.90 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB19 | 0.50 | 0.80 | 1.10 | 1.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB20 | 0.35 | 0.45 | 0.55 | 0.80 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB21 | 0.60 | 0.90 | 1.30 | 1.60 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB22 | 0.60 | 0.90 | 1.10 | 1.50 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB23 | 0.90 | 0.90 | 1.10 | 1.50 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB24 | 0.25 | 0.35 | 0.45 | 0.70 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB25 | 0.30 | 0.50 | 0.60 | 0.90 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB26 | 0.75 | 0.75 | 0.75 | 1.10 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB27 | 0.75 | 0.75 | 0.75 | 1.10 | 3.9 | 3.9 | 3.9 | 13.8 |
| HWB28 | 0.80 | 1.00 | 1.20 | 1.70 | 3.9 | 3.9 | 3.9 | 13.8 |

Machine-readable block:

```yaml
bridge_fragility_base_medians:
  HWB1:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.4, moderate: 0.5, extensive: 0.7, complete: 0.9}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB2:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.6, moderate: 0.9, extensive: 1.1, complete: 1.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB3:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.8, moderate: 1, extensive: 1.2, complete: 1.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB4:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.8, moderate: 1, extensive: 1.2, complete: 1.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB5:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.25, moderate: 0.35, extensive: 0.45, complete: 0.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB6:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.3, moderate: 0.5, extensive: 0.6, complete: 0.9}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB7:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.5, moderate: 0.8, extensive: 1.1, complete: 1.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB8:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.35, moderate: 0.45, extensive: 0.55, complete: 0.8}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB9:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.6, moderate: 0.9, extensive: 1.3, complete: 1.6}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB10:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.6, moderate: 0.9, extensive: 1.1, complete: 1.5}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB11:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.9, moderate: 0.9, extensive: 1.1, complete: 1.5}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB12:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.25, moderate: 0.35, extensive: 0.45, complete: 0.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB13:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.3, moderate: 0.5, extensive: 0.6, complete: 0.9}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB14:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.5, moderate: 0.8, extensive: 1.1, complete: 1.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB15:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.75, moderate: 0.75, extensive: 0.75, complete: 1.1}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB16:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.9, moderate: 0.9, extensive: 1.1, complete: 1.5}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB17:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.25, moderate: 0.35, extensive: 0.45, complete: 0.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB18:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.3, moderate: 0.5, extensive: 0.6, complete: 0.9}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB19:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.5, moderate: 0.8, extensive: 1.1, complete: 1.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB20:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.35, moderate: 0.45, extensive: 0.55, complete: 0.8}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB21:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.6, moderate: 0.9, extensive: 1.3, complete: 1.6}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB22:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.6, moderate: 0.9, extensive: 1.1, complete: 1.5}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB23:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.9, moderate: 0.9, extensive: 1.1, complete: 1.5}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB24:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.25, moderate: 0.35, extensive: 0.45, complete: 0.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB25:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.3, moderate: 0.5, extensive: 0.6, complete: 0.9}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB26:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.75, moderate: 0.75, extensive: 0.75, complete: 1.1}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB27:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.75, moderate: 0.75, extensive: 0.75, complete: 1.1}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
  HWB28:
    ground_shaking:
      im: SA
      period_s: 1.0
      unit: g
      beta_ln: 0.6
      median: {slight: 0.8, moderate: 1, extensive: 1.2, complete: 1.7}
    ground_failure:
      im: PGD
      unit: in
      beta_ln: 0.2
      median: {slight: 3.9, moderate: 3.9, extensive: 3.9, complete: 13.8}
```

---

## 12. Bridge ground-shaking modifiers - Equations 7-1 to 7-4

Given skew angle `alpha` in degrees, `Sa03 = Sa(0.3 s)`, `Sa10 = Sa(1.0 s)`, number of spans `N`, and Table 7-2 coefficients `A`,`B`:

```text
Kskew  = sqrt( sin(90 deg - alpha) )
       = sqrt( cos(alpha) )                 # equivalent if angle handling is consistent

Kshape = 2.5 * Sa10 / Sa03

K3D    = 1 + A / (N - B)
```

**Programming note:** convert degrees to radians before evaluating trigonometric functions.

Modify the Table 7-6 shaking medians as follows:

```text
Factor_slight = 1                         if Ishape == 0
Factor_slight = min(1, Kshape)           if Ishape == 1

new_theta_slight   = old_theta_slight   * Factor_slight
new_theta_moderate = old_theta_moderate * Kskew * K3D
new_theta_extensive= old_theta_extensive* Kskew * K3D
new_theta_complete = old_theta_complete * Kskew * K3D
```

Then compute exceedance using **Sa(1.0 s)** and `beta = 0.6`:

```python
q_ds = Phi(math.log(sa_1p0_g / new_theta_ds) / 0.6)
```

### Guardrails for the shaking modifier

- `Sa03` must be positive when `Ishape == 1` because it appears in the denominator of `Kshape`.
- valid physical skew input should be checked before computing `sqrt(cos(alpha))`; if `cos(alpha) < 0`, reject rather than returning a complex value.
- use the class-specific `K3D` equation from Table 7-1.

---

## 13. Bridge PGD median modifiers - Table 7-7

Let:

- `N` = number of spans
- `W` = span width
- `L` = bridge length
- `alpha` = skew angle

Use the same units for `L` and `W`.

| Class | f1 | f2 |
| --- | --- | --- |
| HWB1 | 1 | 1 |
| HWB2 | 1 | 1 |
| HWB3 | 1 | 1 |
| HWB4 | 1 | 1 |
| HWB5 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB6 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB7 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB8 | 1 | sin(alpha) |
| HWB9 | 1 | sin(alpha) |
| HWB10 | 1 | sin(alpha) |
| HWB11 | 1 | sin(alpha) |
| HWB12 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB13 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB14 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB15 | 1 | sin(alpha) |
| HWB16 | 1 | sin(alpha) |
| HWB17 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB18 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB19 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB20 | 1 | sin(alpha) |
| HWB21 | 1 | sin(alpha) |
| HWB22 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB23 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB24 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB25 | 0.5 * L / (N * W * sin(alpha)) | 0.5 * L / (N * W * sin(alpha)) |
| HWB26 | 1 | sin(alpha) |
| HWB27 | 1 | sin(alpha) |
| HWB28 | 1 | 1 |

```yaml
bridge_pgd_modifiers:
  HWB1: {f1: "1", f2: "1"}
  HWB2: {f1: "1", f2: "1"}
  HWB3: {f1: "1", f2: "1"}
  HWB4: {f1: "1", f2: "1"}
  HWB5: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB6: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB7: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB8: {f1: "1", f2: "sin(alpha)"}
  HWB9: {f1: "1", f2: "sin(alpha)"}
  HWB10: {f1: "1", f2: "sin(alpha)"}
  HWB11: {f1: "1", f2: "sin(alpha)"}
  HWB12: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB13: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB14: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB15: {f1: "1", f2: "sin(alpha)"}
  HWB16: {f1: "1", f2: "sin(alpha)"}
  HWB17: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB18: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB19: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB20: {f1: "1", f2: "sin(alpha)"}
  HWB21: {f1: "1", f2: "sin(alpha)"}
  HWB22: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB23: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB24: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB25: {f1: "0.5 * L / (N * W * sin(alpha))", f2: "0.5 * L / (N * W * sin(alpha))"}
  HWB26: {f1: "1", f2: "sin(alpha)"}
  HWB27: {f1: "1", f2: "sin(alpha)"}
  HWB28: {f1: "1", f2: "1"}
```

Hazus Step 6 states:

```text
new_PGD_theta_moderate = old_PGD_theta_moderate * f1
new_PGD_theta_extensive = old_PGD_theta_extensive * f1
new_PGD_theta_complete = old_PGD_theta_complete * f2
```

The manual does not state a Step-6 modification for the Slight PGD median. Retain its Table 7-6 value in the raw database, but remember the narrative statement that ground-failure Slight/Moderate bridge damage is not considered. A computation API should therefore require an explicit `bridge_pgd_policy`, for example `literal_table_and_steps` or `extensive_complete_only`, and report which policy was used.

Then compute PGD exceedance probabilities with `beta = 0.2` only under the selected policy.

### Critical PGD formula warning

Table 7-7 literally contains factors involving `sin(alpha)` and, for many classes, `1/sin(alpha)`. With the Section 7 definition of skew angle, an unskewed bridge can have `alpha = 0`, producing zero or singular factors. This specification **does not silently correct that source behavior**. A robust implementation must:

1. store the literal Hazus expression;
2. detect `abs(sin(alpha)) < tolerance`;
3. return a `source_formula_singularity` warning/error unless the caller has provided an explicit project-specific interpretation;
4. never substitute an alternative angle convention without recording it.

---

## 14. Bridge damage-state meanings

Use these semantics when a probabilistic model needs a state description.

### Slight

Minor cracking/spalling at abutments, hinges or columns; cracks in shear keys; minor deck cracking; cosmetic repair only.

### Moderate

Moderate column cracking/spalling while structurally sound; abutment movement under 2 inches; extensive shear-key cracking/spalling; cracked shear keys or bent bolts at connections; keeper-bar failure without unseating; rocker-bearing failure; moderate approach settlement.

### Extensive

Column degradation without collapse, including shear failure that renders the column structurally unsafe; significant residual movement at connections; major approach settlement; vertical abutment offset; differential settlement at connections; or shear-key failure at abutments.

### Complete

Column collapse or loss of all bearing support at a connection with possible/imminent deck collapse, or tilting of the substructure due to foundation failure.

---

## 15. Cumulative-to-discrete damage function

Reference implementation:

```python
from math import log
from statistics import NormalDist

_STD_NORMAL = NormalDist()

def lognormal_exceedance(im: float, median: float, beta: float) -> float:
    if im < 0:
        raise ValueError("IM must be non-negative")
    if im == 0:
        return 0.0
    if median <= 0 or beta <= 0:
        raise ValueError("median and beta must be positive")
    return _STD_NORMAL.cdf(log(im / median) / beta)


def discrete_from_exceedance(q_s, q_m, q_e, q_c):
    if not (1 >= q_s >= q_m >= q_e >= q_c >= 0):
        raise ValueError("Non-monotone cumulative damage probabilities")
    return {
        "none": 1 - q_s,
        "slight": q_s - q_m,
        "moderate": q_m - q_e,
        "extensive": q_e - q_c,
        "complete": q_c,
    }
```

For roadways, keep the final state as `extensive_complete` unless a separate source provides a valid partition.

---

## 16. Combining bridge ground shaking and ground failure

Section 7.1.6.2 Step 8 says to combine damage-state probabilities and evaluate functionality but does not print the combination equation on that page. Elsewhere, Hazus Section 5.6.2 treats ground shaking and ground failure as independent and combines exceedance probabilities as the probability of their union.

If reproducing that general Hazus independence assumption for bridges, use:

```text
q_combined(ds) = 1 - (1 - q_GS(ds)) * (1 - q_GF(ds))
```

then difference the combined cumulative probabilities to obtain discrete damage states.

**Provenance rule:** label this as `general_hazus_combination_rule`, not as an equation explicitly printed in the bridge subsection.

If a project uses a dependence model between shaking and ground failure, replace this union rule with the project model and retain the two hazard-mode fragilities separately.

---

## 17. Optional restoration layer for network probabilistic models

Hazus gives normal restoration-time distributions (manual Table 7-3):

| Asset | Damage state | Mean days | sigma days |
| --- | --- | --- | --- |
| Roadways | Slight | 0.9 | 0.05 |
| Roadways | Moderate | 2.2 | 1.8 |
| Roadways | Extensive/Complete | 21 | 16 |
| Highway Bridges | Slight | 0.6 | 0.6 |
| Highway Bridges | Moderate | 2.5 | 2.7 |
| Highway Bridges | Extensive | 75 | 42 |
| Highway Bridges | Complete | 230 | 110 |

Keep restoration separate from fragility. A probabilistic network model can condition a `Functional(t)` node on damage state using these restoration distributions.

---

# Part C - Matching engine

## 18. Required matching behavior

The database/matcher must distinguish three cases.

### A. Exact match

All fields needed for a unique Hazus class and its modifiers are available.

Return:

```yaml
match_status: exact
asset_class: HWB17
curve_ids: [...]
modifiers_applied: [...]
missing_fields: []
provenance: [...]
```

### B. Class known, modifiers incomplete

Example: user states `HWB17` but gives no skew angle or span width.

Return the **base Table 7-6 curves**, but do not claim they are fully asset-specific:

```yaml
match_status: base_curve_only
asset_class: HWB17
missing_fields: [skew_angle_deg, span_width, bridge_length, num_spans]
can_compute_base_fragility: true
can_compute_modified_fragility: false
```

### C. Asset class uncertain

Do not choose the first plausible class. Return candidates and required disambiguating fields.

```yaml
match_status: ambiguous
candidate_classes: [HWB17, HWB18, HWB19]
needed_to_disambiguate: [state, year_built]
```

If the user supplies explicit class probabilities, a probabilistic model may marginalize over class:

```text
P(DS >= ds | IM, evidence) = sum_c P(DS >= ds | IM, class=c) * P(class=c | evidence)
```

This mixture rule is general probabilistic-model guidance rather than a Hazus-specific rule.

---

## 19. Natural-language normalization vocabulary

These aliases are for query parsing only; they do not change Hazus definitions.

```yaml
aliases:
  roadway:
    [road, roadway, road link, highway link, street segment]
  major_road:
    [major road, interstate, interstate highway, state highway, parkway]
  urban_road:
    [urban road, city road, local road, two-lane road]
  bridge:
    [bridge, highway bridge, road bridge, overpass, viaduct]
  damage_states:
    slight: [slight, minor]
    moderate: [moderate]
    extensive: [extensive, severe]
    complete: [complete, collapse, collapsed]
  intensity_measures:
    PGD: [PGD, permanent ground deformation, permanent ground displacement]
    SA_0p3: [Sa0.3, SA(0.3s), spectral acceleration at 0.3 s]
    SA_1p0: [Sa1.0, SA(1.0s), spectral acceleration at 1.0 s]
```

Never map the word `PGA` to `SA(1.0s)` or `SA(0.3s)`.

---

## 20. Matching algorithm pseudocode

```text
function match_fragility(request):
    asset = normalize_asset_description(request)

    if asset.family == roadway:
        class = classify_roadway(asset)
        if class unresolved:
            return ambiguity_report
        return roadway curves for class

    if asset.family == highway_bridge:
        if user supplied valid HWB class:
            class = supplied class
        else:
            class = classify_bridge_using_table_7_1(asset)

        if class unresolved:
            return candidate classes + missing fields

        curves = load Table-7-6 base curves

        if ground-shaking calculation requested:
            require Sa1.0
            if Ishape == 1: require Sa0.3
            require skew and N if Kskew/K3D needed
            apply Eq. 7-1 to 7-4
            calculate exceedance with beta=0.6

        if ground-failure calculation requested:
            require PGD
            calculate Table-7-7 f1/f2 only when required fields exist
            detect sin(alpha) singularity
            modify PGD medians per Step 6
            calculate exceedance with beta=0.2

        if both hazard modes requested:
            return separate GS and GF results
            optionally return combined result using explicitly selected combination policy
```

---

## 21. Provenance and explanation contract

Every match must retain enough information for a user to audit the selection. Return at least:

```yaml
provenance:
  source_document: Hazus Earthquake Model Technical Manual
  version: 6.1
  date: July 2024
  source_section: 7.1.6.1 | 7.1.6.2
  source_table: 7-5 | 7-6 | 7-7
  source_manual_page: 7-5 | 7-6 | 7-9 | 7-10
  classification_basis:
    - field: nbi_class
      value: "501"
    - field: state
      value: "TN"
    - field: year_built
      value: 1968
  assumptions: []
  warnings: []
```

Never return only a curve number with no matching explanation.

---

## 22. Data-quality and source-fidelity rules

The coding agent must follow all of these rules:

1. **Never invent a fragility median or beta.**
2. **Never convert PGA into SA** without an explicitly supplied model.
3. **Never infer PGD from PGA/SA** inside this database; PGD generation belongs to a hazard module.
4. **Never default an incompletely described bridge to HWB28.**
5. **Never ignore units.** Convert to `g` and `inches` before evaluation.
6. **Keep base curves and modifiers separately queryable.**
7. **Return separate ground-shaking and ground-failure results before any combined result.**
8. **Record all assumptions used in classification.**
9. **Flag the Table 7-1 vintage inconsistency** described in Section 9 of this file.
10. **Flag Table 7-7 trigonometric singularities** rather than silently regularizing them.
11. If a project chooses to reinterpret an apparent source inconsistency, store the alternative as a named policy and preserve the literal Hazus value.
12. When metadata are uncertain, represent uncertainty over candidate classes instead of pretending the class is known.

---

# Part D - Copy-paste prompt for a coding agent

## 23. Master prompt

```text
You are implementing a reproducible earthquake fragility database for roadways and highway bridges.
Use only the Hazus 6.1 values and formulas supplied in this specification unless another source is
explicitly added with separate provenance.

GOAL
Build a library that can receive either structured asset metadata or a natural-language description,
match the asset to a Hazus roadway/bridge class, retrieve the appropriate fragility curve, apply all
required Hazus modifiers, and calculate cumulative and discrete damage-state probabilities.

REQUIRED MODULES
1. data.py
   - immutable roadway Table 7-5 parameters
   - immutable bridge Table 7-6 base medians
   - Table 7-1 bridge classification rules
   - Table 7-2 K3D coefficients
   - Table 7-7 PGD modifier expressions
   - source/provenance metadata

2. schemas.py
   - RoadwayAsset
   - BridgeAsset
   - HazardInput
   - FragilityCurve
   - MatchResult
   - DamageProbabilityResult
   Use explicit units in field names or a unit-aware package.

3. classify.py
   - classify_roadway()
   - classify_bridge()
   - return exact/ambiguous/insufficient-information states
   - do not use HWB28 as a missing-data fallback
   - expose the classification rule and fields that triggered it

4. modifiers.py
   - k_skew(alpha_deg)
   - k_shape(sa_0p3_g, sa_1p0_g)
   - k_3d(eq_id, n_spans)
   - modified_shaking_medians()
   - pgd_factors(hwb_class, N, W, L, alpha_deg)
   - modified_pgd_medians()
   - detect the literal Table 7-7 sin(alpha) singularity

5. probability.py
   - lognormal_exceedance(IM, median, beta)
   - cumulative_to_discrete()
   - optional combine_independent_exceedance(q_gs, q_gf)

6. matcher.py
   - normalize natural-language asset descriptions
   - map aliases to canonical concepts
   - return candidate classes and missing fields when not unique
   - support probabilistic mixtures over classes if weights are provided

7. tests/
   - test every numerical row in Tables 7-2, 7-5, 7-6, 7-7
   - test median property: P(exceedance at IM=median) == 0.5
   - test cumulative-to-discrete probabilities sum to 1
   - test roadway HRD1/HRD2 mapping
   - test bridge classification boundaries around vintage thresholds
   - test Kshape and Kskew equations
   - test singularity handling at sin(alpha)=0
   - test exact example fields from Hazus Table 7-8 classify as HWB17 under Table-7-1 policy

API BEHAVIOR
- Never guess missing numeric inputs.
- Never hallucinate a curve outside the database.
- Never hide a source inconsistency.
- Results must include source table, base medians, modifiers, final medians, beta, IM, cumulative
  exceedance probabilities, discrete probabilities, assumptions, missing inputs, and warnings.
- Keep ground shaking and ground failure as separate conditional models; combine them only when the
  caller explicitly selects a combination policy.

PROBABILITY FORM
P(DS >= ds | IM) = Phi( ln(IM/theta_ds) / beta_ds )

BRIDGE SHAKING
Kskew = sqrt(sin(90deg-alpha))
Kshape = 2.5*Sa(1.0s)/Sa(0.3s)
K3D = 1 + A/(N-B)
Slight median *= 1 if Ishape=0 else min(1,Kshape)
Moderate, Extensive, Complete medians *= Kskew*K3D
Use Sa(1.0s) as the fragility IM and beta=0.6.

BRIDGE GROUND FAILURE
Use PGD in inches, Table-7-6 base medians and beta=0.2.
Modify Moderate and Extensive by f1 and Complete by f2 according to Table 7-7.
Preserve the source's unmodified Slight PGD median unless another documented rule is selected.

ROADWAYS
Use PGD in inches and beta=0.7.
HRD1 medians: Slight=12, Moderate=24, Extensive/Complete=60.
HRD2 medians: Slight=6, Moderate=12, Extensive/Complete=24.

Before producing code, parse this entire specification into typed constants and write tests that compare
all parsed constants against the embedded tables.
```

---

## 24. Suggested output for a fragility lookup request

Example structure:

```yaml
request_summary: "3-span conventional prestressed-concrete highway bridge"
match_status: exact
asset_class: HWB17
hazard_mode: ground_shaking
im:
  type: SA
  period_s: 1.0
  value_g: 0.43
base_medians_g:
  slight: 0.25
  moderate: 0.35
  extensive: 0.45
  complete: 0.70
modifiers:
  k_skew: ...
  k_shape: ...
  k_3d: ...
final_medians_g:
  slight: ...
  moderate: ...
  extensive: ...
  complete: ...
beta_ln: 0.6
exceedance_probability:
  slight: ...
  moderate: ...
  extensive: ...
  complete: ...
discrete_probability:
  none: ...
  slight: ...
  moderate: ...
  extensive: ...
  complete: ...
provenance:
  source: "FEMA Hazus 6.1 Technical Manual"
  tables: ["7-1", "7-2", "7-6"]
  equations: ["7-1", "7-2", "7-3", "7-4"]
warnings: []
```

---

## 25. Source map

- Hazus 6.1 Technical Manual, Section 7.1.1: roadway/bridge inputs
- Section 7.1.2: lognormal form and applicable IMs
- Section 7.1.3 and Table 7-1: highway component descriptions and 28 bridge classes
- Table 7-2: K3D coefficients
- Section 7.1.4: damage-state definitions
- Table 7-5: roadway PGD fragility medians and beta
- Table 7-6: bridge base shaking and PGD medians
- Equations 7-1 through 7-4: bridge shaking modifiers
- Table 7-7: bridge PGD median modifiers
- Table 7-8: example NBI fields used for matching
- Section 5.2 / 5.4.2: generic cumulative lognormal fragility formulation
- Section 5.6: cumulative-to-discrete damage probabilities and general GS/GF combination logic

**Document version:** Hazus Earthquake Model Technical Manual, Hazus 6.1, July 2024.
