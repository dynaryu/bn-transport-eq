# GMPE (ground-motion prediction equation)

Supplies the intensity of shaking at one asset's site, given an earthquake scenario.
A GMPE says

```
ln Y = mu(rupture, distance, site) + eps,    eps ~ N(0, sigma)
```

so the factor is a lognormal conditional density, and everything that distinguishes
one of these models from another lives in `mu`, `sigma` and the coefficient table
behind them.

**Each model family is its own asset file**, `<model_id>.asset.json` in this
folder — thirty of them, with `cap:assetId` `hazard/gmpe/<model_id>`. A file holds
the family's contract, its source record and, under `parameters`, everything the
catalogue pins down about it. Each observes the
[schema derived from the ontology](../../README.md#single-file-assets); `model.py` is
the wrapper they share, and `params/conventions.json` explains the vocabulary their
`parameters` blocks use.

## State at version 0.1.0 — read this first

**No published GMPE in this asset can predict yet**, but the reason has narrowed.
Anchor-branch coefficients (normally PGA) are now available for 22 of the 30 from the
benchmark specification's coefficient appendix; what is still missing is the *spectral*
tables and, for every model, a `MedianKernel` implementing the published equation.
`coefficients.status` only becomes `present` when a coefficient file with provenance
and a registered kernel both exist, so construction still raises:

```python
>>> Gmpe("akkar-bommer-2010")
CoefficientsUnavailable: Akkar & Bommer (2010) ... coefficients are anchor_only.

Reproduction status: ANCHOR_BRANCH_REPRODUCIBLE (branch: PGA)
  An anchor branch (PGA) IS reproducible from the spec appendix. What is missing is
  the spectral table, and no MedianKernel is implemented for this model yet.
  Implementing the anchor branch is unblocked; off-anchor periods must still be
  refused rather than interpolated from it.
...
```

The refusal distinguishes three situations on purpose, because they call for
different actions:

| `coefficients.status` | Models | What it means |
|---|---|---|
| `anchor_only` | 22 | One branch is numerically complete. Implementable now. |
| `partial` | 3 | Not even the anchor branch is complete — see below. |
| `original_table_required` | 4 | Equation and metadata only; no numbers at all. |
| `unavailable_by_nature` | 1 | Beyer & Bommer: the catalogue never gives a form. |

Telling someone to go and find a table they already have wastes their afternoon,
which is why the message names the available anchor rather than saying "missing".

The three `partial` models each fail for a different reason, and the reason is the
useful part:

- **Youngs et al. (1997)** — rock and deep-soil constants are given, but not the
  event-type term, so neither interface nor intraslab can be evaluated.
- **Hong et al. (2009b)** — the intraslab equation needs `c4·log10(R)` and no
  numerical `c4` is reported. The appendix says explicitly not to assume one.
- **Musson et al. (1994)** — PGA model 1 is complete; model 2 depends on coefficients
  from **Dahle et al. (1990b)**, a second source entirely.

A GMPE with plausible-looking wrong coefficients returns a confident number that no
downstream test can catch, because every value it produces is in the right order of
magnitude. `test_no_published_model_can_predict_yet` fails the day the first kernel
lands, to make you narrow it rather than forget it.

### The completeness gate

`ANCHOR_BRANCH_REPRODUCIBLE` does **not** license the spectral periods. Generating a
period grid from an anchor coefficient is the specific error the appendix's gate
exists to prevent. The published grid is now recorded in the registry for 18 of the
30 (see *What is still unresolved*), but that is metadata: the kernel validates
periods against its own coefficient file, so off-grid periods are still refused and
no spectral prediction is possible until that file exists. `reproduction_status()`
returns the gate per model.

The default `model_id` is `_verification`: a **synthetic** kernel with invented
coefficients, labelled as such in its own parameter file, which exists only so the
framework below can be tested. Every prediction it returns carries a `SYNTHETIC`
warning. It is not a GMPE and must never be used for hazard assessment.

## What it maps

`HazardEvent` → `LocalIntensity` at a single site.

| Side | Content |
|---|---|
| Parents | rupture (magnitude, mechanism, event type, geometry), distance, site |
| Target | intensity measure: PGA, SA/PSA, PSV or SD |
| Units at the API | `g` for acceleration, `cm/s` for velocity, `cm` for displacement |
| Internally | each model's own units and log base, preserved and reported in provenance |

## Capabilities, and why each holds

| Capability | Holds because |
|---|---|
| `SampleForward` | `Y = exp(mu + sigma·eps)`, one site at a time |
| `EvaluateDensity` | the lognormal density is closed-form once `mu` and `sigma` exist |
| `EvaluateLikelihood` | the same ordinate read over the parents — this is what makes inference from a recorded intensity possible |

The undeclared half is the more informative one:

- **`NormalisingConstant`** — not declared. The lognormal is normalised over `Y` by
  construction, so `Z = 1` trivially and uselessly. The `Z` an evidence calculation
  actually needs is over the upstream hazard distribution, which this asset does not
  hold.
- **`BatchClassify`** — not declared. The target is continuous; there is nothing to
  classify. (`cap:batchable true` is a separate claim: the arithmetic vectorises.)
- **`Gradient`** — not declared. Nothing differentiates the kernel here.

## The spatial correlation gap

`cap:drawKind` is **`cap:Marginal`**, and this is the asset's most consequential
limitation.

A GMPE gives the distribution at *one* site. **None of the 30 benchmark families
carries a spatial correlation model** — the benchmark specification does not mention
one, and it is not an oversight in the catalogue either: correlation models are
published separately. So drawing at several sites and stacking the results imposes
independence across them, which understates the probability that many assets are hit
hard at once. For a network risk calculation that joint tail is the answer, so the
error is not conservative and not small.

This is exactly the failure `assets/README.md` warns about, and the design brief is
explicit that the `LocalIntensity` factor "must carry spatial correlation across asset
sites, not per-site marginals alone". Closing it needs a correlation model
(Jayaram–Baker or similar) as a **separate asset** composed with this one, after which
the pair could declare `cap:Joint`. Until then the compiler should not be allowed to
use this asset to produce a field over multiple assets. Every prediction records
`provenance["spatial_correlation"] = None` so the gap travels with the number.

## What the framework enforces

The non-negotiable rules from the specification, as code rather than as prose. Each
has a test.

| Rule | How it is enforced |
|---|---|
| Preserve the log base | kernels return `mu` in their own base; one conversion to `ln` at the boundary |
| Preserve units | explicit factors; conversion across dimensions raises |
| Units must not touch sigma | a unit change shifts `ln Y` and leaves its spread alone |
| No heuristic distance substitution | the required metric must be present; `Rrup` is never accepted for `Rhypo` |
| Do not invent site variables | a categorical-site model refuses a bare `Vs30` |
| No silent extrapolation | outside the envelope raises, unless `allow_extrapolation=True`, then warns |
| Separate aleatory terms | `tau` and `phi` returned separately; derived `sigma` is flagged as derived |
| Periods and damping are grids | off-grid values raise; no interpolation without a documented rule |
| Coefficients are data | `params/coeffs/`, with provenance — see that folder's README |

Two deliberate refusals worth naming:

- **Vs30 → site class is not applied automatically**, even for the three models
  (Ambraseys, Ghasemi, Lussou) whose papers publish the exact boundaries. The
  boundaries are recorded in each model's asset file; applying them belongs with the
  model's own kernel and coefficient set. Until then the caller maps deliberately and
  the assumption is visible in their code rather than in ours.
- **Tectonic type is never inferred from focal depth.** Six models require an explicit
  `event_type`. Kanno et al. branches on depth at 30 km, and the registry records that
  this is a *depth* branch and not a tectonic one, because conflating the two is how a
  crustal event silently gets intraslab coefficients.

## What the 30 cover

Generated from the thirty asset files; see `coverage_matrix()` for the full table.

| Axis | Spread across the set |
|---|---|
| Tectonic | 21 active shallow crustal, 8 interface, 8 intraslab, 3 stable continental |
| Distance | 14 `Rrup`, 7 `Rhypo`, 6 `Rjb`, 1 `Repi`, 1 `Rc`, 1 unresolved (Chen & Yu) |
| Site | 15 categorical, 5 continuous `Vs30`, 4 fixed condition, 3 categorical-from-`Vs30`, 1 banded, 1 none, 1 unresolved |
| IMT | PGA 25, SA 15, PSA 11, SD 2, PSV 1 |
| Damping | 0.5%, 2%, 5%, 7%, 10%, 20%, 30% |
| Magnitude scale | 26 `Mw`, 1 `ML`, 1 `Ms`, 1 `MJMA`, 1 `Mw` or `mLg` (EPRI, two regressions) |
| Component | 10 geometric mean, 6 GMRotI50, 5 larger, 3 both components as separate observations, 1 arithmetic mean, 1 random, 1 peak SRSS, 2 multiple by design, 1 unresolved |

### What is still unresolved

**20 fields across the 30 are still marked `RESOLVE`**, down from 183 originally and
from 114 before the second pass. That pass read the full Douglas (2011) report
(sections 2 and 4 plus its two summary tables, Illustrations 1 and 2) and the
OpenQuake hazardlib implementations. Each value it filled in records its evidence
under the model's `resolution_provenance`. `metadata()` and `validity()` return the
list per model; `missing_sources()` returns it for all 30.

| Still unresolved | Count | Why neither source settles it |
|---|---|---|
| `periods.grid` | 12 | Douglas gives only the number and range of periods (kept in `periods.published_grid_summary`), and no OpenQuake table exists to check them against |
| `chen-yu-2008a` distance, log base, units | 5 | Douglas codes the distance metric U and states no units; `log` next to `exp` in its form is ambiguous |
| `internal_units` (EPRI) | 1 | Coefficients published as graphs only |
| `component` (Musson) | 1 | Douglas codes it U. Only a third party believes it is the larger component |
| `site.parameterization` (Beyer & Bommer) | 1 | Douglas codes the site term U |

**Rule for period grids.** An OpenQuake table was accepted only where its period
count and range match Douglas's Illustration 2 exactly. Rows OpenQuake adds itself
were left out: Akkar & Bommer's 0.01–0.04 s and 4 s, the 4 s row in Atkinson & Boore
and in Youngs, and two rows beyond 20 s in Cauzzi & Faccioli. These grids come from an
external implementation, so they are labelled `original paper not checked`.

Traps worth naming:

- **Faccioli et al. (2010)** — Douglas gives the units as "cm/s²" for a displacement
  spectrum, which cannot be right. The PGA anchor only gives a plausible value if read
  as m/s² (0.13 g at Mw 6, 10 km), and OpenQuake treats the displacement spectrum as
  cm. So `internal_units` is recorded per IMT, `{"PGA": "m/s2", "SD": "cm"}`, and the
  entry carries a `units_warning`. Confirm against the paper's Table 1.
- **Danciu & Tselentis (2007)** — the site term is ordinal, not binary: `S` is 0, 1 or
  2 for classes B, C, D, all under one coefficient. The earlier reading from the anchor
  alone was wrong. Separately, the published intra/inter sigma labelling appears
  **transposed** in the catalogue. Silently "correcting" it would swap `tau` and
  `phi`, so the doubt is carried forward as a warning on the entry rather than
  resolved.

Where a coefficient block *implies* something without stating it — an unnamed
reference class, a branch labelled horizontal whose component definition is not given
— the inference is recorded as an `anchor_note` and the field itself stays `RESOLVE`.
The inference is visible; it is not promoted to fact. Same for the Zhao-family
version: the coefficient naming (`SR`/`SI`/`SS`/`SSL`/`CH`, `hc` = 15 km) points at
Zhao et al. (2006), and that is recorded as probable rather than settled, because
merging versions of that family is the likeliest way to corrupt it.

Aleatory vocabularies are deliberately **not** normalised onto one scheme: Bindi
publishes event/station terms, Hong adds an orientation term, several models give only
a total sigma, and Ambraseys and Youngs give magnitude-dependent expressions rather
than constants. A station term is not `phi`.

## Thirty assets, one wrapper

Each family is a separate asset, `hazard/gmpe/<model_id>`, declared by its own file.
What a compiler selects on — tectonic type, distance metric, IMT, period range — is
in that file, so a family can be contributed, corrected or withdrawn without touching
the other twenty-nine, and the registry is simply whatever files are in the folder.
The split is at the declaration, not at the wrapper: `model.py` stays one framework,
because the validation rules above must not drift between families.

Thirty contracts for thirty models that cannot run would be thirty overclaims, so
**every file declares `"hasCapability": []`.** That is the honest contract, and the
conformance check holds each file to it: `Gmpe.from_asset_file()` raises
`CoefficientsUnavailable`, the harness reports `data` rather than `ok`, and a file
that claimed `SampleForward` in that state would fail as an overclaim. A family gains
capabilities in the same change that gives it a coefficient file and a kernel — see
`params/coeffs/README.md`.

The synthetic verification kernel is deliberately **not** one of the files. It is a
test fixture in `params/verification-synthetic.json`; an asset file for it would put
invented coefficients in the registry for a compiler to pick up.

## Assumptions

- One site per call. No correlation across sites — see above.
- The caller supplies the rupture geometry. Nothing is derived from a single distance
  measure, and no distance metric is converted into another.
- `HazardEvent` is a scenario, not a rate. Return periods and recurrence belong
  upstream of this factor.
- The aleatory term is taken as published. No single-station sigma, no sigma
  truncation; truncation is a decision for the risk calculation, not for this asset.
- Site response is whatever the chosen model publishes. No generic amplification
  factor is layered on top of a fixed-condition model.
