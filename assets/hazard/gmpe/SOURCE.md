# Source

**STATUS: catalogue collected; anchor coefficients available for 22 of 30; no
spectral table collected for any. 20 registry fields still `RESOLVE` (from 183).**

The benchmark set, every model's applicability envelope and its completeness status
are recorded per model, in the `parameters` block of its own asset file,
`<model_id>.asset.json`. Run `reproduction_status()` for the gate per
model and `missing_sources()` for what remains to be obtained.

## Citation

John Douglas (2011). *Ground-motion prediction equations 1964–2010.* PEER Report
2011/102, Pacific Earthquake Engineering Research Center, University of California,
Berkeley. Also published as BRGM/RP-59356-FR.

Two revisions of the benchmark specification were used, and the distinction matters:

- `gmpe_30_coding_prompts.md` — the 30-family selection, the inclusion rationale and
  the per-model coding requirements, drawn from Douglas's sections 2 and 4.
- `gmpe_30_coding_prompts_with_coefficients.md` — adds **section 11**, a coefficient
  appendix transcribing the numerical values Douglas actually reports, with a
  per-model completeness status, and **section 12**, a completeness gate. This is
  what resolved `log_base` and `internal_units` for most of the set, and it supplies
  the anchor-branch coefficients. Its own transcription note applies: values are taken
  from Douglas **without** being reconciled against later implementations.

The registry records each model's Douglas section numbers so any entry can be traced
back to the catalogue, and each `reproduction` block cites the appendix.

**Retrieved / written.** 2026-10-04.

## What the appendix resolved, and what it did not

Applying it took the registry's unresolved fields from 183 to 115. Resolved:
`log_base` for 24 models (28 of 30 now known), `internal_units` for 24, seven site
class enumerations, three site parameterisations, Lin & Lee's event types, McVerry's
component definition ("larger component"), the Zhao-family depth threshold (`hc` = 15
km), EPRI's site classes as depth-to-rock bands, Ghasemi's fixed `a5` = 0.42, Idriss's
magnitude break at 6.75, and Bindi's Rjb/Repi question — which turns out not to be a
fallback at all but four separately published coefficient sets.

The appendix could not settle the spectral period grids, the component definition
for 25 models, the calibration distance ranges, or five magnitude scales.

## Second pass: the full report and OpenQuake (2026-10-04)

The appendix is a *summary* of Douglas. The full report was obtained instead: the
BRGM edition, `infoterre.brgm.fr/rapports/RP-59356-FR.pdf`, 446 pages, same content
as PEER 2011/102. Its two summary tables do state many of the fields the appendix
dropped:

- **Illustration 1** (PGA) and **Illustration 2** (spectral) give, per model, the
  magnitude scale, `rmin`/`rmax` and distance metric, a component code (G, I50, L, A,
  B, C, R, U…), and for spectra the number of periods with `Tmin`/`Tmax`.
- The per-model sections 2.x / 4.x define site and mechanism classes and usually
  state the units.

Period grids came from **OpenQuake hazardlib** (`gem/oq-engine`, master, retrieved
2026-10-04). A grid was accepted only where its count and range match Douglas's
Illustration 2. OpenQuake is an external implementation, AGPL-3.0: these values
are *metadata about which periods exist*, not coefficients, and every one is labelled
`original paper not checked`.

That pass took the unresolved fields from 114 to 20. Each value it filled in carries
its evidence in `resolution_provenance` on the model entry. What is still open is
listed in the README. In summary: 12 period grids with no checkable source (count and
range kept in `periods.published_grid_summary`), Chen & Yu's distance metric, log base
and units, EPRI's units, Musson's component, and Beyer & Bommer's site term.

**Corrections the second pass made to earlier readings:**

- **Faccioli et al. (2010)** — Douglas does state units, as "cm/s²". For a
  displacement spectrum that is impossible, and for the PGA anchor it is off by a
  factor of 100. Recorded per IMT as PGA m/s², SD cm, with a `units_warning`.
- **Danciu & Tselentis (2007)** — the site variable is `S ∈ {0, 1, 2}` (classes B, C,
  D) under a single coefficient, not a binary flag. Thrust and strike-slip share
  `F = 1`.
- **Hong et al. (2009b)** — the distance is Rrup (Rhypo for small events). Only the
  missing intraslab `c4` keeps the model `PARTIAL` now.
- **Chen & Yu (2008a)** — no site classes at all. Records are restricted to Vs30 ≥ 500
  m/s, so the site is a fixed condition.

**Caution carried forward rather than resolved.** Danciu & Tselentis (2007)'s
intra/inter sigma labelling appears transposed in the catalogue; "correcting" it
silently would swap `tau` and `phi`, so the entry carries a warning instead. Arroyo
et al. (2010)'s component (`geometric_mean`) rests on OpenQuake alone — Douglas codes
it unknown — and carries a `component_warning`.

## What the catalogue is, and is not

Douglas (2011) is a **catalogue, not a critical ranking**. It summarises empirical
GMPEs published through 2010 without assessing their quality, and it warns that a
report of its size may contain errors or omissions. Two consequences for this asset:

- The envelopes in each `<model_id>.asset.json` are the catalogue's summaries and are
  frequently approximate — Douglas qualifies many of them, and those carry
  `approximate: true`. They are indicative, not boundaries anyone should defend.
- **These 30 are a heterogeneity benchmark, not a recommendation.** They were chosen
  to span tectonic regimes, distance metrics, site parameterisations, component
  definitions, intensity measures and damping ratios, precisely so the common asset
  interface gets stressed. Several are superseded for modern hazard analysis. Nothing
  here implies any of them is a current best choice.

## Development background

Each of the 30 was developed by different authors, for different regions and
record sets, on different magnitude scales, with different component definitions.
That heterogeneity is the point of the set and is why the registry records
provenance per model rather than per asset. Source populations range from global NGA
databases (models 01–05) through single-country strong-motion networks (Japan K-Net,
Italy, Greece, Iran, Iceland, Taiwan, New Zealand, Mexico) to low-seismicity and
stable-continental compilations (UK, eastern North America) with magnitudes down to
about 1.5.

For any individual model, the development background must be recorded in its
coefficient file when that file is created — see `params/coeffs/README.md`. Asserting
it here from the catalogue summary alone would be the same error as inventing a
coefficient.

## Licence and redistribution — unresolved, check before committing coefficients

Douglas (2011) is a PEER technical report, freely distributed. **The coefficient
tables are a different matter.** Most originate in copyrighted journal articles
(*Earthquake Spectra*, *BSSA*, *Bulletin of Earthquake Engineering*, *Journal of
Seismology* and others), and some are not in the papers at all:

- **Cauzzi & Faccioli (2008)** — coefficients were published as electronic
  supplementary material.
- **Beyer & Bommer (2006)** — Douglas states the functional form and coefficients are
  **not given**. It cannot become a median GMPE from the catalogue, which is why it is
  registered as `kind: "benchmark"` with status `unavailable_by_nature`.

So the redistribution question has to be answered **per model, before its numbers are
committed**, not once for the asset. Where redistribution is not permitted, the
options are to reference an externally licensed implementation (OpenQuake's hazardlib
carries many of these under AGPL-3.0, with its own licence consequences) or to keep
the table outside the repository and load it by path. Each coefficient file must
record the answer in its `provenance.licence` field.

## Limitations the sources state themselves

From the catalogue and the benchmark specification:

- **Long-period coefficients in the NGA-era models rest on far fewer records** than
  short-period ones. The published validity notes must be preserved; not all periods
  are equally constrained even inside one model's stated range.
- **Several models use magnitude scales other than Mw.** Musson et al. (1994) uses ML,
  Lussou et al. (2001) MJMA, Berge-Thierry et al. (2003) Ms, and EPRI (1993a) was
  regressed separately on Mw and on mLg. Converting everything to Mw without
  reproducing the authors' own conversion procedure *changes the model* and must not
  be done silently.
- **Coefficients given only for a subset of analysed periods** — Faccioli et al.
  (2010) is explicit about this. Interpolating across the gap is not permitted.
- **Damping-diverse models publish discrete ratios only.** Chen & Yu (2008a) gives
  {0.5, 2, 7, 10, 20}%, Berge-Thierry {5, 7, 10, 20}%, Cauzzi & Faccioli {5, 10, 20,
  30}%. Substituting 5% for an unlisted ratio is not an approximation, it is a
  different model.
- **Component definitions are not interchangeable.** Geometric mean, larger component,
  GMRotI50, random orientation and vertical models answer different questions. Three
  models (Chen & Yu, Berge-Thierry, Lussou) use *both* horizontals as separate
  observations, which is not a geometric mean either. Only Musson's is still
  `RESOLVE`.
- **Single-site only.** No model in the set supplies spatial correlation. See
  `README.md`.

## Still needed

**For an anchor branch (22 models) — implementation is unblocked.** What is still
needed is not source material but work: a coefficient file carrying the appendix's
values with provenance, a `MedianKernel`, and regression tests. Two caveats:

- The **component definition** is now recorded for all 22, mostly from Douglas's
  summary tables rather than the papers. Arroyo et al. (2010) rests on OpenQuake
  alone; treat that one as provisional when comparing models.
- The appendix's values are transcribed from Douglas, not from the papers, and
  Douglas warns it may contain errors. Each coefficient file should record
  `original_paper_checked: false` until someone does.

**For a spectral branch (all 30).** The original publication or supplement. For 18
models the period grid is known (from OpenQuake, matching Douglas's counts) but the
coefficients are not. For the other 12 even the grid needs the paper.

**For the three `partial` models.** Youngs needs the event-type term; Hong needs
intraslab `c4` *and* its distance metric named; Musson model 2 needs Dahle et al.
(1990b) — a second publication, not just the original paper.

**In every case.** A licence answer for redistributing the table here, and published
example values or independently calculated anchor cases to test against. Without the
latter the implementation is unverified however carefully the equation was
transcribed — the appendix supplies no worked examples.

**For multi-version families.** Takahashi (2005) / Zhao (2006) / Fukushima (2006)
needs a decision on *which* version, recorded in the coefficient file. The appendix's
coefficient naming points at Zhao et al. (2006) but does not say so, and the registry
records that as probable rather than settled.

## Also needed, and separate from all of the above

A **spatial correlation model**, as its own asset, before `LocalIntensity` can be
drawn jointly across sites. This asset cannot supply it, and `assets/hazard/shakecast`
is the other place the question lands — a ShakeMap product consumed as a grid carries
correlation implicitly, which is a different design from a GMPE plus a correlation
model and should not be mixed with it by accident.
