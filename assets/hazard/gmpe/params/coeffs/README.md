# Coefficient files

One file per model family and version. **This folder is still empty of data**, but no
longer for want of numbers: the benchmark specification's coefficient appendix
supplies a reproducible anchor branch (normally PGA) for 22 of the 30. Writing those
files and their kernels is outstanding work, not a blocked dependency.

Nothing here is to be reconstructed from figures, from a secondary summary, or from
memory. See `../../SOURCE.md` and `reproduction_status()`.

## The completeness gate

Before implementing any branch, check `reproduction.status` under `parameters` in the
model's asset file, `../../<model_id>.asset.json`:

| Status | What you may do |
|---|---|
| `ANCHOR_BRANCH_REPRODUCIBLE` | Implement that exact branch. Refuse every other period. |
| `PARTIAL` | Stop. List the missing coefficient; do not implement the branch. |
| `ORIGINAL_TABLE_REQUIRED` | Metadata, validation and loader only. Return no numbers. |
| `NOT_A_STANDALONE_GMPE` | Only the diagnostic quantity the source supports. |

**An anchor branch does not license the spectral periods.** Generating a period grid
by interpolating anchor coefficients is the specific error this gate exists to
prevent. Never treat coefficients from a later revision, another region, another
damping ratio, another component definition or a software implementation as if they
were the published values, unless that source is added explicitly and its provenance
recorded.

## Adding a model

1. Obtain the coefficient table. For an anchor branch the appendix is sufficient and
   its values should be copied with `source: "spec appendix §11.3"` and
   `original_paper_checked: false`. For a spectral branch you need the **original
   publication** or its electronic supplement.
   Resolve every field the registry still marks `RESOLVE` for that model — component
   definition, magnitude scale, exact period grid, and units where the block omits
   them. These are not guessable: a wrong log base is a factor-of-`ln 10` error that
   looks entirely plausible in the output.
2. Answer the redistribution question for that table. If the answer is no, keep the
   file outside the repository and load it by path.
3. Write `<model_id>.json` in the format below, filling the `provenance` block.
4. Register a `MedianKernel` subclass in `model.py` implementing the published
   equation, and add it to `_KERNELS` under the same `model_id`.
5. In the model's `<model_id>.asset.json`, set `coefficients.status` to `"present"`
   and list under `hasCapability` what the model can now do. Until this step the file
   declares none, and the conformance check holds it to that.
6. Add regression tests against published example values or independently calculated
   anchor cases. Transcription errors survive careful reading; they do not survive a
   known-value test.

Step 6 is not optional. `test_no_published_model_can_predict_yet` and
`test_each_family_is_its_own_asset_file_and_claims_nothing_it_cannot_do` will fail as
soon as step 5 lands — narrow them to the families still missing rather than
deleting them.

## Format

```json
{
  "model_id": "<must match the asset file's name, without .asset.json>",
  "version": "<which publication, where a family has several>",
  "log_base": "ln | log10",
  "units": "g | cm/s2 | m/s2 | cm/s | cm",
  "component": "<the published component definition, exactly>",
  "magnitude_scale": "Mw | ML | ...",
  "damping": [0.05],
  "periods_s": ["PGA", 0.01, 0.02, "..."],
  "coefficients": {
    "PGA": {"...": "one entry per published coefficient, named as the paper names it"},
    "0.01": {"...": "..."}
  },
  "sigma": {
    "note": "tau/phi/sigma as published, in the SAME log base as the medians",
    "PGA": {"tau": null, "phi": null, "sigma": null}
  },
  "provenance": {
    "citation": "full reference, with DOI",
    "catalogue": "Douglas (2011), PEER 2011/102",
    "catalogue_section": "2.xxx / 4.xxx",
    "source": "spec appendix §11.3 | original paper | electronic supplement",
    "coefficient_scope": "PGA | PGA0 | spectral | component-conversion",
    "completeness": "complete_for_branch | partial | missing",
    "original_paper_checked": false,
    "external_implementation_checked": false,
    "table": "which table or supplement, by number",
    "retrieved": "YYYY-MM-DD",
    "licence": "whether these numbers may be redistributed here, and on what basis",
    "transcribed_by": "who, and checked against what",
    "verified_against": "published example values, or the anchor cases used",
    "notes": "Do not infer missing values."
  }
}
```

Conventions that the wrapper relies on:

- **Coefficient names follow the paper**, not a house style. The file is reviewed
  against the publication, and renaming defeats that.
- **`log_base` and `units` describe the file**, not the output. The wrapper converts
  once, at the API boundary, and records both in `provenance`.
- **`sigma` entries are in the file's own log base**, like the medians. Mixing a
  log10 median with an `ln` sigma is silent and severe.
- **Periods are the published grid.** Omit a period rather than interpolating into it.
  `null` for a coefficient the paper does not give is better than a filled-in value.
- **One file per version.** Do not merge coefficient sets from different publications
  of the same family into one file.
