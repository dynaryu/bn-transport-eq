# Source

- Supplied reference: `hazus61_roadway_bridge_fragility_prompt.md`, from the user's
  Downloads folder; retained in `docs/sources/` on 2026-10-04 with a SHA-256 in
  every parameter record.
- Cited primary source: FEMA, *Hazus Earthquake Model Technical Manual*, Hazus 6.1,
  July 2024, Section 7.1. [Official PDF](https://www.fema.gov/sites/default/files/documents/fema_hazus-earthquake-model-technical-manual-6-1.pdf).
- Numerical source: Table 7-5; roadway descriptions and damage semantics in the supplied sections 4–6.
- Verification level: values reproduced from the supplied document. The primary
  PDF URL was located but returned HTTP 403; no independent FEMA-software
  reproduction is claimed. Unverified page numbers remain null.
- Development scope: Hazus earthquake loss-estimation classes; the bridge matcher
  is based on US NBI structure classes and California/non-California vintage.
- Licence: source attribution retained; no new licence or redistribution rights
  are asserted for the user-supplied compilation or incorporated references.
- Parameters are published-model transcription, not synthetic calibration. The
  example asset geometries and hazard scenarios are illustrative.

Preserve the combined extensive/complete state. These PGD damage curves do not
supply a deterioration process, ground-shaking response, or functionality mapping.
