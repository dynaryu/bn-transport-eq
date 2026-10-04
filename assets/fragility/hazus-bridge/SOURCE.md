# Source

- Supplied reference: `hazus61_roadway_bridge_fragility_prompt.md`, from the user's
  Downloads folder; retained in `docs/sources/` on 2026-10-04 with a SHA-256 in
  every parameter record.
- Cited primary source: FEMA, *Hazus Earthquake Model Technical Manual*, Hazus 6.1,
  July 2024, Section 7.1. [Official PDF](https://www.fema.gov/sites/default/files/documents/fema_hazus-earthquake-model-technical-manual-6-1.pdf).
- Numerical sources: Tables 7-1, 7-2, 7-6 and 7-7; shaking Equations 7-1 through 7-4.
- Verification level: values reproduced from the supplied document. The primary
  PDF URL was located but returned HTTP 403; no independent FEMA-software
  reproduction is claimed. Unverified page numbers remain null.
- Development scope: Hazus earthquake loss-estimation classes; the bridge matcher
  is based on US NBI structure classes and California/non-California vintage.
- Licence: source attribution retained; no new licence or redistribution rights
  are asserted for the user-supplied compilation or incorporated references.
- Parameters are published-model transcription, not synthetic calibration. The
  example asset geometries and hazard scenarios are illustrative.

The specification records conflicting vintage statements, maximum-span versus
total-length wording, a PGD damage-state inconsistency, K3D single-span
singularities, and zero-skew PGD sine singularities. Named policy choices retain
these distinctions; invalid literal results are rejected. Classification precedence
and the severe-only PGD convention are explicit implementation policies. No
corrosion or inspection-health adjustment is inferred.
