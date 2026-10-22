# 2. Instances: what does exist

Everything here fills the [schema](../schema/README.md) and is checked against
it by `python -m src check`. Each file holds parameter sets for one function, θ plus
provenance, and sits in the folder named by that function's `role`:

| Path | Function | Holds |
|---|---|---|
| `hazard/campbell-2003.json` | `hazard.gmm` | Campbell (2003), SA(1.0), central and eastern North America |
| `hazard/demo-gmm.json` | `hazard.gmm` | Campbell's numbers over a California box, for the `toy-ca` demo only |
| `hazard/spatial-correlation.json` | `hazard.spatial-correlation` | Jayaram & Baker (2009) at PGA, 0.3 s, 1 s and 3 s, and independent residuals |
| `fragility/hazus61-bridge-sa1.json` | `fragility.bridge` | the 28 Hazus 6.1 bridge classes (base Table 7-6 curves); `hazus61-source.md` is the transcription those numbers were taken from |
| `functionality/usable-from-damage.json` | `functionality.usable-from-damage` | closure policies: which damage states leave a bridge usable |

A fragility curve is one set: its `form`, the `imt` it is defined on, the
`damage_scale` it predicts and that form's values. Another source's curves are
another file in `fragility/`, with no change to the schema.

The networks these are applied to are in [`../examples/`](../examples/README.md).
