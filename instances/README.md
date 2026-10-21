# 2. Instances: what does exist

Everything here fills the [schema](../schema/README.md) and is checked against
it by `python -m src check`.

| Path | Holds |
|---|---|
| `parameters/*.json` | parameter sets, one file per function: θ plus provenance. `hazus61-bridge-sa1.json` holds the 28 Hazus 6.1 bridge classes (base Table 7-6 curves); `hazus61-source.md` is the transcription those numbers were taken from |
| `networks/<n>/network.json` | one network's rows, one table per entity type, with every assumed value named under `assumptions`. Its `SystemEvent` table defines what the analysis asks: per origin-destination pair, which analysis, and what counts as surviving |
| `networks/<n>/<name>.assembly.json` | an assembly: which function, with which parameter set or inline parameters, describes which rows. Compiling one gives a BN; its id is `<n>/<name>` |

| Assembly | BN |
|---|---|
| `ema-highway/gmm` | Eastern Massachusetts highways, 129 bridges, a fixed M8 scenario through Campbell (2003), 288 system events; 1093 nodes |
| `hebei-transport/shakemap` | Hebei directed network, 33 bridges, under the 1966 Xingtai earthquake's USGS ShakeMap, with spatially correlated residuals, 2 system events; 169 nodes |

The two networks came from the `network-datasets` repository through the earlier
pilot's packages (commit `784e6cc`). These files are now the source.
