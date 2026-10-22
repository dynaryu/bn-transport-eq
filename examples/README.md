# Examples: networks and their assemblies

A report concerns one network, so the networks here are examples of the input the
tools take, checked against the [schema](../schema/README.md) like everything else.

| Path | Holds |
|---|---|
| `<n>/network.json` | one network's rows, one table per entity type, with every assumed value named under `assumptions`. Each edge carries `length_km` and `avg_speed_kph`; `key_destination` marks the nodes system functions use as destinations |
| `<n>/<name>.assembly.json` | an assembly: which function, with which parameter set or inline parameters, describes which rows. Compiling one gives a BN; its id is `<n>/<name>` |

| Assembly | BN |
|---|---|
| `ema-highway/gmm` | Eastern Massachusetts highways, 129 bridges, a fixed M8 scenario through Campbell (2003), travel time to the nearest airport from 72 nodes; 661 nodes |
| `hebei-transport/shakemap` | Hebei directed network, 33 bridges, under the 1966 Xingtai earthquake's USGS ShakeMap, with spatially correlated residuals, travel time to Handan from 21 nodes; 207 nodes |
| `toy-ca/demo` | an 8-node toy network in central California under a hypothetical M7.8 on the San Andreas, with an approximate demo GMM; 53 nodes. `toy-ca-demo.qmd` walks through it |

The EMA and Hebei networks came from the `network-datasets` repository through the
earlier pilot's packages (commit `784e6cc`). These files are now the source.
