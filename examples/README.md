# Examples: networks and their assemblies

A report concerns one network, so the networks here are examples of the input the
tools take, checked against the [schema](../schema/README.md) like everything else.

| Path | Holds |
|---|---|
| `<n>/network.json` | one network's rows, one table per entity type, with every assumed value named under `assumptions`. Each edge carries `length_km` and `avg_speed_kph`; `key_destination` marks the nodes system functions use as destinations |
| `<n>/<name>.assembly.json` | an assembly: which function, with which parameter set or inline parameters, describes which rows. Compiling one gives a BN; its id is `<n>/<name>` |

| Assembly | BN |
|---|---|
| `ema-highway/gmm` | Eastern Massachusetts highways, 129 bridges, a fixed M8 scenario through Campbell (2003), whether each of 72 nodes reaches the nearest airport in time; 588 nodes |
| `hebei-transport/shakemap` | Hebei directed network, 33 bridges, under the 1966 Xingtai earthquake's USGS ShakeMap, with spatially correlated residuals, whether each of 21 nodes reaches Handan in time; 186 nodes |
| `toy-ca/demo` | an 8-node toy network in central California under a hypothetical M7.8 on the San Andreas, with an approximate demo GMM; 46 nodes. `toy-ca-demo.qmd` walks through it; `answer-sheet.md` lists the questions we expect and the probability queries they translate to |

The EMA and Hebei networks came from the `network-datasets` repository through the
earlier pilot's packages (commit `784e6cc`). These files are now the source.

The tbnpy builder stores generated models beneath the corresponding
example, in `tbnpy/<assembly-name>/` (for example, `toy-ca/tbnpy/demo/`). Each
generated model defines tbnpy variables and probability objects using the
shared PyTorch implementations in `src/builder/kernels/`. RSR reference tensors
and their metadata live alongside that model in `rsr/<origin>/target_add_h=<h>/`. Separate assembly
directories keep different model selections for the same network from overwriting
one another. Generate toy-ca with `python -m src build toy-ca/demo`, then run
`python examples/toy-ca/tbnpy/demo/model.py --samples 2000`. Queries (`python -m
src query`, the page, the chat) build the same tbnpy model in memory and reuse the
stored rules of their threshold. A threshold without its own rules is extracted in
the background on its first run; meanwhile rules that still hold from other
thresholds, and ndtools, give the same answer.
