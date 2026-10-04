# Prompt A instance — Hebei directed transport network

Paste the two sections below into the `description` and
`known parameter values` slots of `tbnpy/prompts/02_prompt_nl_to_spec.md`,
together with `00_context_tbnpy_api.md` and `01_model_spec_schema.md` in full.

---

## System description

I want to assess earthquake risk to a transport network in Hebei, specifically
**the connectivity from the network's source nodes to `n22` (Handan)** after an
earthquake.

**The question the model must answer.** Given one earthquake scenario: the
probability that each origin still reaches Handan. Forward only — scenario in,
distribution out. Diagnostic questions (given an observed outage, which arc
matters most) are out of scope for now and the model should refuse them rather
than approximate them.

**The network.** 22 nodes and 33 **directed** arcs, from the
`hebei_transport_liu2021` v1 dataset. Direction matters: reachability must be
evaluated along arc directions, not treated as an undirected graph. Nodes carry
city names (Chengde, Handan, and so on), which I want preserved so a question can
name a city rather than a node id. **Each arc carries exactly one highway
bridge.**

**Coordinates are digitised from a figure and are unitless.** They are not a
projection and carry no scale, so there is no meaningful epicentral distance to
compute. That rules out a distance-based ground-motion relation for this case.
Instead the scenario is specified directly as **one spectral acceleration at 1 s
applied uniformly to every site**. The analyst sets that intensity; it is an
input, not an uncertain quantity.

**What causes what.**

1. The scenario fixes the ground motion at every site to the same value. Because
   the intensity is uniform and set by the analyst, the ground-motion variable at
   each site is degenerate — a point mass. It should still exist as a variable
   rather than being folded away (see the dispersion note below).
2. Whether an arc's bridge is usable depends on the ground motion at its site,
   through the Hazus 6.1 lognormal fragility curve for its class, at the
   **Extensive** limit state.
3. Whether an origin still has service depends on which arcs are usable:
   directed reachability to Handan.

**Deterministic parts.** The closure policy (Extensive or worse ⇒ unusable) is a
threshold. Directed reachability is settled once arc states are fixed. Both are
degenerate kernels.

**Replication.** Ground-motion and usability variables repeat over the 33 arcs.
Service repeats over the origins. As with the other case, I want one system node
whose children are the per-origin service variables, rather than one node per
origin each re-running the same search.

**Dispersion.** Keep the ground-motion and fragility dispersions in separate
nodes even though the ground motion is degenerate here. The point is structural
consistency with the other case in the same pilot: the same code should assemble
both, and the uniform-intensity case is the degenerate special case of the
distance-based one, not a different model.

**No population data.** This dataset carries no population figures, so the
population-weighted accessibility metric that the other case reports cannot be
computed. The model should omit that node rather than invent uniform weights.

**What will be observed, and what I want to infer.** Nothing is observed yet;
everything is forward simulation.

## Known parameter values

| quantity | value | source |
|---|---|---|
| Nodes, arcs | 22, 33 **directed** | `hebei_transport_liu2021` v1 |
| Node coordinates | digitised from a figure, **unitless** | same |
| Node names | city names (`CD`/Chengde, Handan, …) | same |
| Destination | `n22` (Handan) | the decision question |
| Hazard representation | uniform SA(1 s), analyst-set, in g | pilot decision, forced by unitless coordinates |
| Default intensity | 0.7 g | pilot default |
| Bridge class per arc | **HWB11 assumed for every arc** | pilot assumption — see open questions |
| Fragility | Hazus 6.1 lognormal, Extensive limit state | Hazus 6.1 specification |
| Closure policy | Extensive or Complete ⇒ unusable | my assumption, needs engineering review |
| Population weights | **none in this dataset** | — |
| Ground-motion relation | **not applicable** — no usable distances | — |

### Open questions I already know about — please carry these into `open_questions`

1. **The arcs are not really binary.** The source data gives each arc
   `n_states: 3` with `capacities: [0, 1, 2]`. This model collapses that to
   usable/unusable, which discards the partial-capacity state. That is a
   modelling decision, not a property of the data, and it should be stated as
   one. A multi-state version is the natural next step, and the system function
   would need to change with it.
2. **Every arc is assumed to be one HWB11 bridge.** The dataset does not give
   bridge classes. This is the strongest unverified assumption in the case and
   it drives every number the model produces.
3. **No ground-motion model.** With unitless coordinates there is no distance,
   so the uniform intensity is a stand-in for a hazard model rather than a
   result of one. Scenario comparisons are therefore relative, not absolute.
4. **Uniform intensity means perfectly correlated ground motion**, which is the
   opposite extreme from the independent case. Neither is right; the truth is in
   between, and the structure should leave room for a residual.
5. **No population data**, so the decision metric is connectivity only. If a
   population-weighted answer is wanted, that data has to come from somewhere.
