# Prompt A instance — Eastern Massachusetts highway network

Paste the two sections below into the `description` and
`known parameter values` slots of `tbnpy/prompts/02_prompt_nl_to_spec.md`,
together with `00_context_tbnpy_api.md` and `01_model_spec_schema.md` in full.

---

## System description

I want to assess earthquake risk to the Eastern Massachusetts highway network,
specifically **the connectivity from every other node to the two airport nodes,
`n22` and `n66`** — reaching *either* counts as having access.

**The question the model must answer.** Given one earthquake scenario: for each
origin node, the probability that it still reaches at least one airport node;
and across all origins, the expected share of the population that still has
access. These are forward questions — scenario in, distribution out. I do not
want diagnostic questions answered yet (given an observed outage, which bridge
is most critical), and the model should make that boundary explicit rather than
answering them badly.

**The network.** 74 nodes and 129 undirected links, from the `ema_highway` v1
dataset in my `network-datasets` repository. Node coordinates are planar, in
kilometres, and each node carries a population figure. **Each link carries
exactly one highway bridge**, and the link's `fragility_type` field gives that
bridge's Hazus class (HWB1–HWB28; most are HWB11). The link is the component —
there is no separate bridge inventory.

**What causes what.**

1. The earthquake scenario is something the analyst sets: a moment magnitude and
   an epicentre in the same planar coordinates. It is an input, not an uncertain
   quantity, so it should parameterise the model rather than be a variable in it.
2. Ground motion at each bridge's site depends on magnitude and on the distance
   from the epicentre to that bridge's midpoint. Use the Campbell (2003) T = 1 s
   relation exactly as the dataset's own script implements it — I need to be able
   to reproduce that script's numbers. Spectral acceleration at 1 s, in g.
3. Whether a bridge is usable depends on the ground motion at its own site,
   through the Hazus 6.1 lognormal fragility curve for its class, evaluated at
   the **Extensive** limit state.
4. Whether an origin still has service depends on which bridges are usable: it
   is a graph reachability question over the surviving links, undirected.
5. The population share with access is a population-weighted average over the
   origins' service states.

**Deterministic parts.** The closure policy — None, Slight and Moderate damage
leave a bridge usable; Extensive and Complete make it unusable — is a threshold,
not a random variable. Connectivity is settled once bridge states are fixed. The
population share is a weighted sum. All three are degenerate kernels and should
not be represented as near-degenerate tables.

**Replication.** The ground-motion and usability variables repeat over the 129
links; the service variable repeats over the 72 origins in scope. The system
function is one evaluation that answers every origin at once, so I would rather
have a single node with many children than 72 nodes each re-running the same
graph search over the same bridge states.

**Dispersion — please keep the two sources separate.** The ground-motion
relation has its own dispersion, and the fragility curve has its own. It is
algebraically equivalent to combine them into a single sigma and sample
usability directly, and an earlier version of this model did exactly that. I do
not want that here: I want ground motion to be its own variable, because a
correlated spatial residual has to attach *somewhere*, and with the dispersions
merged there is nowhere to put it. Expect the marginals to be unchanged; the
point is where the structure leaves room.

**What will be observed, and what I want to infer.** Nothing is observed yet.
Everything is forward simulation from a scenario. Evidence on bridge states, and
the diagnostic questions that follow from it, come later — but the model should
be built so that adding them is attaching evidence to existing variables, not
restructuring.

## Known parameter values

| quantity | value | source |
|---|---|---|
| Nodes, links | 74, 129 undirected | `ema_highway` v1 `nodes.json`, `edges.json` |
| Node coordinates | planar, kilometres | same |
| Node population | per-node figure | same |
| Bridge class per link | `fragility_type`, HWB1–HWB28 | same |
| Destinations | `n22`, `n66` — either counts | the decision question |
| Ground-motion relation | Campbell (2003), T = 1 s | the dataset's own script; reproduce it exactly |
| Distance measure | epicentre to link midpoint, planar | the dataset's script |
| Ground-motion aleatory sigma | `1.110 − 0.0793·M` for M < 7.16, else `0.543` | the dataset's script |
| Ground-motion epistemic sigma | 0.3 | **placeholder in the dataset's script, not a calibrated value** |
| Fragility | Hazus 6.1 lognormal, Extensive limit state, by HWB class | Hazus 6.1 specification |
| Closure policy | Extensive or Complete ⇒ unusable | my assumption, needs engineering review |
| Magnitude range to support | 4.0 to 9.5 | pilot scope |
| Spatial correlation of ground motion | **unknown — none available** | see open questions |

### Open questions I already know about — please carry these into `open_questions`

1. **No spatial correlation model.** Ground motion at different sites is
   independent here, so bridge failures are independent given the scenario. For
   a network question this is the wrong tail: it understates the chance that
   many bridges fail together, and the joint tail is what a connectivity answer
   depends on. I know of no correlation model attached to this dataset. The
   structure should leave the residual attachable without a rewrite.
2. **The 0.3 epistemic sigma is a placeholder** carried over from the dataset's
   script. It is not calibrated and should not be reported as if it were.
3. **The closure policy needs engineering review.** "Extensive or worse is
   unusable" is a modelling convenience; real closure depends on the damage mode
   and on what traffic the link carries.
4. **Planar coordinates in kilometres** are the dataset's own schematic, not a
   projection of real geography. Distances are therefore approximate.
5. **Bridge classes are mostly HWB11.** Whether that reflects the real inventory
   or a dataset default, I do not know.
