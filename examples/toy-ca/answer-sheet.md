# Answer sheet: questions in words → probability queries

How quakenet's language layer should translate the questions we expect into
probability queries on the BN, which the engine then computes. Toy-ca names are
used throughout (bridges b1–b10, nodes n1–n8, key destinations n1 and n2). The
sheet is also the start of a test set for the translation: each entry can gain
phrasings and the expected query JSON.

Answers use a defined probability, expectation or explicitly interpreted comparison.
State the event or random quantity, its conditions and any units. Report probability
intervals where available. A probability difference is a change in probability,
not the probability of an event. Do not invent composite priority scores.

## Notation

- **B_i** = 1 means bridge i is open to traffic; B_i = 0 means it is closed. A bridge has no
  node of its own: its damage closes the edge it carries, so B_i is that edge's state
  (`EdgeOpen::e_i` for bridge `b_i`, which carries edge `e_i` in this demo).
- **F** means the selected system service requirement fails (`SystemState::<origin> = 0`).
- **H** is the selected hazard scenario (the Event row and the hazard model).
- **E** is the available emergency evidence, such as inspection results.

"System failure" must be defined consistently. In the current demo, each origin
succeeds if it can reach either N1 or N2 within 0.5 hours of its intact travel
time (the SystemEvent row: `target_add_h = 0.5`). Failure probabilities can be
reported separately for each origin.

## Preparation mode

### 1. What would be the effect of retrofitting the old bridges?

    Baseline: P(F | H, current bridge models)
    After retrofit: P(F | H, bridge i's fragility model changed to HWB11)

Show the baseline and post-retrofit failure probabilities as separate results,
with the affected bridge(s) named. Evaluate b2, b4 and b6 individually, then all
three together. HWB11 is the assumed retrofit model for this demo. If the user
requests a reduction, define it as baseline minus post-retrofit probability and
label it in percentage points.

### 2. Which bridge plays the most important role in maintaining connectivity?

"Most important" has several mathematical meanings. Suggest the relevant
definitions and ask the user to choose before calculating:

| Meaning | Quantity |
|---|---|
| Failure under a forced bridge outage | P(F given do(B_i = 0), H) |
| Dependence on the bridge's availability | P(F given do(B_i = 0), H) and P(F given do(B_i = 1), H), reported separately |
| Diagnostic association with an observed system failure | P(B_i = 0 given F, H) |

After the user chooses, read back and compute that definition. The forced-outage
and forced-open cases are interventions: other bridges retain their hazard-driven
uncertainty. The diagnostic case conditions on observed failure and is not a
causal importance measure. If explicitly requested, the closed-minus-open
probability difference is a defined sensitivity comparison; label it as such,
not as an event probability. Do not silently choose it as the default score.

If the question concerns topology alone, close each bridge individually while
keeping all other bridges open, and compare the resulting loss of connectivity
or increase in travel time.

### 3. What happens to the system under a 1,000-year hazard?

    P(system state | H_1000)

H_1000 must represent a defined hazard scenario or spatial ground-motion model
associated with a 1,000-year return period. A return period alone does not
specify an earthquake's magnitude, location or shaking field. The appropriate
hazard input must be established before calculating system probabilities.

## Emergency mode

### 1. Inspection of which bridge should be prioritised?

Inspection informs a decision; it does not restore a bridge. Use expected
value of sample information (EVSI), in the same units as the supplied losses:

    EVSI_i = min_a E[L(a, X) | H, E]
             - sum_y P(Y_i=y | H, E) min_a E[L(a, X) | H, E, Y_i=y]
    Net value_i = EVSI_i - inspection cost_i

Specify the available actions, their losses for every modelled state, loss
units, inspection accuracy and cost. Optimise the action after each possible
inspection result. Rank by net value; choose no inspection when every net value
is non-positive. Do not substitute an uncertainty-times-sensitivity score.

The pilot evaluates one inspection at a time over up to eight binary state
nodes, using an explicit loss table. That table must encode action consequences;
the app does not infer repair consequences or invent losses. Perfect inspection
uses the identity accuracy matrix. Sequential policies remain future work.

If the intended question is instead "Which bridge should we restore first?":

    Current scenario: P(F | H, E)
    Forced-open scenario: P(F | do(B_i = 1), H, E)

Report the two system-failure probabilities separately for each candidate
bridge. An explicitly requested reduction can be reported as a probability
difference in percentage points. For an actual
repair after inspection, observations of pre-repair state and interventions on
post-repair state need distinct variables; the current static model does not
provide that temporal repair model.

### 2. We need to reach N1. Which path should we use?

Specify the starting node, then for each candidate path:

    Path survival probability_p = P(all edges on path p are open | H, E)

Recommend the path with the highest survival probability, and show its travel
time alongside that probability. Calculate this probability jointly from the
samples, preserving dependence between bridge conditions.

A survival rule can be used directly only if it identifies or guarantees a valid
path to N1 from the chosen starting node. The current rules for reaching either
N1 or N2 do not automatically answer this destination-specific question.

## Executable cases and current scope

The JSON files in `answer-cases/` specify context, alternative phrasings, expected
tool arguments and forbidden interpretations. `few_shot` cases enter the prompt;
`held_out` cases are reserved for evaluation. See `answer-cases/README.md` for
adding cases. The examples are expectations, not stored numerical answers.

| Question | Implementation / boundary |
|---|---|
| Retrofit scenarios | separate event queries for baseline and post-retrofit failure probabilities |
| Bridge importance | clarify the intended definition, then compute its event queries; an explicitly requested closed-minus-open comparison can use `analysis.kind=importance` |
| 1,000-year hazard | clarification, then a sourced proposal; no magnitude guessed |
| Inspection | `analysis.kind=voi`; supplied finite losses, accuracy and cost |
| Restoration scenarios | separate current and forced-open failure probabilities; no temporal repair model |
| Path to N1 | `analysis.kind=paths`; joint probabilities of valid candidate routes to N1 |

VOI is an expected improvement in the best decision, with loss units; it is a
valid expectation-based answer. It is not an arbitrary score and need not be
replaced by a bridge-failure probability. Keep its loss table and inspection
assumptions explicit.

Discrete observations use rejection conditioning. Results include drawn and
accepted sample counts; fewer than 30 matches are refused. Continuous evidence
and interventions on jointly owned nodes remain unsupported. Candidate path
search is bounded; the chosen route is best among those evaluated. Comparisons
and VOI currently report point estimates without derived confidence intervals.

## How answers are kept

Every query the page, the command line or the language layer computes is
appended to `records/queries.jsonl`, with:
- **the question and its answer:** the question in words and the model that
  translated it, when it came through Ask;
- **the computation:** the query, its statement, each probability with its 95%
  interval, the samples, the seed, the time taken and the rsr rules used;
- **what produced it:** the parameter sets, the model fingerprint, the code
  commit and the library versions.

Small scenarios can share a whole-network sample bank under `records/samples/`,
keyed by the model, interventions, sample count and seed. Otherwise the query's
scope is sampled on demand. Evidence filters matching rows from the joint draws;
it does not overwrite node values. `python -m src scope <assembly> --event ...`
shows scope; `python -m src replay <id>` checks a recorded run. Reusing a seed
alone does not guarantee paired draws across changed models.

## Proposal dialogues

See `proposals/README.md` at the repository root for examples covering missing
hazards, closure policies, destination-specific system definitions and VOI inputs.
Drafts stay outside the active loader until their source, data and validation
report have been reviewed and a named user activates the displayed version.
