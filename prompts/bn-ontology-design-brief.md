# Design brief — a minimal, extensible ontology that compiles to Bayesian networks

**What this is.** A reusable prompt. Fill in Part 0 for a new application domain, delete nothing else, and
paste the whole file into a capable LLM. Parts 1–9 are domain-independent and are what make the result
reusable: the four-layer architecture and the compilability contract stay fixed while the domain changes.

**Version** 1.0 · **Designed around** disaster risk assessment of infrastructure networks
**Run successfully on** *(nothing yet — add each successful domain)*

---

## Part 0 — Fill these in

Replace every `«...»`. If a field does not apply, write `none` rather than deleting it — the model needs
to know it was considered. The groups below follow the four layers of Part 2, so each input's purpose is
visible from where it sits: **A** names the ontology, **B** bounds it, **C–F** supply one layer each, and
**G–H** supply reuse targets and test cases.

```yaml
# ═══ A. Identity ══════════════════════════════════════════════════════════════
DOMAIN:            «e.g. disaster risk assessment of infrastructure networks»
NAMESPACE:         «e.g. https://example.org/dra#  →  prefix dra:»
ONTOLOGY_NAME:     «short human name + version, e.g. "Lean DRA ontology v0.2"»


# ═══ B. Scope — what the ontology must and must not support ═══════════════════
QUERY_PATTERNS:                # classes of query to support. These set scope and justify every term.
                               # Specific instance-level questions belong in the BN-design prompt.
  - «forward: hazard scenario → impact distribution»
  - «backward: observed evidence → likely asset states»
  - «comparative: rank interventions by effect on a metric»
  - «...»

SCOPE_LIMITS:                  # deliberately NOT modelled yet
  - «no explicit time dynamics beyond state validFrom/validTo»
  - «...»


# ═══ C. Layer 1 — nodes and factors ═══════════════════════════════════════════
VARIABLE_BEARING_ENTITIES:     # properties REPRESENTED AS NODES. Mark each STOCHASTIC or DETERMINISTIC.
  - «HazardEvent»              # ROOT. e.g. earthquake scenario: magnitude, location, return period
  - «LocalIntensity»           # STOCHASTIC fn of HazardEvent. Intensity at ONE asset's site, e.g. PGA.
                               #   Spatially correlated across sites — see the ShakeMap note in D.
  - «AssetHealth»              # ROOT. condition / deterioration of one asset. Noisily observed — see E.
  - «AssetDamageState»         # STOCHASTIC fn of (LocalIntensity, AssetHealth) via the fragility curve
  - «AssetFunctionality»       # DETERMINISTIC fn of AssetDamageState, e.g. 100 / 50 / 0 %
  - «Service»                  # STOCHASTIC. system-side function level; what the simulator outputs
  - «Impact»                   # receptor-side decision basis, e.g. affected population.
                               #   Always present: states what the decision is measured in. See F.

STRUCTURAL_ENTITIES:           # topology, scope, or factor parameters. NOT nodes.
  - «Asset»                    # located component; each Asset* variable describes one
  - «Network»                  # topological connection of assets; supplies the simulator's parent set
  - «Receptor»                 # who bears the loss; what Impact is counted over
  - «AssetFragility»           # parametric form of the AssetDamageState factor, NOT a node.
                               #   Promote to a node only to learn μ/σ from observed damage.


# ═══ D. Layer 1 — one entry per factor ════════════════════════════════════════
FACTORS:                       # each supplies P(target | given). Name both sides.
  - name: «ShakeMap»;             target: «LocalIntensity»;     given: «HazardEvent»
    note: «must carry spatial correlation across asset sites, not per-site marginals alone»
  - name: «fragility curves»;     target: «AssetDamageState»;   given: «LocalIntensity, AssetHealth»
    note: «parametric — lognormal μ/σ indexed by asset attributes»
  - name: «DS→capacity lookup»;   target: «AssetFunctionality»; given: «AssetDamageState»
    note: «DETERMINISTIC; one table per asset type, onto the shared capacity scale»
  - name: «network flow sim»;     target: «Service»;            given: «AssetFunctionality in scope»
    note: «parent set derived from Network topology, not hand-enumerated»
  - name: «exposure model»;       target: «Impact»;             given: «Service, Receptor»
    note: «classify it under F»
  - name: «...»

CAPABILITIES:                  # computational and numerical character of each factor's asset.
                               # This is what makes inference-operator choice derivable rather
                               # than hand-assigned: an operator is admissible only if the
                               # asset offers the operations it needs.
  per_asset:
    - name: «ShakeMap / GMPE»
      evaluate_density:  «no — black box»
      sample_forward:    «yes»
      evaluate_likelihood: «no»
      deterministic:     «no»
      support:           «continuous»
      joint_or_marginal: «JOINT across sites — carries spatial correlation»
      batchable:         «yes»
      cost_per_eval:     ««order of magnitude»»
      reproducible:      «yes, given seed»
    - name: «...»
  # Record for every asset:
  #   evaluate_density / sample_forward / evaluate_likelihood / batch_classify / gradient
  #   deterministic?            — a degenerate kernel, not a near-degenerate table
  #   support                   — discrete, continuous, or mixed; plus units
  #   joint_or_marginal         — whether it returns joint samples or per-site marginals only.
  #                               Marginal-only silently forces an independence assumption.
  #   normalising_constant      — can it return one? Needed where evidence probabilities are.
  #   batchable, cost_per_eval, reproducible
  #   validity_scope            — see Part 5
  # Declare capabilities the asset ACTUALLY has, not those its documentation implies. A wrapper
  # that claims an operation it cannot perform is worse than one that declares the gap.


# ═══ E. Layer 2 — evidence ════════════════════════════════════════════════════
DATA_SOURCES:                  # what can be observed, and how noisily
  - name: «bridge inspection records»; observes: «AssetHealth»; noisy: «yes — rater variability»
    note: «noisy ⇒ needs a measurement model, not hard evidence»
  - name: «...»


# ═══ F. Layer 3 — decision ════════════════════════════════════════════════════
DECISION_METRICS:              # functionals of the Impact distribution. Mark expectation-linearity.
  - «expected affected population»     # LINEAR — a classical utility node
  - «CVaR at α = 0.95»                 # NOT LINEAR — needs the full distribution, not an expectation
  - «...»

IMPACT_FACTOR:                 # Service → Impact. Which of the four is it?
  kind: «DETERMINISTIC rescaling»
  # 1. RESCALING   — fixed multiplier. Deterministic; changes no ranking, but keeps the decision
  #                  basis explicit and the metric stable if 2–4 arrive later.
  # 2. AGGREGATION — combines services in incommensurable units onto one scale. Deterministic.
  # 3. STOCHASTIC  — the multiplier is itself random (day/night population, seasonal demand).
  # 4. NONLINEAR   — threshold or weighting, so CVaR(f(X)) ≠ f(CVaR(X)).
  # 3 and 4 need their own data and cannot be folded into the metric.


# ═══ G. Layer 4 — intervention ════════════════════════════════════════════════
INTERVENTIONS:                 # what a decision-maker can do. State the mechanism for each.
  - «retrofit a bridge»  → «RE-PARAMETERISE the AssetDamageState factor»
  - «close a link»       → «FIX AssetFunctionality (do-operator; severs incoming edges)»
  - «...»


# ═══ H. Reuse targets and test cases ══════════════════════════════════════════
CANDIDATE_ONTOLOGIES:          # to reuse. Give a locator if you have one.
  - «IRG v0.3.1 — doi:10.5281/zenodo.19660339 — infrastructure resilience core»
  - «INSPIRE — EU spatial data themes»
  - «InfraRisk», «COVER», «ResiliOnt», «ECMO»
  - «PROV-O, SOSA/SSN, GeoSPARQL, SKOS, QUDT»   # cross-domain; almost always relevant

DOMAIN_INSTANCES:              # examples ONLY, to prove the core flexes. Not core vocabulary.
  HazardEvent: [«flood», «earthquake»]
  Asset:       [«bridge», «roadway»]
  Network:     [«transport network A–B», «transport network C–D»]
```

---

## Part 1 — Your role and the objective

You are an ontology engineer who also builds probabilistic graphical models. Both halves matter: an
ontology that is philosophically tidy but cannot be compiled into a runnable model is a failure here, and
so is a set of BN scaffolding with no reusable semantics.

**Objective.** Produce the smallest OWL/RDF vocabulary for `DOMAIN` such that a software agent, given a
decision query matching any `QUERY_PATTERN` plus a knowledge graph of available models and datasets,
can **assemble a runnable Bayesian network on demand** — selecting the relevant variables, wiring them,
sourcing each conditional distribution, attaching available evidence, and evaluating the decision metric.

Minimal means: every class and property either (a) is required by at least one `QUERY_PATTERN`, or
(b) is required by the compilability contract in Part 5. Anything else is deferred to a named extension
point, not added "for completeness".

---

## Part 2 — The invariant architecture

Four layers. This is the part that does not change between domains. **Do not redesign it.** Your job is to
express it in the vocabulary of `DOMAIN` and align it to `CANDIDATE_ONTOLOGIES`.

### L1 — Variable layer (structure)

The nodes of the network and their dependence structure.

- A `Variable` is a node. It is **not** an entity: it is a *property of* an entity, represented as a node.
  Model this as a reified node carrying `about` (the entity), `quantityKind` (which property), and a
  **state space**. One bridge with two such properties is two variables.
- **A node need not be uncertain.** A deterministic node is a degenerate conditional distribution, and is
  a legitimate, often necessary, modelling device: aggregating several parents into a common scale,
  divorcing parents to break up an oversized conditional table, enforcing a constraint, or computing a
  value in an influence diagram. An observed constant — a design attribute known for certain — is likewise
  a valid node. What makes something a variable here is that it takes values in a declared state space and
  participates in the dependence structure, not that its value is in doubt.
- Mark a factor as **deterministic** where it is, both because it documents intent and because inference
  engines can exploit it. Do not model a deterministic relationship as a near-degenerate probability
  table with small off-diagonal values to look probabilistic.
- A property that only *parameterises* a factor — a construction year that selects which fragility curve
  applies — is a factor parameter, not a node. It becomes a node only when it is itself unknown, or when
  something else in the graph depends on it.
- A `Model` or `Simulator` is **not a variable**. It is an individual that supplies a conditional
  distribution. Represent that supply relation explicitly — call it a `Factor`: a `Factor` has one (or
  more) `targetVariable`, zero or more `conditioningVariable`, and is `suppliedBy` exactly one `Model`.
  The dependence edges of the BN are *derived* from factors, not asserted separately. Never assert a
  parent edge that no factor accounts for.
- **Several factors may target the same variable.** This is the normal case with multiple models, and the
  reason `Factor` must be first-class. Carry enough metadata to choose or combine: validity scope,
  spatial/temporal resolution, provenance, calibration status, and a precedence or weight.
- **A scoping entity is not a variable.** `STRUCTURAL_ENTITIES` carry membership, topology and simulator
  scope; they get no variable of their own unless a specific node-bearing property is named for them. In
  particular, do not insert an aggregate "whole-system state" variable between components and services.
  Such a node only earns its place if it genuinely screens the components off from the services, and
  where different services depend on different components it does not — you pay for a conditional
  distribution and buy no conditional independence.

### L2 — Data layer (evidence)

- A `Dataset` or `Observation` individual binds to the `Variable` it informs.
- **Do not assume data observes a variable directly.** Most data is a noisy or indirect measurement, so
  the binding carries a *measurement model*: either hard evidence, or a likelihood / soft evidence, or an
  explicit observation variable that is a child of the latent variable. Declare which of the three.
- Every variable declares whether it is **observable in principle** and, separately, whether data is
  **available in this graph**. The compiler needs both: the first is schema, the second is state.

### L3 — Decision layer (metrics and utilities)

- Distinguish two things that are easily conflated:
  - a **`DecisionMetric`** is a functional of the posterior — expected affected population, P(loss > x),
    CVaR. Some of these are *not* expressible as a classical utility node because they depend on the whole
    distribution rather than on expectation. Declare, per metric, whether it is expectation-linear.
  - a **`Utility`** encodes preference over outcomes and may be combined and traded off.
- A metric **reads a named L1 variable** rather than standing in for one. Make that link explicit — a
  `readsVariable` property — because it is the entry point for the backward traversal in Part 5. That
  variable is the `Impact` variable: the metric is a functional of its distribution, which is where CVaR,
  exceedance probabilities and expectations all come from. Keep the variable and the functional distinct;
  a metric is not a node, and a node is not a metric.
- If full influence-diagram optimisation is in scope, add explicit `Decision` nodes with an information
  set (what is known when the decision is taken). If not, say so and leave it as an extension point.

### L4 — Intervention layer

- An `Intervention` acts on the model, and there are **two distinct mechanisms**. Keep them separate:
  1. **Fix a variable** — Pearl's do-operator. Severs incoming edges; the mutilated graph differs from the
     observational one.
  2. **Re-parameterise a factor** — e.g. retrofit shifts a fragility curve's μ/σ. Structure unchanged,
     conditional distribution replaced.
  Mislabelling (2) as (1) silently produces wrong answers, so make the mechanism a required property.
- A `Scenario` is a named set of interventions plus assumed exogenous conditions, so scenarios can be
  compared and, later, searched over by an optimiser.

---

## Part 3 — Reuse policy

Reuse aggressively, but **import nothing wholesale.**

1. For every entry in `VARIABLE_BEARING_ENTITIES` and `STRUCTURAL_ENTITIES`, and for each architectural
   class above, search `CANDIDATE_ONTOLOGIES` for an existing term. Prefer, in order: an exact reuse of
   the term; `rdfs:subClassOf` specialisation of it; a SKOS mapping to it; minting a new term.
2. **Justify every minted term** in one line stating which existing term you rejected and why. "Nothing
   fitted" is not a justification; name the near-miss and the specific mismatch — most often a domain or
   range that is narrower than you need.
3. Structure the output as **a core module plus thin bridge modules**, one per aligned ontology. The core
   must load and be useful standalone; each bridge holds only the mapping axioms to one external
   vocabulary. This keeps a single upstream version bump from breaking everything, and lets a user adopt
   the core without accepting six external imports.
4. Prefer `owl:equivalentClass` only where the terms genuinely coincide. Where they nearly coincide use
   `skos:closeMatch` and a comment — an over-strong equivalence propagates other people's modelling
   errors into your graph.
5. Watch for **punning**: a hazard *type* and a hazard *occurrence* are different things. If the reused
   ontology models types as individuals, follow its convention rather than fighting it, and say so.
6. Record, for each external ontology, the exact version and locator you aligned against, and whether its
   namespace actually dereferences. Several published ontology namespaces do not resolve; if so, the
   alignment target must be a local file and you should note that.

---

## Part 4 — Extensibility policy

The test is: *adding a new hazard, a new asset type, a new simulator, a new service, or a new decision
metric must require no change to the core.*

- Put **type-level taxonomies in individuals or SKOS concept schemes**, not in OWL class hierarchies.
  New flood type = a new individual, not an ontology edit. Reserve subclassing for things that genuinely
  need distinct properties.
- Name your **extension points** explicitly: a short list of where growth is expected and what the pattern
  is for adding to each.
- State your **state-space discretisation policy** once, generically, rather than hard-coding bin edges
  into the core.
- Do not pre-build classes for `DOMAIN_INSTANCES`. Those are test cases that prove the core flexes; if the
  core needs a `Bridge` class to work, the core is wrong.

---

## Part 5 — Compilability contract

This is the part generic ontology work usually omits, and it is what makes on-demand assembly possible. The
graph must carry enough information for a compiler to run **without human interpretation**.

For every variable:

- **State space**, explicitly. A closed enumeration, or a named parametric family with its parameters, or
  a continuous range *plus* a discretisation rule. Reuse an existing enumeration where one exists rather
  than minting a parallel one. No variable may be left with an implicit state space.
- **Observability** and current **evidence binding** (see L2).

For every factor:

- Its target and conditioning variables, its supplying model, and **where the numbers come from**: an
  inline CPT, a parametric form, a callable simulator, or a fitted dataset. Include enough to locate and
  invoke it, not merely to describe it.
- Its **validity scope** — the conditions under which it may legitimately be used. A compiler selecting
  among competing models needs this.
- Whether parent state spaces must match its own expectations, i.e. the **interface contract** between
  factors. Mismatched discretisation between two chained models is the most common silent failure.
- Its **capability profile** from `CAPABILITIES`: which operations the underlying asset supports, its
  support type and units, whether it returns joint or marginal draws, and whether it can supply a
  normalising constant. An inference operator is admissible only where the asset offers what it needs,
  so this is what lets the compiler select a method rather than having one assigned by hand.

For query-driven assembly:

- The metadata needed to traverse from a `DecisionMetric` *backwards* through factors to the variables
  that are actually required, so the compiler can prune everything irrelevant to the query. State which
  property is traversed and in which direction.
- A worked example: for one `QUERY_PATTERN`, show the extracted subgraph as a list of
  variables, factors, evidence and metric.

**Validate with SHACL, not OWL.** `rdfs:domain` and `rdfs:range` are inference rules, not constraints — a
reasoner will infer a type rather than report a violation, so OWL alone cannot tell you a graph is
compilable. Ship SHACL shapes that enforce the bullets above as a first-class deliverable, not a later
phase. At minimum: every variable has a state space; every factor has a target, a supplier and a numeric
source; every asset declares a capability profile; every intervention declares its mechanism; every
metric declares expectation-linearity.

---

## Part 6 — Deliverables

1. **`core.ttl`** — the minimal vocabulary. Every term carries `rdfs:label` and an `rdfs:comment` saying
   what it is *for*. Includes `owl:versionInfo`.
2. **`bridge-<name>.ttl`** — one thin alignment module per external ontology reused.
3. **`shapes.ttl`** — SHACL shapes encoding the Part 5 contract.
4. **`example.ttl`** — one instance graph covering exactly one `QUERY_PATTERN` end to end, using
   `DOMAIN_INSTANCES`. Use placeholders for numeric values you do not have, clearly marked; do not invent
   plausible-looking data.
5. **Reuse table** — one row per core term: term · reused-or-minted · source ontology · justification.
6. **Binding table** — abstract core class → this domain's realisation → how a new one is added. This is
   the table that proves the core is domain-independent, and the first thing to check when the domain
   changes.
7. **Assembly walkthrough** — the Part 5 worked example, in prose plus the SPARQL that extracts it.
8. **Extension points** — the named list from Part 4, with the pattern for each.
9. **Open questions** — modelling decisions you had to guess at, each with the alternative you rejected.

---

## Part 7 — Acceptance checks

Run these against your own output before presenting it, and report the results honestly.

1. **Minimality.** For each core term, name the `QUERY_PATTERN` or Part 5 requirement that forces it.
   Any term failing this is removed or moved to an extension point.
2. **Compilability.** Walk one `QUERY_PATTERN` end to end. At every step, state where the number comes
   from. Any step requiring human judgement is a gap in the ontology — report it rather than papering over.
3. **Extensibility.** Add a new hazard, a new asset type and a new simulator *on paper*. If any requires
   editing `core.ttl`, fix the core.
4. **Dangling references.** Every external term you reference must exist in the version you aligned
   against. Check, don't assume — and say which file you checked against.
5. **Domain/range compatibility.** For every reused property, confirm your usage sits inside its declared
   domain and range, including union axioms.
6. **State-space coherence.** For every chained pair of factors, confirm the upstream output space matches
   the downstream input space.
7. **No free-riding variables.** For every variable, name either the conditional independence it buys or
   the evidence it can carry. One that does neither is an aggregate standing in the middle of a chain
   earning nothing — remove it.
8. **Impact factor classified.** Name which of the four kinds `IMPACT_FACTOR` is, and confirm the metric
   reads `Impact` rather than `Service`. If it is stochastic or nonlinear, confirm it has its own data
   source — those two cannot be folded into the metric.

---

## Part 8 — Do not

- Do not add classes because a reference ontology has them. Scope is set by `QUERY_PATTERNS`.
- Do not model a simulator or dataset as a random variable.
- Do not give a `STRUCTURAL_ENTITY` a variable just because it exists, and do not insert an aggregate
  whole-system state node between components and the functions computed over them.
- Do not have a metric read `Service` directly when an `Impact` variable exists. The decision basis is
  the impact, stated in its own units.
- Do not fold a stochastic or nonlinear `IMPACT_FACTOR` into the metric. Only a rescaling or an
  aggregation may be deterministic.
- Do not assert dependence edges that no factor accounts for.
- Do not leave a variable's state space implicit.
- Do not conflate fixing a variable with re-parameterising a factor.
- Do not invent numeric values, fragility parameters or population counts to make an example look
  complete. Mark them as placeholders.
- Do not claim an alignment you have not verified against a specific version of the target.
- Do not defer SHACL. Without it there is no definition of a compilable graph.
- Do not produce a deep subclass hierarchy where a SKOS scheme of individuals would do.

---

## Part 9 — How to work

1. Restate `QUERY_PATTERNS` as the variables, metrics and interventions they imply. Show this first —
   it is the specification everything else is checked against.
2. Inspect `CANDIDATE_ONTOLOGIES` before designing. Report what each actually provides for this problem,
   including where one is a poor fit. If you cannot access one, say so rather than guessing at its content.
3. Draft the reuse table and the binding table **before** writing Turtle. Most errors are visible there.
4. Write `core.ttl`, then `shapes.ttl`, then the bridges, then the example.
5. Run Part 7 and report results, including failures.
6. Finish with the open questions.

Think before each stage rather than emitting all files in one pass. If `DOMAIN` is under-specified for a
decision that materially changes the design, ask — but ask once, with your recommended default stated, and
proceed on that default for everything else.
