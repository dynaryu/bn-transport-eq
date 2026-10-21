# 1. Schema: what can exist

The schema stores **functions, not just their metadata**. Each one is a typed
signature that says what it consumes and produces, and on which table of rows it
is replicated:

```
outputs[per-row]  =  f( inputs  ;  fixed data  ;  θ )
                       ▲            ▲              ▲
                       BN parents   row attributes  a parameter set,
                       (wired by    and whole       shared by every row
                       type + via)  tables          bound to it
```

The split into three kinds of argument is what lets a few records describe a
whole network. Three Hazus parameter sets (θ) serve all 129 EMA bridges: each
bridge's class (fixed data) picks its set, and its own lat/lon (fixed data) is where
the hazard functions evaluate its shaking.

**Two drawings of all of this**, both generated from the files below by
`python -m src diagram` (add `--open` to view the HTML one), and checked by a test
that fails if either falls behind them:

| File | Holds |
|---|---|
| `diagram.html` | the interactive drawing: open it in a browser. **Model** puts the functions and the tables they read in one picture: solid wires are random variables, dashed ones the data a function reads (attributes, whole tables, the keys that route parents), dotted ones foreign keys. **Functions** and **Plates** show each half alone |
| `diagram.md` | the same as Mermaid diagrams, which GitHub renders, plus a table of quantities |
| `quantities.json` | the types that flow between functions: a unit and range, ordered states, or record fields |
| `entities.json` | the plates: tables of rows, with typed attributes and foreign keys (`ref`) |
| `functions/<id>.json` | one typed signature each: its plate, ports, the attributes and tables it reads, a JSON Schema for its parameters, and the kernel that implements it |
| `meta/*.schema.json` | the JSON Schema each file here and in `instances/` obeys |

The instances that fill this schema (parameter sets, networks, assemblies) are in
[`../instances/`](../instances/README.md).

## How a function is replicated: `per` and `via`

A function's `per` is its plate. Each input names the plate its parent lives on
(`over`) and how to reach it from one row (`via`):

| `via` | Parents of one row | Example |
|---|---|---|
| `self` | the same row | `Usable[bridge] ← DamageState[bridge]` |
| `<fk>` | the row this foreign key points at | e.g. `X[bridge] ← LinkOpen[bridge.edge]`, a bridge reading its own link's state; none in use yet |
| `<Over>.<fk>` | every `Over` row whose key points here | `LinkOpen[edge] ← Usable[b] for every b with b.edge = edge` |
| `*` | every row of `Over` | `TravelTime[od] ← LinkOpen[all edges]` |

`*` over a singleton plate is how global variables enter: `Rupture` lives on the
one `Event` row, and every bridge's ground-motion node shares it as a parent.

## Intensity measures are left open

`SA1` is no longer special. The IMs are quantities in their own right (`PGA`, `PGV`,
`SA(0.3)`, `SA(1.0)`, `SA(3.0)`, named as in OpenQuake), and a port may declare its
quantity as `"{imt}"`: the IM its parameter set names. The GMM, the ShakeMap reader
and the fragility curve are all declared that way, so one function serves every IM.

Wiring is still checked on the resolved types. Hazus's curves say `imt: "SA(1.0)"`,
so they connect only to a hazard whose set also says `SA(1.0)`; a PGA hazard feeding
them is a compile error that names each bridge and what it has instead. For a templated
port the check also demands that `imt` be a required parameter whose allowed values
are all quantity ids, so every set that validates resolves to a real type.

## Hazard: two routes to the same IM

Both produce `{imt}` on the `Bridge` plate, so either feeds any fragility function
on the same IM:

- **GMM** (`hazard.gmm`): one function for every ground-motion model. Each parameter
  set is one model: its functional `form` (a kernel in `GMM_FORMS`), the `imt` it
  predicts, its coefficients and σ, and a `region` where it applies. Campbell (2003)
  is one such set. The parent is `Rupture[Event]`, the global parameters; fixed data,
  the bridge's lat/lon. The scenario is itself a root node (`hazard.scenario-fixed`), so
  making magnitude uncertain means swapping that one function for a prior.

  **Which GMM is chosen by where the network is.** A binding with
  `"parameter_set_by_location": ["model.a", "model.b"]` gives each bridge the first
  listed set whose `region` polygon contains it. EMA's bridges all fall in Campbell's
  "central and eastern North America" box. A bridge no listed region covers is an
  error that names it, never a silent fallback. A query can still force one set
  everywhere with `parameter_sets`.
- **ShakeMap** (`hazard.shakemap`): the map is the event's own data, so θ names the
  event. Each bridge reads its median from `grid.xml` and its σ from `uncertainty.xml`
  at its lat/lon; its one parent is its residual, `GMResidual[bridge]`.

Spatial correlation lives in that residual: `hazard.spatial-correlation` draws
`GMResidual` jointly over the whole `Bridge` plate (`drawKind: Joint`), with Jayaram &
Baker (2009) or, as a parameter set, independent bridges. Making it a node rather than
burying it in the ShakeMap kernel keeps the dependence visible in the BN, where
observed intensities could later condition it. A GMM with a between/within-event
split would add an inter-event residual on `Event` and reuse the same field for
the within-event part: new functions and ports, no new kind of record.

## What `python -m src check` refuses

- a port naming an unknown quantity, or a `via` that does not follow a real foreign key;
- parameter values that do not satisfy their function's `parameters` schema;
- a declared capability the kernel lacks (`OVERCLAIM`);
- rows with unknown or missing attributes, wrong types, or dangling foreign keys;
- a function reading an attribute that some bound row lacks;
- a `parameter_set` template that resolves to a missing set (it names the rows);
- a `parameter_set_by_location` that leaves some row uncovered (it names the rows);
- an `{imt}` port whose parameter is not a required enum of quantity ids;
- a node with no producer or with two, and any cycle among bindings.

Wiring is by resolved quantity, so a hazard on one IM cannot feed curves on another
silently: the check names the bridges and the IM they do have.

## Adding things

- **A bridge class**: add a set to `instances/parameters/hazus61-bridge-sa1.json`.
  Rows with that `hazus_class` pick it up through `"hazus61.{hazus_class}"`.
- **A fragility model**, e.g. one using span count: a new function file with
  `"attributes": ["span_count"]`, and that attribute on `Bridge`. The check then
  lists every bridge that lacks it.
- **A GMM of an existing form**: only a parameter set for `hazard.gmm`, with its
  `form`, `imt`, coefficients, σ and `region`, then list it in the assembly's
  `parameter_set_by_location`. No code.
- **A GMM of a new form**: a function in `GMM_FORMS` (`src/builder/kernels/hazard.py`)
  returning ln median and σ, and a branch for its coefficients under `allOf` in
  `functions/hazard.gmm.json`. If it needs Vs30, add `vs30` to `Bridge` and list it
  under `attributes`.
- **Curves on another IM**: a parameter set for `fragility.lognormal` with that `imt`.
- **A system function**: a function `per: SystemEvent` reading `LinkOpen` via `*`,
  producing its metric and a `SystemState`, and reading its threshold as a
  `SystemEvent` attribute. Bind it with `where: {"analysis": ...}`.
- **A system event**: a `SystemEvent` row in the network, with its `analysis`,
  `origin`, `destination` and threshold.
- **A kernel** lives in `src/builder/kernels/`. It gets a `Call` (rows, input
  arrays, θ) and returns `(n_samples, n_rows)` arrays, one per output port.

The records map one-to-one onto relational tables (quantity, entity, function,
port, parameter_set, binding), so they could be loaded into Postgres or Supabase.
They are kept as files because contributions and reviews are diffs.
