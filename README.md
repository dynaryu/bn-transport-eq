# Transport networks under seismic risk, as Bayesian networks on demand

Hazard functions (a ground-motion model or a USGS ShakeMap), bridge fragility
curves and network simulators (shortest path, maximum flow) are stored as **typed
functions**: each says what it consumes and produces, and on which table of rows
it is replicated. A few parameter sets describe every bridge, and compiling an
assembly of functions over a network's rows gives the Bayesian network. A query
(by form, or asked in words and translated by a language model) runs on it.

## The six parts

| # | Part | Where | What it holds |
|---|---|---|---|
| 1 | **Schema**: what can exist | [`schema/`](schema/README.md) | quantities (the types), entity types (the plates), functions (typed signatures), and in `meta/` the JSON Schema every file obeys. [`diagram.md`](schema/diagram.md) draws it |
| 2 | **Instances**: what does exist | [`instances/`](instances/README.md) | parameter sets (28 Hazus bridge curves, GMM coefficients, closure policies), networks (rows), assemblies (which function and parameters describe which rows) |
| 3 | **BN builder** | `src/builder/` | validate 1 and 2, then compile an assembly into BN nodes and parents; the kernels behind each function; the ShakeMap reader |
| 4 | **Inference engine** | `src/inference/` | a probabilistic query ([`query.schema.json`](src/inference/query.schema.json)) in, an answer out, by forward Monte Carlo |
| 5 | **Language** | `src/llm/` | a question in words becomes a query, the engine's numbers become an answer |
| 6 | **App** | `src/app/` | one local page: the BN drawn like Supabase's schema visualiser, a query form and a chat panel |

```
schema/  ─┐                                         question ─► src/llm ─┐
          ├─► src/builder ─► BN nodes ─► src/inference ◄── query ────────┤
instances/┘    (check, compile)           (answer)  ──► numbers ──► answer
                     └──────────────── src/app: draws the BN, runs both ─┘
```

## Run it

Set up once, with either conda or uv:

| | conda | uv |
|---|---|---|
| Set up | `conda env create -f environment.yml` | `uv sync` (add `--extra anthropic` for Claude) |
| Each session | `conda activate dra` | prefix commands with `uv run` |
| Update after a `git pull` | `conda env update -f environment.yml --prune` | `uv sync` |

```bash
python -m src check                          # 1-3: validate everything, compile every assembly
python -m src compile ema-highway/gmm        # 3: what one assembly expands to, binding by binding
python -m src query ema-highway/gmm --event "SystemState::*>n66:time==0" --set ema-reference.mw=7
python -m src ask ema-highway/gmm "Would n1 still reach n66 after an M7?"             # 5: needs a model
python -m src serve --open                   # 6: the page, at http://127.0.0.1:8765
python -m src export --open                  # 6: its drawings as one file, no server (.cache/schema.html)
python -m src diagram                        # 1: rewrite schema/diagram.md after editing the schema
python -m pytest -q                          # the tests
```

## The idea, in one line per part

- **A function is a typed signature**, `outputs = f(inputs ; fixed data ; θ)`. Inputs
  are BN parents, wired only between ports of the same quantity; fixed data are row
  attributes such as a bridge's class; θ is a parameter set many rows share.
- **A function is replicated over a plate** (`per: Bridge`), and each input says
  how one row finds its parents (the same row, every bridge on this edge, all edges).
- **An assembly populates the BN**: `"parameter_set": "hazus61.{hazus_class}"` gives
  each of the 129 EMA bridges one of three curves. A GMM and a ShakeMap both
  produce an IM on `Bridge`, from each bridge's own lat/lon, so swapping one for the other is one line.
  Which GMM is chosen by where the network is, and the IM itself is left open:
  `PGA`, `SA(0.3)`, `SA(1.0)`... wire only to curves on the same IM.
- **A system event is defined in the network**: each `SystemEvent` row names its
  analysis (shortest path or max flow), its origin and destination, and what counts
  as surviving (a travel time within `travel_time_tolerance` of the intact network's,
  or a max flow of at least `min_flow`). Each becomes a binary `SystemState` node.
- **A query is a probability statement**: targets and threshold events, under
  `do()`-style settings of rows and parameters. Evidence is refused: the engine is
  forward-only.
- **The language model only chooses the query.** Every number in an answer comes
  from the engine, and each query is shown, as a statement, before it runs.

The schema's design and how to add to it: [`schema/README.md`](schema/README.md).

## A query

```json
{
  "assembly": "ema-highway/gmm",
  "events": [{"node": "SystemState::*>n66:time", "op": "==", "value": 0}],
  "rows": {"ema-reference": {"mw": 7.0}},
  "parameter_sets": {"functionality.usable-from-damage": "policy.closed-at-moderate"},
  "samples": 2000
}
```

reads back as `P(SystemState[*>n66:time] = 0) | do(ema-reference.mw = 7.0,
functionality.usable-from-damage with policy.closed-at-moderate)`: the chance that each
journey to n66 fails its event, i.e. takes more than 1.5 × its intact time or finds no
route. The answer gives one probability per matching event, with a 95% interval, and
runs only the bindings those nodes depend on (max flow is skipped here).

## Language models

The page's chat panel and `python -m src ask` need one backend. Choose it with
`--llm` (or `BN_LLM`), and another model of it with `--model` (or `BN_MODEL`).

| Backend | Costs | Setup |
|---|---|---|
| `ollama` (default) | free, local | install [Ollama](https://ollama.com), then `ollama pull qwen3:8b` (any model with tool support) |
| `gemini` | free tier, hosted | a key from aistudio.google.com/apikey in `GEMINI_API_KEY` |
| `anthropic` | API credit, hosted | `ant auth login`, or `ANTHROPIC_API_KEY`; with uv, `uv sync --extra anthropic` |

On startup the server prints the backend and whether it is ready;
`/api/chat-status` gives the reason when not. A small local model may translate a
question into the wrong query, never the wrong numbers, and the stated query is
there to check. Ollama settings: `OLLAMA_BASE_URL`, `OLLAMA_THINK=1`,
`OLLAMA_KEEP_ALIVE`. Claude API access bills to an Anthropic Console account, not a
Claude subscription, and a key on a zero balance authenticates and then fails on
the first question.

## ShakeMap

`hazard.shakemap` takes each bridge's median IM (e.g. `PSA10` for `SA(1.0)`) from the ShakeMap's `grid.xml` and
its log standard deviation from `uncertainty.xml`, interpolated on the log scale, and
sets ln IM = ln median + σ·ε. The residual ε is its own node, `GMResidual`, drawn
jointly over all bridges by `hazard.spatial-correlation` with Jayaram & Baker (2009),
ρ(h) = exp(−3h/b) (b = 25.7 km at 1 s), as OpenQuake does for ShakeMap scenarios:
independent normals, truncated at 3σ, then correlated. Choosing the parameter set
`independent.any-im` instead turns the correlation off, for comparison; pick the
Jayaram & Baker set whose period matches the IM (`jayaram-baker-2009.pga`, `.sa1p0`, ...). Results name the map
version used, and say when the map has no station data or no uncertainty grid.

The page's **Find ShakeMaps near this network** button lists candidate events. Grids
come from a ShakeCast server's data directory (`SHAKECAST_USER_DIRECTORY`) when it
has them, else from the USGS catalogue, cached in `.cache/shakemaps/`
(`BN_SHAKEMAP_SOURCE` = `auto`, `shakecast` or `comcat`). A grid is about 30 MB.

## Limits, stated

- Inference is forward Monte Carlo only. Backward queries (diagnosis, criticality
  given an outage) are refused, not approximated.
- ShakeMap residuals are spatially correlated, but with the unconditional
  Jayaram & Baker model applied to ShakeMap's station-conditioned σ (OpenQuake's
  approximation). Near stations the true conditional correlation is weaker.
- Each binding carries one IM. Two IMs at once (e.g. SA(0.3) and SA(1.0) for Hazus's
  shape modifier) means two bindings with independent residuals; their cross-IM
  correlation is not modelled.
- Only one GMM's numbers are in the repository: Campbell (2003), for SA(1.0) in
  eastern North America, and its region is an approximate box. Another model is a
  parameter set (or, for a new functional form, a kernel too), with its published coefficients.
- The GMM path draws bridges independently: the Campbell (2003) set has only a total
  σ, and correlating it properly needs the between- and within-event split.
- Nothing checks that a residual field's period matches the IM it multiplies; the
  parameter-set labels say which is which.
- Kernels implement `SampleForward`; only the fragility also has `EvaluateDensity`.
- EMA's scenario epicentre and both networks' link speeds and capacities are
  assumptions, each named in its `network.json` or parameter-set file.

## History

The repository began as a domain-general ontology-to-BN pilot. Its capability
ontology, asset contracts, 30-family GMM catalogue, deterioration placeholders,
Hazus engine, hand-built tbnpy model, and ShakeMap watcher with email alerts were
removed when the scope narrowed to this. They are all in commit `784e6cc`.
