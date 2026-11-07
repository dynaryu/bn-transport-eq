# Transport networks under seismic risk, as Bayesian networks on demand

Hazard functions (a ground-motion model or a USGS ShakeMap), bridge fragility
curves and network system functions are stored as **typed
functions**: each says what it consumes and produces, and on which table of rows
it is replicated. A few parameter sets describe every bridge, and compiling an
assembly of functions over a network's rows gives the Bayesian network. A query
(by form, or asked in words and translated by a language model) runs on it.

## The six parts

| # | Part | Where | What it holds |
|---|---|---|---|
| 1 | **Schema**: what can exist | [`schema/`](schema/README.md) | quantities (the types), entity types (the plates), functions (typed signatures), and in `meta/` the JSON Schema every file obeys. [`diagram.md`](schema/diagram.md) draws it |
| 2 | **Instances**: what does exist | [`instances/`](instances/README.md) | parameter sets, filed by role: GMMs and correlation models under `hazard/`, the 28 Hazus bridge curves under `fragility/`, edge closure policies under `edge/` |
| 3 | **BN builder** | `src/builder/bn_builder.py` | validate 1 and 2, then build BN nodes and parents; the kernels behind each function and the ShakeMap reader live alongside it |
| 4 | **Inference engine** | `src/inference/` | a probabilistic query ([`query.schema.json`](src/inference/query.schema.json)) in, an answer out, by forward sampling with [tbnpy](https://pypi.org/project/tbnpy/) (PyTorch) |
| 5 | **Language** | `src/llm/` | a question in words becomes a query, the engine's numbers become an answer |
| 6 | **App: quakenet** | `src/app/` | one local page (`python -m src serve --open`): an example's scenario and models, the network map coloured by failure probabilities, the results with intervals, the BN drawn like Supabase's schema visualiser, and questions in words |

The networks the tools run on are in [`examples/`](examples/README.md): each one's rows, and its
assemblies (which function and parameters describe which rows). A report concerns one network,
so they are examples of the input rather than a library to choose from.

```
schema/  ─┐                                         question ─► src/llm ─┐
          ├─► src/builder ─► BN nodes ─► src/inference ◄── query ────────┤
instances/┤    (check, compile)           (answer)  ──► numbers ──► answer
examples/ ┘         └──────────────── src/app: draws the BN, runs both ─┘
```

## Run it

Set up once, with either conda or uv:

| | conda | uv |
|---|---|---|
| Set up | `conda env create -f environment.yml` | `uv sync` (add `--extra anthropic` for Claude) |
| Each session | `conda activate quakenet` | prefix commands with `uv run` |
| Update after a `git pull` | `conda env update -f environment.yml --prune` | `uv sync` |

```bash
python -m src check                          # 1-3: validate everything, compile every assembly
python -m src compile ema-highway/gmm        # 3: what one assembly expands to, binding by binding
python -m src build toy-ca/demo              # 3: generate runnable tbnpy data, model.py and RSR rules
python examples/toy-ca/tbnpy/demo/model.py --samples 2000  # sample with tbnpy/PyTorch
python -m src query ema-highway/gmm --event "EdgeOpen::*==False" --set ema-reference.mw=7
python -m src scope toy-ca/demo --event "SystemState::n4==Fail"     # 4: what that query depends on and samples
python -m src replay <record id>             # 4: re-run a recorded query (records/queries.jsonl) exactly
python -m src ask ema-highway/gmm "Would n1 still reach n66 after an M7?"             # 5: needs a model
python -m src serve --open                   # 6: quakenet, at http://127.0.0.1:8765
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
- **System functions run on [ndtools](https://pypi.org/project/ndtools-duco/).**
  `system.travel-time-to-nearest` gives each non-key node a binary `SystemState`: `Survival`
  if it reaches its nearest `key_destination` within `target_add_h` hours of the intact
  network's time, else `Fail` (an edge is `True` open or `False` closed). A query may name a
  state by its label or by its index (`Fail` is 0). The network supplies each edge's `length_km` and `avg_speed_kph`, and
  its `SystemEvent` row names the analysis and `target_add_h` (defaults: this function, 0.5 h).
  Rules extracted by [rsr](https://jieunbyun.github.io/rsr/) are stored with the example per
  threshold, so ndtools runs only where no rule applies. A threshold without its own rules
  gets them extracted in the background on its first run in quakenet (or by hand with
  `python -m src rules <assembly> --target-add-h <h>`); meanwhile the rules that still hold
  (survival rules from a stricter threshold, failure rules from a looser one) and ndtools give
  the same answer.
- **A query is a probability statement**: targets, threshold or joint events,
  under scenario settings and discrete node `do()` interventions. Discrete evidence
  uses rejection conditioning with acceptance diagnostics; continuous evidence is unsupported.
- **The language model only chooses the query.** Every number in an answer comes
  from the engine, and each query is shown, as a statement, before it runs.

The app also computes comparisons, bridge importance, candidate paths and one-step
inspection VOI from supplied loss tables. Review new inputs in Proposals before
activating them. [The answer cases](examples/toy-ca/answer-cases/README.md) explain
how to expand translation tests: `python -m src evaluate` checks their contracts;
`python -m src evaluate --live` evaluates held-out phrasings with the configured LLM.

The schema's design and how to add to it: [`schema/README.md`](schema/README.md).

## A query

```json
{
  "assembly": "ema-highway/gmm",
  "events": [{"node": "EdgeOpen::*", "op": "==", "value": "False"}],
  "rows": {"ema-reference": {"mw": 7.0}},
  "parameter_sets": {"edge.open": "policy.closed-at-moderate"},
  "samples": 2000
}
```

reads back as `P(EdgeOpen[*] = False) | do(ema-reference.mw = 7.0,
edge.open with policy.closed-at-moderate)`: the chance that each
edge is closed. The answer gives one probability per matching node, with a 95%
interval, and runs only the bindings those nodes depend on.

## Language models

The map's Decision query and Evidence panels (their Draft) and `python -m src ask` need one backend. Choose it with
`--llm` (or `BN_LLM`), and another model of it with `--model` (or `BN_MODEL`).

| Backend | Costs | Setup |
|---|---|---|
| `ollama` (default) | free, local | install [Ollama](https://ollama.com), then `ollama pull gpt-oss:20b` (about 14 GB; fits a 16 GB GPU). Another model with tool support via `--model` |
| `anthropic` | API credit, hosted | `ant auth login`, or `ANTHROPIC_API_KEY`; with uv, `uv sync --extra anthropic`. Claude Opus 5.5, with refusal fallbacks on |
| `gemini` | free tier, hosted | a key from aistudio.google.com/apikey in `GEMINI_API_KEY` |

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

- Inference is forward sampling only. Backward queries (diagnosis, criticality
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
- EMA's scenario epicentre and both networks' edge speeds and capacities are
  assumptions, each named in its `network.json` or parameter-set file.

## History

The repository began as a domain-general ontology-to-BN pilot. Its capability
ontology, asset contracts, 30-family GMM catalogue, deterioration placeholders,
Hazus engine, hand-built tbnpy model, and ShakeMap watcher with email alerts were
removed when the scope narrowed to this. They are all in commit `784e6cc`.
