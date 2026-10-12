# Ontology → Bayesian network, on demand

A working demonstration of one idea: **a user states a decision problem, and a
Bayesian network is assembled to answer it** — from an ontology, a registry of
knowledge assets, and the query itself, rather than being hand-built per project.

Two earthquake cases are live: access to either airport on the Eastern
Massachusetts highway network, and access to Handan on a Hebei transport network.

## Run it

```powershell
conda env create -f environment.yml
conda activate dra
python -m src.forward_pilot serve --open
```

That opens the pilot at `http://127.0.0.1:8765`. Nothing else is needed — the two
cases ship as self-contained packages.

```powershell
python -m src.conformance.check    # every asset against its declared contract
python -m pytest -q                # 210 tests
```

## The workflow, in the order it happens

| # | Step | Where |
|---|---|---|
| 1 | Design the ontology from a domain brief | `prompts/bn-ontology-design-brief.md` |
| 2 | Declare what a model asset *can compute* | `ontology/capabilities.ttl` |
| 3 | Make each asset sign that contract | `assets/*/*/*.asset.json` or `asset.ttl`, + `model.py` |
| 4 | Check the contract is true, not just claimed | `src/conformance/check.py` |
| 5 | State a decision problem in plain language | `prompts/forward-pilot/` |
| 6 | Assemble a portable BN package for it | `src/forward_pilot/package.py` |
| 7 | Build the network itself, in tbnpy | `src/forward_pilot/tbn/` |
| 8 | Query it — by form, or by asking in words | `src/forward_pilot/server.py`, `chat.py` |

## The three ideas worth arguing about

**An asset declares what it can do, and the claim is tested.** Its declaration says
whether a model can evaluate a density, sample forward, return a likelihood.
`conformance/check.py` calls every declared capability and reports *overclaim*
separately from *underclaim*, because a capability claimed but absent makes the
compiler pick an operator that cannot run — which surfaces as a wrong posterior
rather than an error.

The declarations are many small files, as a registry filled by a community would
be: each bridge class, roadway class and ground-motion family is one
`<name>.asset.json` carrying its own contract, provenance and numbers — 60 of them.
They all observe one schema, and it is not written by hand:
`src/conformance/schema.py` derives `ontology/asset.schema.json` from the ontology,
so a contributed file is checked against the vocabulary the compiler actually
reads. See [the asset registry](assets/README.md).

**An asset may refuse to answer.** `assets/hazard/gmpe/` registers 30 published
ground-motion models and **cannot predict with any of them**: no authoritative
coefficient table is in the repository, so construction raises and reports what
source material is missing, and each of the 30 files declares no capability.
Inventing plausible coefficients would produce confident numbers that no
downstream test could catch.

**The query boundary is enforced in code, not in prose.** The pilot answers
forward questions only. "Which bridge is most critical given the outage?" needs
backward inference, so `engine._validate_query` refuses it — including when the
question arrives in natural language, because that path routes to the same
validator.

## What you are looking at in the UI

Three views of the same model: **plate** notation, the **actual tbnpy nodes**, and
the network **over the map** — ground motion above each bridge, usability at the
bridge itself, service at the junctions below. The last is read off the live
model, so a node that gains a parent gains an arrow.

The EMA network is 331 variables across 260 nodes.

## Optional extras

- **Ask in words.** With a local [Ollama](https://ollama.com) server (free) or an
  Anthropic credential, a chat panel appears: the language model picks the scenario
  parameters, the engine computes the numbers, and every answer shows the query it
  ran. The model is never asked to estimate a probability. Setting up a backend is
  its own section below.
- **tbnpy and torch** build the Bayesian network. Without them a closed-form
  backend runs instead, and an equivalence test holds the two together.
- **Rebuilding packages** from the raw datasets needs a `network-datasets`
  checkout and `NETWORK_DATASETS_ROOT`. Only needed to change a case, not to run
  one.

## Real earthquakes: USGS ShakeMap

Both packages can take the ground motion of a real earthquake instead of an
analyst-set intensity. Choose **USGS ShakeMap** as the hazard representation in
the UI, then pick an event: the list is the USGS events with a ShakeMap around the
network. Each bridge then gets the ShakeMap's median SA(1 s) and its uncertainty
at the arc midpoint, and the forward query runs as usual. The chat tool takes
`shakemap_event_id` too, so "what if the 1966 Xingtai earthquake happened again?"
works.

```bash
python -m src.forward_pilot shakemap search --case hebei-transport --min-magnitude 6
python -m src.forward_pilot shakemap fetch iscgem849480      # Xingtai 1966, M6.8
```

Query JSON: `"hazard": {"model": "shakemap", "event_id": "iscgem849480"}`, with
optional `"version"` and `"source"`.

**Where the maps come from** (`FORWARD_PILOT_SHAKEMAP_SOURCE`):

- `comcat` (the default): searched and downloaded from the USGS catalogue, and
  cached in `experiments/forward-bn-pilot/shakemaps/`, which is gitignored. A grid
  is about 30 MB. With no network connection, the newest cached version is used.
- `shakecast`: read from the data directory of a running USGS ShakeCast server
  (`SHAKECAST_USER_DIRECTORY`, default `~/.shakecast`). ShakeCast already watches
  for new events and saves each ShakeMap version, so the pilot follows whatever it
  has received. The ComCat cache uses the same folder layout, so either source can
  read the other's files.

**Limits to keep in mind** — every result states these in its warnings:

- Hebei's lat/lon are georeferenced from a schematic figure: city coordinates
  for labelled nodes, and about ±22 km for junctions. EMA's are fitted to town
  centres, about ±7 km. See each dataset's PROVENANCE.md. Bridge positions are arc midpoints, not surveyed sites.
- Historical Chinese events are ShakeMap Atlas reconstructions, mostly without
  station recordings, so their shaking is a ground-motion-model estimate with a
  large sigma.
- Sites are independent given the map: the spatial correlation of ground motion
  is not represented.
- A bridge outside the map's extent is refused, not treated as unshaken.

EMA's lat/lon are fitted to the town of each node, named in the source benchmark
(held-out error about 7 km). New England's recorded events are small: the 2020 M3.6
in Buzzards Bay (`us7000cc4d`) covers the network, but peaks at 0.003 g. EMA's
reference scenario still reads x/y as kilometres, but one unit is about 1.81 real km.

## Asking in words

The chat panel takes a question in plain language: *"How reliable is access to
Handan if the whole network shakes at 0.7 g?"* becomes a tool call carrying the
scenario parameters, `run_forward_query` runs exactly as it does for the form
controls, and the reply reports the numbers that came back under the query that
produced them. The model is never asked to estimate a probability.

The reply is built in the order the work is done. First the query, written as a
probability statement in the model's own variables — `P(S::o = 1 | Mw = 7,
epicentre = (1, 10))` for each origin, `P(X::e = 0 | …)` for each bridge,
`E[PopFraction | …]` — and shown before the engine is called. Then the values
the engine computed for it. Then the prose. `/api/ask` streams these as one JSON
object per line (`query`, `computed` or `not_computed`, `answer`) so the page can
show each as it happens.

The panel stays hidden until the **server process** can reach a model.
`chat.available()` decides, `/api/chat-status` reports the reason, and the page
asks once on load — so start Ollama or set the credential first, then the server, then reload
the page. To check without hunting for the panel:

```powershell
curl http://127.0.0.1:8765/api/chat-status
```

`{"available": true, ...}` means the panel will appear; otherwise `reason` names
what is missing.

### Choosing a backend

`FORWARD_PILOT_LLM` picks the model behind the panel: `ollama` (the default),
`gemini` or `anthropic`. Set it in the shell that starts the server.

### Ollama: free, local (the default)

The model runs on your own machine and nothing leaves it. The server talks to
Ollama's HTTP API with the standard library, so there is no Python package to add.

```bash
curl -fsSL https://ollama.com/install.sh | sh   # Linux / WSL; Windows and macOS installers at ollama.com
ollama serve                                     # if the installer did not start it as a service
ollama pull qwen3:8b                             # ~5 GB; needs a model with tool support
```

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:8b` | Any pulled model with tool support, e.g. `llama3.1:8b`, `mistral-small` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the Ollama server listens |
| `OLLAMA_THINK` | off | `1` turns on the model's thinking mode: slower (about 2× in a test with qwen3:8b), rarely needed to pick parameters |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded after a question. The server also loads it at startup, so the first question does not wait for it |

A small model is worse than Claude at turning a question into parameters: it may
pick the wrong scenario or misread a place name. It cannot get the numbers wrong,
because it never produces them, and the query it chose is shown before it runs —
so check that line. On a CPU without a GPU each answer can take a minute or more;
`OLLAMA_TIMEOUT` in `chat.py` caps a single call. On WSL, run Ollama inside WSL,
alongside the server; a Windows-side Ollama is not on WSL's `localhost` unless
mirrored networking is enabled.

### Gemini: hosted, with a free tier

Google's Gemini models, through their OpenAI-compatible endpoint. No Python
package is needed; the server calls the API with the standard library.

1. Create a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Start the server with the key in its environment:

   ```bash
   export FORWARD_PILOT_LLM=gemini
   export GEMINI_API_KEY="..."             # keep it out of the repository
   export GEMINI_MODEL=gemini-3.8-flash    # optional; this is the default
   python -m src.forward_pilot serve
   ```

The free tier is rate-limited, and Google may use free-tier prompts to improve
its products. Check the current terms before sending anything sensitive. A
rate-limit error appears in the panel as a sentence, not a traceback. Gemini is generally
more reliable than a local 8B model at choosing parameters and at quoting the tool's
numbers accurately. The numbers still come only from the engine.

### Anthropic: Claude, with a credential

Set `FORWARD_PILOT_LLM=anthropic` and `pip install anthropic`, then get a
credential. **Either route below works, and neither is a Claude subscription.** API access
bills to an Anthropic Console account with its own credits; a Claude Pro or Max
plan, or a Claude Code login, does not grant it. Signing up as an individual is
enough — the Console calls your account an *organisation* and creates one for
you, which is a billing container, not a company you have to register.

*Signing in — nothing to copy or store:*

```powershell
ant auth login          # opens a browser; pick the org and workspace
ant auth status         # shows which credential source won
```

The CLI stores an OAuth profile under `%APPDATA%\Anthropic` and sets no
environment variable; the SDK finds it on its own, which is why
`chat.available()` looks there as well as at the environment. Install `ant` from
[its releases](https://github.com/anthropics/anthropic-cli/releases), or with
`go install github.com/anthropics/anthropic-cli/cmd/ant@latest` (Go 1.22+).

*Or an API key, if you would rather hold one.* Three steps, in the
[Anthropic Console](https://console.anthropic.com) and then your shell:

1. **Create the key** at
   [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
   → *Create Key*, and copy it there and then. The console never shows a key
   again — only its name, creation date and a truncated prefix — so a key you
   did not save cannot be recovered. Delete it and make another.
2. **Add credit** at
   [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing).
   This is the step that surprises people: the key authenticates fine on a zero
   balance, so the panel appears and then fails on the first question.
3. **Give it to the server**, in the shell that will start it:

   ```powershell
   $env:ANTHROPIC_API_KEY = "sk-ant-..."   # this shell only
   setx ANTHROPIC_API_KEY "sk-ant-..."     # or persist it, then open a new shell
   ```

One trap worth knowing: a key set to an *empty* string still outranks a CLI
profile and authenticates with nothing, so unset it rather than blanking it.
`chat.available()` reports that case specifically.

When a question does fail, the panel shows a sentence rather than a traceback,
and the full SDK error goes to the server log — the mapping is `_API_ERRORS` in
`server.py`. A rejected credential and an exhausted balance each get their own
line there, because they are the two that bite first.

Whichever route, the credential stays server-side: the browser talks only to
`127.0.0.1`, never to Anthropic. Keep it out of the repository — nothing here is
gitignored for you.

### What a question costs

Each question is a tool round-trip — one short system prompt, a tool result, and
a few sentences back — so two or three API calls, on the order of a cent or two
at Claude Opus 5 rates ($5 per million input tokens, $25 per million output).
For the handful of questions a demonstration needs, the console's minimum credit
top-up is the real cost, not the usage.

With the Anthropic backend the panel calls `claude-opus-5`, set as
`chat.ANTHROPIC_MODEL` — one line to change if you want a cheaper model. The
Ollama backend costs nothing per question.

The engine itself is free: the Monte Carlo sampling behind every number in this
repository costs nothing but your own CPU, and the form controls reach all of it
without a credential.

## Not included

This is a curated subset of a larger working repository. Left out: four network
simulator adapters that need an external dataset package, an exact-enumeration BN
compiler that depends on them, and the core ontology/SHACL layer, which is not
written yet. Everything here runs from a clean clone.
