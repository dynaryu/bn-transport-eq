# Ontology → Bayesian network, on demand

A working demonstration of one idea: **a user states a decision problem, and a
Bayesian network is assembled to answer it** — from an ontology, a registry of
knowledge assets, and the query itself, rather than being hand-built per project.

Two earthquake cases are live: access to either airport on the Eastern
Massachusetts highway network, and access to Handan on a Hebei transport network.

## Run it

Set up once, with either conda or uv:

| | conda | uv |
|---|---|---|
| Set up | `conda env create -f environment.yml` | `uv sync` |
| Each session | `conda activate dra` | (nothing; prefix commands with `uv run`) |
| Start the pilot | `python -m src.forward_pilot serve --open` | `uv run python -m src.forward_pilot serve --open` |
| Update after a `git pull` | `conda env update -f environment.yml --prune` | `uv sync` |
| Includes | everything: the Claude SDK, the tbnpy/torch backend, the conformance checks | the pilot only (no torch). Add `--extra anthropic` for Claude |

That opens the pilot at `http://127.0.0.1:8765`. Nothing else is needed: the two
cases ship as self-contained packages. The commands below are written for conda,
as `python -m ...`; with uv, run each as `uv run python -m ...`.

```bash
python -m pytest -q                # the tests
python -m src.conformance.check    # every asset against its declared contract (conda set-up)
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

- **Ask in words.** With a local [Ollama](https://ollama.com) server (free), a
  Gemini key (free tier) or a Claude credential, a chat panel appears: the language model picks the scenario
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

- `auto` (the default): ShakeCast first, then the USGS catalogue. A running
  ShakeCast server's events are read from local disk. Anything it lacks is
  fetched from the catalogue: events outside its group regions, events older
  than its `keep_eq_for` retention, or a version it hasn't received. Without a
  ShakeCast directory this is simply the catalogue. Event lists merge both
  sources. Each result says which one supplied the map, because ShakeCast can
  trail the catalogue by a version.
- `shakecast`: only the data directory of a USGS ShakeCast server
  (`SHAKECAST_USER_DIRECTORY`, default `~/.shakecast`). ShakeCast keeps an event
  only if its epicentre lies inside one of its group regions, so a shared server
  needs a group covering each network. A group with no facilities and no
  notification settings is enough: it gets the ShakeMap downloaded without
  emails or reports.
- `comcat`: only the USGS catalogue. Downloads are cached in
  `experiments/forward-bn-pilot/shakemaps/`, which is gitignored, using
  ShakeCast's folder layout. A grid is about 30 MB. With no network connection,
  the newest cached version is used. Any earlier version (`"version": 2`) is
  fetched from the catalogue too, which keeps every one.

**Watching for new earthquakes.** `shakemap watch` is the part of ShakeCast the
pilot needs. It polls the USGS real-time feed and keeps events that have a
ShakeMap, at or above M4 by default, with the epicentre around a package. It
downloads each new ShakeMap version and runs that package's forward query. No
language model is involved. A revised map is fetched and run again; an unchanged
one is not.

```bash
python -m src.forward_pilot shakemap watch --once          # one poll, for cron or a scheduled job
python -m src.forward_pilot shakemap watch --interval 120  # or keep polling every 2 minutes
python -m src.forward_pilot shakemap heartbeat             # exit 0 if alive and well, 1 if not
```

Everything goes under `experiments/forward-bn-pilot/watch/`, which is gitignored
(`--state-dir` or `FORWARD_PILOT_WATCH_DIR` to move it):

- `results/<event>/<event>-<version>/<package>.json`: the query, a summary, and
  the full result.
- `state.json`: what has been seen and run.
- `heartbeat.json`: written on every poll, successful or not.

`heartbeat` fails when the last poll is older than `--max-age-minutes` (default
10) or failed outright. Run it from a second cron entry or an alarm, so a dead
watcher is noticed before an earthquake rather than during one. One bad event
marks the poll `degraded` and does not stop the others. A lock file keeps
overlapping polls from running.

The page has a **Recent earthquakes** panel above the controls, filled straight
from USGS. It lists events of the last 30 days, M4 or more, with a ShakeMap near
a network, and checks again every 2 minutes while the page is open. No watcher
is needed for it. An event that appears after the page was opened, such as an
aftershock, is marked NEW, announced at the top of the page, and counted in the
tab title. While a ShakeMap result is on screen, the page also checks whether
USGS has revised that ShakeMap. If it has, a notice offers to re-run with the
new version. Clicking an event loads its package and runs that ShakeMap, ready
for the chat. The ShakeMap event picker uses the same M4 floor.

A minimal deployment on a Linux instance is two cron lines. `PY` is the
environment's interpreter: `.venv/bin/python` with uv, or the output of `which
python` with the conda environment active.

```
PY=/path/to/python
* * * * *    cd /path/to/bn-on-demand-pilot && $PY -m src.forward_pilot shakemap watch --once >> watch.log 2>&1
*/10 * * * * cd /path/to/bn-on-demand-pilot && $PY -m src.forward_pilot shakemap heartbeat || <your alert command>
```

**Email when an earthquake happens: the feed watcher on AWS Lambda.** The page
only sees new events while it is open. `feedwatch.py` runs without it. It polls
the same feed every 2 minutes and emails you once per earthquake near a network,
when the event's first ShakeMap appears. The email gives the magnitude, place,
time, the networks concerned, the USGS event id and its page. Events found in
the same poll share one email, so an aftershock sequence does not arrive as a
flood. Revised ShakeMaps are not emailed; the open page shows those. The watcher
downloads no grid and runs no BN. Its only dependencies are the standard library
and boto3 (part of the Lambda runtime). `shakemap watch --fetch-only` runs the
same poll locally, without email.

Deploy it with one script, from WSL or Linux with the AWS CLI logged in (`aws
sso login`). Run it inside the uv project or with the conda environment active;
it uses that Python to build the zip:

```bash
bash scripts/deploy-feedwatch.sh you@example.com    # subscribes this address; confirm the email AWS sends
```

It creates:
- an S3 bucket, `bn-pilot-watch-<account>` in `ap-southeast-2`;
- an SNS topic for the emails;
- the zip (`shakemap lambda-zip`);
- a role limited to logs, the bucket, the topic and the health metric;
- the function, with reserved concurrency 1 so polls never overlap;
- a schedule that runs it every 2 minutes;
- an alarm that emails the same topic when no healthy poll has been reported for
  10 minutes, because polls are failing or the schedule has stopped.

It finishes with one test poll. It is safe to re-run: existing pieces are kept,
and the function gets fresh code. That is also how to update it after a
package's network changes. The script's header lists its overrides (region,
bucket, names).

The bucket holds two small files under `watch/`. `state.json` records each
event, its networks, its ShakeMap versions, and whether it has been emailed.
`heartbeat.json` records the last poll. A Lambda keeps nothing between runs, so
they live in S3. Nothing needs syncing to your machine: the page asks USGS
itself. A failed email is retried on the next poll.

The function's settings are environment variables, set in the Lambda console;
re-running the script keeps them. They are `FEEDWATCH_MIN_MAGNITUDE` (default
4), `FEEDWATCH_FEED` (default `2.5_day`), `FEEDWATCH_TOPIC_ARN` (set by the
script) and `FEEDWATCH_PREFIX` to share a bucket. Expected cost is well under
US$1 a month: about 22,000 one-second invocations (inside Lambda's free tier),
two small S3 writes per poll, one custom metric with an alarm, and a few
emails. Check current AWS pricing for your account.

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

Pick the model behind the panel when starting the server with `--llm`. Add
`--model` to use a model of that backend other than its default.

| Backend | Costs | Setup | Start |
|---|---|---|---|
| Ollama (default) | free, runs locally | install Ollama, `ollama pull qwen3:8b` | `python -m src.forward_pilot serve` |
| Gemini | free tier, hosted | a key from aistudio.google.com/apikey in `GEMINI_API_KEY` | `python -m src.forward_pilot serve --llm gemini` |
| Claude | paid API credit, hosted | `ant auth login` or `ANTHROPIC_API_KEY`. The SDK comes with the conda environment; with uv, `uv sync --extra anthropic` | `python -m src.forward_pilot serve --llm anthropic` |

For example, Claude with an API key:

```bash
# conda
conda activate dra
export ANTHROPIC_API_KEY="sk-ant-..."      # PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."
python -m src.forward_pilot serve --llm anthropic --open

# uv
uv sync --extra anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python -m src.forward_pilot serve --llm anthropic --open
```

Keep the key out of the repository.

On startup the server prints which backend and model it is using, and either
"ready" or what is missing. The default models are `qwen3:8b`,
`gemini-3.8-flash` and `claude-opus-5-5`. `FORWARD_PILOT_LLM` and
`FORWARD_PILOT_MODEL` do the same as the flags, for a shell profile.

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
| `--model` | `qwen3:8b` | Any pulled model with tool support, e.g. `llama3.1:8b`, `mistral-small` |
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
   export GEMINI_API_KEY="..."             # keep it out of the repository
   python -m src.forward_pilot serve --llm gemini
   ```

The free tier is rate-limited, and Google may use free-tier prompts to improve
its products. Check the current terms before sending anything sensitive. A
rate-limit error appears in the panel as a sentence, not a traceback. Gemini is generally
more reliable than a local 8B model at choosing parameters and at quoting the tool's
numbers accurately. The numbers still come only from the engine.

### Anthropic: Claude, with a credential

Start the server with `--llm anthropic`. The SDK is already in the conda
environment; with uv, run `uv sync --extra anthropic` first. Then get a
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
