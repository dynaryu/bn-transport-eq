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
| 3 | Make each asset sign that contract | `assets/*/*/asset.ttl` + `model.py` |
| 4 | Check the contract is true, not just claimed | `src/conformance/check.py` |
| 5 | State a decision problem in plain language | `prompts/forward-pilot/` |
| 6 | Assemble a portable BN package for it | `src/forward_pilot/package.py` |
| 7 | Build the network itself, in tbnpy | `src/forward_pilot/tbn/` |
| 8 | Query it — by form, or by asking in words | `src/forward_pilot/server.py`, `chat.py` |

## The three ideas worth arguing about

**An asset declares what it can do, and the claim is tested.** `asset.ttl` says
whether a model can evaluate a density, sample forward, return a likelihood.
`conformance/check.py` calls every declared capability and reports *overclaim*
separately from *underclaim*, because a capability claimed but absent makes the
compiler pick an operator that cannot run — which surfaces as a wrong posterior
rather than an error.

**An asset may refuse to answer.** `assets/hazard/gmpe/` registers 30 published
ground-motion models and **cannot predict with any of them**: no authoritative
coefficient table is in the repository, so construction raises and reports what
source material is missing. Inventing plausible coefficients would produce
confident numbers that no downstream test could catch.

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

- **Ask in words.** With an Anthropic credential a chat panel appears: Claude picks
  the scenario parameters, the engine computes the numbers, and every answer shows
  the query it ran. Claude is never asked to estimate a probability. Setting up the
  credential is its own section below.
- **tbnpy and torch** build the Bayesian network. Without them a closed-form
  backend runs instead, and an equivalence test holds the two together.
- **Rebuilding packages** from the raw datasets needs a `network-datasets`
  checkout and `NETWORK_DATASETS_ROOT`. Only needed to change a case, not to run
  one.

## Asking in words

The chat panel takes a question in plain language: *"How reliable is access to
Handan if the whole network shakes at 0.7 g?"* becomes a tool call carrying the
scenario parameters, `run_forward_query` runs exactly as it does for the form
controls, and the reply reports the numbers that came back under the query that
produced them. Claude is never asked to estimate a probability.

The panel stays hidden until the **server process** holds a credential.
`chat.available()` decides, `/api/chat-status` reports the reason, and the page
asks once on load — so set the credential first, start the server, then reload
the page. To check without hunting for the panel:

```powershell
curl http://127.0.0.1:8765/api/chat-status
```

`{"available": true, ...}` means the panel will appear; otherwise `reason` names
what is missing.

### Getting a credential

**Either route below works, and neither is a Claude subscription.** API access
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

The panel calls `claude-opus-5`, set as `chat.MODEL` — one line to change if you
want a cheaper model, and the only place the model appears.

The engine itself is free: the Monte Carlo sampling behind every number in this
repository costs nothing but your own CPU, and the form controls reach all of it
without a credential.

## Not included

This is a curated subset of a larger working repository. Left out: four network
simulator adapters that need an external dataset package, an exact-enumeration BN
compiler that depends on them, and the core ontology/SHACL layer, which is not
written yet. Everything here runs from a clean clone.
