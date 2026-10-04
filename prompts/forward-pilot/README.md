# Where the two pilot networks came from

Two worked instances of **Prompt A** from tbnpy's model-construction pipeline
(`tbnpy/prompts/02_prompt_nl_to_spec.md`): the plain-language description of a
decision problem that the assembled Bayesian network was derived from.

| file | problem |
|---|---|
| [ema-highway.md](ema-highway.md) | Eastern Massachusetts highway network — access to either airport node after an earthquake |
| [hebei-transport.md](hebei-transport.md) | Hebei directed transport network — access to Handan after an earthquake |

## How to use them

Each file is the filled `description` and `known parameter values` slots. To run
the pipeline, paste the file's body into Prompt A along with
`00_context_tbnpy_api.md` and `01_model_spec_schema.md` in full, as that prompt
instructs. The output is a YAML spec for review, not code — the checkpoint is the
point.

The assembled result of that pipeline is in
[`src/forward_pilot/tbn/`](../../src/forward_pilot/tbn/): one module per node
class, as tbnpy's convention expects.

## Why they are worth keeping

They record a derivation that is otherwise invisible. Reading
`src/forward_pilot/tbn/model.py` tells you the network has an `SA1` root per
bridge; it does not tell you that the alternative — one shared ground-motion
node — was rejected because the dataset gives per-site distances, nor that
collapsing Hebei's three-state arcs to binary was a decision rather than a
property of the data. Those are in the `open_questions` sections, which is where
a reviewer should start.

Both descriptions are written to the state of the pilot as built. Where the
pilot makes an assumption that an engineer should overturn, it is listed as an
open question rather than smoothed over — in particular the absence of spatial
correlation, which affects every multi-site result the pilot produces.
