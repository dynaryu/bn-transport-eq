# Asset registry

One folder per knowledge asset. Every asset has the same five parts, whatever
kind of model it is:

```
<category>/<asset-name>/
├─ asset.ttl     # the contract: target, parents, state spaces, capabilities
├─ README.md     # what it maps, why each capability holds, assumptions
├─ SOURCE.md     # citation, development background, licence, retrieval date
├─ model.py      # the wrapper, implementing exactly what asset.ttl declares
├─ params/       # the numbers, separate from the code that uses them
└─ tests/        # known-value regression
```

**`asset.ttl` is the contract, `model.py` must satisfy it, `tests/` proves it.**
Without the third the registry rots quietly: a wrapper claims `EvaluateDensity`
because the source paper describes a density, nobody checks, the compiler selects
MCMC, and the failure appears as poor mixing rather than as an error.

So before trusting the registry:

```bash
python -m src.conformance.check
```

It calls every declared capability on every asset and reports three things:

- **OVERCLAIM** — declared but missing, or present and raising. Correctness bug.
- **underclaim** — implemented but not declared. Harmless, but the compiler will
  never choose it.
- **BROKEN** — the asset would not load.

It checks the **contract against the wrapper**, and nothing else. In particular it
says nothing about whether an asset has the *numbers* it needs: a wrapper whose
machinery is sound but whose parameter tables are absent conforms, and prints `ok`.
`hazard/gmpe` is the live example — it conforms, and not one of its 30 registered
model families can produce a prediction. Where an asset's data can be incomplete,
have it expose that separately and say so in its README; the gmpe asset uses
`coefficient_status()` and `missing_sources()`.

## Categories

| Category | Supplies |
|---|---|
| `hazard/` | `HazardEvent` → `LocalIntensity` |
| `fragility/` | `LocalIntensity`, `AssetHealth` → `AssetDamageState` |
| `deterioration/` | `AssetHealth` over time |
| `ds-to-functional/` | `AssetDamageState` → `AssetFunctionality` |
| `system/` | `AssetFunctionality` → `Service` |

`ds-to-functional/` is a first-class category on purpose. It is usually buried
inside a fragility implementation or a simulator, and that is exactly why damage
state vocabularies silently fail to line up between asset types. Giving it its
own assets with declared state spaces forces the mismatch into the open, where
the conformance check can see it.

## Naming

`fragility/hazus-roadway` and `fragility/hazus-bridge` provide the supplied Hazus
6.1 earthquake curves. They use distinct four-state and five-state damage spaces;
neither declares an AssetHealth parent. See [the guide](../docs/hazus61-fragility.md)
before connecting either to deterioration or damage-to-functionality assets.

The system registry includes `global-connectivity`, `travel-time`,
`population-accessibility`, and `staged-max-flow` adapters for the local ndtools
checkout. See [network simulators](../docs/network-simulators.md) for setup and
their shared `component-ratios-v1` input contract. `cap:serviceMetric` distinguishes
these meanings of Service so the pilot selects an appropriate factor.

Folder names are free — wrappers load by file path, not module name, so hyphens
are fine. Keep `cap:assetId` equal to `<category>/<folder>`.

Folders named `_pending` hold a `SOURCE.md` stating what is still needed. Replace
them with real assets as the sources land; do not fill them with placeholder
numbers.
