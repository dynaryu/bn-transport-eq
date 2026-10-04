# s-t connectivity

Returns whether a source node still reaches a target node, given which links are
usable. The canonical binary system function for a transport or utility network.

## What it maps

`AssetFunctionality` for every link in scope → `Service` (binary connected).

A link is usable when its functionality is at or above `threshold` (default 0.5
on a 0–1 scale). That threshold is the entire content of the DS→usable mapping
for this asset, and is deliberately a parameter rather than a constant — see
`params/default.json`.

## Why it is deterministic

Given link states, connectivity is settled. All the uncertainty lives upstream in
`AssetFunctionality`. So its conditional kernel is degenerate, declared via
`cap:deterministic true`, and must not be represented as a near-degenerate
probability table.

## Capabilities, and why each holds

| Capability | Holds because |
|---|---|
| `SampleForward` | evaluate the indicator at the given link states |
| `EvaluateDensity` | degenerate: 1 if the state matches, else 0 |
| `EvaluateLikelihood` | same indicator read as a function of the parents — this is what makes backward inference from an observed outage possible |
| `BatchClassify` | vectorised over many sampled link-state vectors |

`NormalisingConstant` is **not** declared. Z depends on the upstream distribution
over link states, which this asset does not hold.

## Assumptions

- Undirected. Directed s-t connectivity would be a separate asset.
- Binary usability. Degraded-but-passable capacity needs `system/max-flow`.
- Node failures are not modelled; only links carry state.
