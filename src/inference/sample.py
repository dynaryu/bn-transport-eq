"""Forward Monte Carlo over a compiled assembly.

Bindings run in the compiled order. Within one, rows sharing a parameter set
are handed to the kernel together, so the 129 bridges of three Hazus classes
are three vectorised calls, not 129. This is the reference semantics of the
graph: a tbnpy or MBN backend built from the same compiled nodes should agree
with it in distribution.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.builder.compile import Compiled, node_id
from src.builder.kernels import Call
from src.builder.load import via_kind


def forward_sample(compiled: Compiled, n: int, seed: int | None = None,
                   bindings: set[int] | None = None, warnings: list[str] | None = None) -> dict[str, np.ndarray]:
    """{node id: (n,) samples}. Discrete nodes hold the index into their quantity's states.

    ``bindings`` restricts the run to those bindings, which must be closed under
    ``upstream`` (``Compiled.ancestors`` gives such a set): a query about travel
    time need not pay for max flow. ``warnings``, if given, collects what the
    kernels report, such as which ShakeMap version supplied the shaking.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    lib, network = compiled.library, compiled.network
    rng = np.random.default_rng(seed)
    store: dict[str, dict[str, np.ndarray]] = defaultdict(dict)    # quantity -> row id -> (n,)

    for index in compiled.order:
        if bindings is not None and index not in bindings:
            continue
        b = compiled.bindings[index]
        fn = b.function
        kernel = lib.kernel(fn["id"])
        tables = {name: network.rows(name) for name in fn.get("tables", [])}

        groups: dict[Any, list[int]] = defaultdict(list)
        for r, set_id in enumerate(b.parameter_sets):
            groups[set_id].append(r)

        for positions in groups.values():
            rows = [b.rows[r] for r in positions]
            # One parameter set per group, so one resolution of "{imt}"-style ports for all its rows.
            resolved = b.quantities[positions[0]]
            quantities = {name: lib.quantities[q] for name, q in resolved.items()}
            inputs: dict[str, Any] = {}
            parents: dict[str, list[list[str]]] = {}
            for port in fn["inputs"]:
                name, values = port["name"], store[resolved[port["name"]]]
                pids = [b.parents[name][r] for r in positions]
                parents[name] = pids
                if via_kind(port["via"]) in ("self", "fk"):
                    inputs[name] = np.stack([values[p[0]] for p in pids], axis=1)
                else:
                    stacked: dict[tuple[str, ...], np.ndarray] = {}
                    for p in pids:
                        if tuple(p) not in stacked:
                            stacked[tuple(p)] = (np.stack([values[x] for x in p], axis=1)
                                                 if p else np.empty((n, 0), dtype=int))
                    inputs[name] = [stacked[tuple(p)] for p in pids]

            call = Call(n, rows, inputs, parents, b.parameters[positions[0]], tables, quantities, rng)
            result = kernel.sample_forward(call)
            if warnings is not None:
                warnings.extend(w for w in call.warnings if w not in warnings)
            for port in fn["outputs"]:
                out = np.asarray(result[port["name"]])
                if out.shape[:2] != (n, len(rows)):
                    raise ValueError(f"{fn['id']} returned {port['name']} with shape {out.shape}, "
                                     f"expected ({n}, {len(rows)})")
                for j, row in enumerate(rows):
                    store[resolved[port["name"]]][row["id"]] = out[:, j]

    return {node_id(q, r): v for q, by_row in store.items() for r, v in by_row.items()}


def summarise(quantity: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    """State probabilities for a discrete quantity, moments for a continuous one."""
    if quantity["kind"] == "discrete":
        counts = np.bincount(np.asarray(x, int), minlength=len(quantity["states"])) / len(x)
        return {"P": dict(zip(map(str, quantity["states"]), counts.round(6).tolist()))}
    if quantity["kind"] == "continuous":
        finite = x[np.isfinite(x)]
        stats: dict[str, Any] = {"P_inf": float(np.mean(~np.isfinite(x)))}
        if finite.size:
            p5, p50, p95 = np.percentile(finite, [5, 50, 95])
            stats.update(mean_finite=float(finite.mean()), p5=float(p5), p50=float(p50), p95=float(p95))
        return stats
    return {f: float(np.mean(x[f])) for f in quantity["fields"]}
