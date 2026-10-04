"""s-t connectivity as a deterministic system operator.

Given per-link functionality, decide whether source still reaches target.
See README.md for the capability rationale and SOURCE.md for provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.wrappers.base import Asset

_PARAMS = Path(__file__).parent / "params" / "default.json"


class StConnectivity(Asset):
    asset_id = "system/st-connectivity"
    version = "0.1.0"
    target = ("Service",)
    given = ("AssetFunctionality",)
    deterministic = True

    def __init__(self, threshold: float | None = None) -> None:
        params = json.loads(_PARAMS.read_text())
        self.threshold = params["threshold"] if threshold is None else threshold

    # ---- core ------------------------------------------------------------

    @staticmethod
    def _connected(
        edges: Sequence[tuple[Any, Any]], usable: np.ndarray, source: Any, target: Any
    ) -> np.ndarray:
        """Union-find over usable links. ``usable`` is (n_samples, n_edges) boolean."""
        usable = np.atleast_2d(np.asarray(usable, dtype=bool))
        out = np.empty(usable.shape[0], dtype=bool)

        for i, row in enumerate(usable):
            parent: dict[Any, Any] = {}

            def find(x: Any) -> Any:
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for (u, v), live in zip(edges, row):
                if live:
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[ru] = rv
            out[i] = find(source) == find(target)
        return out

    def _usable(self, functionality: np.ndarray) -> np.ndarray:
        return np.asarray(functionality) >= self.threshold

    # ---- capabilities ----------------------------------------------------

    def sample_forward(
        self, given: Mapping[str, Any], n: int = 1, rng: np.random.Generator | None = None
    ) -> Mapping[str, np.ndarray]:
        """Deterministic, so ``n`` and ``rng`` are ignored beyond shape handling."""
        connected = self._connected(
            given["edges"], self._usable(given["AssetFunctionality"]),
            given["source"], given["target"],
        )
        return {"Service": connected.astype(int)}

    def evaluate_density(
        self, target: Mapping[str, Any], given: Mapping[str, Any]
    ) -> np.ndarray:
        """Degenerate kernel: 1 where the system state matches, else 0."""
        actual = self.sample_forward(given)["Service"]
        return (actual == np.asarray(target["Service"])).astype(float)

    def evaluate_likelihood(
        self, target: Mapping[str, Any], given: Mapping[str, Any]
    ) -> np.ndarray:
        """L(given) = 1{phi(given) == target}, read as a function of the parents.

        Same indicator as evaluate_density, but the intended reading differs: this
        is the backward message, unnormalised over ``given``. Kept as its own
        method so the distinction survives into the interface.
        """
        return self.evaluate_density(target, given)

    def batch_classify(self, given: Mapping[str, np.ndarray]) -> np.ndarray:
        return self._connected(
            given["edges"], self._usable(given["AssetFunctionality"]),
            given["source"], given["target"],
        ).astype(int)

    # ---- conformance -----------------------------------------------------

    def smoke_inputs(self) -> dict[str, Any]:
        edges = [("s", "a"), ("a", "t"), ("s", "b"), ("b", "t")]
        return {
            "given": {
                "edges": edges,
                "source": "s",
                "target": "t",
                "AssetFunctionality": np.array([[1.0, 1.0, 0.0, 0.0]]),
            },
            "target": {"Service": np.array([1])},
        }
