"""Population share with access to a destination -- the decision metric.

Deterministic weighted mean over the per-origin service states. It is a node
rather than a post-processing step so the network runs all the way to the
quantity the decision is actually measured in, which is what the ontology's
`Impact` layer asks for.

Packages with no population weights (Hebei) have nothing for this node to
average, so the model simply omits it rather than inventing uniform weights.
"""

from __future__ import annotations

from typing import Sequence

import torch


class PopulationAccess:
    """Continuous share in [0, 1]. Parents = every origin's service variable."""

    def __init__(self, childs, parents, weights: Sequence[float], device: str = "cpu"):
        assert len(childs) == 1, "PopulationAccess defines exactly one variable"
        assert len(parents) == len(weights), "one weight per origin"
        total = float(sum(weights))
        assert total > 0, "population weights must sum to something positive"
        self.childs = childs
        self.parents = parents
        self.device = device
        self.weights = torch.tensor(
            [float(w) / total for w in weights], dtype=torch.float32, device=device
        )

    def _fraction(self, S: torch.Tensor) -> torch.Tensor:
        return (S.to(self.device).float() * self.weights).sum(dim=1)

    def sample(self, Cs_pars):
        Cs = self._fraction(Cs_pars)
        return Cs, torch.zeros(Cs.shape[0], device=self.device)

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        expected = self._fraction(Cs[:, 1:])
        consistent = torch.isclose(Cs[:, 0], expected, rtol=1e-6, atol=1e-9)
        zero = torch.zeros(Cs.shape[0], device=self.device)
        return torch.where(consistent, zero, torch.full_like(zero, -float("inf")))
