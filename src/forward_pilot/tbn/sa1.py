"""Spectral acceleration at 1 s at one bridge site, given the earthquake scenario.

A root node: the scenario is an input the user sets, not an uncertain quantity,
so it parameterises this node rather than being a variable of its own.

    SA1 ~ LogNormal(ln_median, sigma)

``sigma`` is the ground-motion dispersion ONLY. The fragility dispersion belongs
to the child node, which is the point of splitting them: the previous engine
combined the two analytically and sampled usability directly, which gives the
same marginal but leaves nowhere to attach a correlated residual. Here the
residual would enter this node, which is what makes the spatial-correlation
extension point real rather than aspirational.

A degenerate case is legal and needed: `uniform_sa1` applies one fixed intensity
to every site, so sigma is zero and the node becomes deterministic.
"""

from __future__ import annotations

import math

import torch


class SA1:
    """Ground motion at one site. Root node; child is continuous, in g."""

    def __init__(self, childs, parents, ln_median: float, sigma: float, device: str = "cpu"):
        assert len(childs) == 1, "SA1 defines exactly one variable"
        assert not parents, "SA1 is a root: the scenario parameterises it"
        self.childs = childs
        self.parents = []
        self.device = device
        self.ln_median = float(ln_median)
        self.sigma = float(sigma)
        assert self.sigma >= 0.0, "ground-motion dispersion cannot be negative"

    @property
    def deterministic(self) -> bool:
        return self.sigma == 0.0

    def _dist(self, n: int):
        mean = torch.full((n,), self.ln_median, device=self.device)
        return torch.distributions.Normal(mean, torch.full_like(mean, self.sigma))

    def sample(self, n_sample):
        n = int(n_sample)
        if self.deterministic:
            Cs = torch.full((n,), math.exp(self.ln_median), device=self.device)
            return Cs, torch.zeros(n, device=self.device)
        ln_sa = self._dist(n).sample()
        # log-density of SA1 itself, not of ln SA1: the Jacobian term is -ln(sa).
        logp = self._dist(n).log_prob(ln_sa) - ln_sa
        return torch.exp(ln_sa), logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        sa = Cs[:, 0] if Cs.dim() > 1 else Cs
        n = sa.shape[0]
        if self.deterministic:
            target = math.exp(self.ln_median)
            close = torch.isclose(sa, torch.full_like(sa, target), rtol=1e-9, atol=1e-12)
            return torch.where(close, torch.zeros_like(sa), torch.full_like(sa, -float("inf")))
        # Outside the support, return -inf rather than raising on log(0).
        valid = sa > 0
        ln_sa = torch.log(torch.where(valid, sa, torch.ones_like(sa)))
        logp = self._dist(n).log_prob(ln_sa) - ln_sa
        return torch.where(valid, logp, torch.full_like(sa, -float("inf")))
