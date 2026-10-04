"""Usability of one bridge, given the ground motion at its site.

    P(X = 0 | SA1) = Phi( (ln SA1 - ln median) / beta )

the Hazus lognormal fragility evaluated at the Extensive limit state, with the
pilot's closure policy folded in: Extensive or Complete is unusable.

States are ordered worst to best, as tbnpy expects, so index 0 is `unusable`
and index 1 is `usable`.

This node carries the FRAGILITY dispersion only. The ground-motion dispersion is
in the parent. Their convolution reproduces the single combined sigma the
analytic engine uses, which is the equivalence the tests check.
"""

from __future__ import annotations

import math

import torch

UNUSABLE, USABLE = 0, 1


class X:
    """Binary usability. Parents = [SA1 at this bridge]. Child is 0/1."""

    def __init__(self, childs, parents, median: float, beta: float, device: str = "cpu"):
        assert len(childs) == 1, "X defines exactly one variable"
        assert len(parents) == 1, "X parents = [SA1::<edge>]"
        self.childs = childs
        self.parents = parents
        self.device = device
        self.median = float(median)
        self.beta = float(beta)
        assert self.median > 0 and self.beta > 0, "Hazus curve needs a positive median and beta"
        self._ln_median = math.log(self.median)

    def p_unusable(self, sa1: torch.Tensor) -> torch.Tensor:
        """Phi((ln sa - ln median) / beta), with zero intensity meaning no damage."""
        sa1 = sa1.to(self.device)
        positive = sa1 > 0
        ln_sa = torch.log(torch.where(positive, sa1, torch.ones_like(sa1)))
        z = (ln_sa - self._ln_median) / self.beta
        normal = torch.distributions.Normal(
            torch.zeros((), device=self.device), torch.ones((), device=self.device)
        )
        return torch.where(positive, normal.cdf(z), torch.zeros_like(sa1))

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        p_fail = self.p_unusable(Cs_pars[:, 0])
        failed = torch.rand(p_fail.shape, device=self.device) < p_fail
        Cs = torch.where(failed, torch.zeros_like(p_fail), torch.ones_like(p_fail)).long()
        logp = torch.log(torch.where(failed, p_fail, 1.0 - p_fail).clamp_min(1e-300))
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        x = Cs[:, 0].long()
        p_fail = self.p_unusable(Cs[:, 1])
        p = torch.where(x == UNUSABLE, p_fail, 1.0 - p_fail)
        # Out of support -> -inf, never an error.
        in_support = (x == UNUSABLE) | (x == USABLE)
        return torch.where(
            in_support, torch.log(p.clamp_min(1e-300)), torch.full_like(p, -float("inf"))
        )
