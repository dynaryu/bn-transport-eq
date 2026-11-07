"""Observation kernels: intensity -> a measurement of it."""

from __future__ import annotations

import torch
from torch.distributions import Normal

from . import Probability


class ObservedIntensity(Probability):
    """A measurement of the intensity IM at one bridge, with an error proportional to it.

        IM_obs ~ N(IM, (relative_sd * IM)^2)
    """

    def __init__(self, childs, parents, relative_sd, device="cpu"):
        """
        childs      : [Variable IM_obs], in the IM's unit
        parents     : [Variable IM], e.g. SA(1.0) in g
        relative_sd : epsilon, the standard deviation as a fraction of IM
        """
        super().__init__(childs, parents, device)
        assert len(childs) == 1 and len(parents) == 1, "ObservedIntensity: one observed child, one IM parent"
        if relative_sd <= 0:
            raise ValueError(f"relative_sd must be positive, got {relative_sd}")
        self.relative_sd = float(relative_sd)

    # ------------------------------------------------------------------
    def form(self):
        return (f"{self.childs[0].name} ~ N({self.parents[0].name}, ({self.relative_sd:g} * "
                f"{self.parents[0].name})^2){self.parameter_sets()}")

    # ------------------------------------------------------------------
    def distribution(self, im: torch.Tensor) -> Normal:
        """IM_obs given im (N,): Normal of shape (N,) (the EvaluateDensity capability)."""
        if (im <= 0).any() or not torch.isfinite(im).all():
            raise ValueError(f"{self.parents[0].name} must be finite and positive")
        return Normal(im, self.relative_sd * im)

    def sample(self, Cs_pars):
        """
        Cs_pars : (N, 1) IM

        Returns
        -------
        Cs   : (N, 1) IM_obs
        logp : (N,) log density of IM_obs given IM
        """
        dist = self.distribution(Cs_pars.to(device=self.device, dtype=torch.float64)[:, 0])
        obs = dist.sample()
        return obs[:, None], dist.log_prob(obs)

    def log_prob(self, Cs):
        """
        Cs : (N, 2)
            Cs[:, 0] = IM_obs
            Cs[:, 1] = IM

        Returns (N,) log density of IM_obs given IM
        """
        Cs = Cs.to(device=self.device, dtype=torch.float64)
        return self.distribution(Cs[:, 1]).log_prob(Cs[:, 0])
