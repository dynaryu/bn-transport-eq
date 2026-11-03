"""Fragility kernels: intensity -> damage state."""

from __future__ import annotations

import torch

from . import Probability


def lognormal(im: torch.Tensor, medians: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    """P(DS >= k | im) = Phi(ln(im / median_k) / beta_k) for each limit state k; 0 at im = 0.

    im : (N,);  medians, betas : (K-1,), one per damage state after the first
    Returns (N, K-1)
    """
    return torch.special.ndtr((torch.log(im)[:, None] - torch.log(medians)) / betas)


# Functional forms, by the parameter set's "form". A new curve with an existing form is only
# a parameter set; a new form is a function here plus its branch in schema/functions/fragility.bridge.json.
FRAGILITY_FORMS = {
    "lognormal": lognormal,
}


class Fragility(Probability):
    """Damage state DS of one bridge given the intensity IM at its site.

        P(DS >= k | IM) = Phi(ln(IM / median_k) / beta_k),    k = each damage state after the first
        P(DS = k | IM)  = P(DS >= k | IM) - P(DS >= k+1 | IM)
    """

    def __init__(self, childs, parents, medians, betas, curve="lognormal", device="cpu"):
        """
        childs  : [Variable DS], values = the damage states, least damage first
                  (e.g. none, slight, moderate, extensive, complete)
        parents : [Variable IM], e.g. SA(1.0) in g
        medians : median IM of each damage state after the first, in that order
        betas   : log standard deviation of each of those states, or one number for all
        curve   : the functional form, a key of FRAGILITY_FORMS
        """
        super().__init__(childs, parents, device)
        assert len(childs) == 1 and len(parents) == 1, "Fragility: one DS child, one IM parent"
        if curve not in FRAGILITY_FORMS:
            raise ValueError(f"fragility form {curve!r} is not implemented; have {sorted(FRAGILITY_FORMS)}")

        self.states = list(childs[0].values)
        n_limit = len(self.states) - 1
        self.curve = curve
        self.medians = self.tensor(medians)                                  # (K-1,)
        self.betas = self.tensor(betas).expand(n_limit).clone()             # (K-1,)
        if self.medians.shape != (n_limit,):
            raise ValueError(f"{n_limit} medians needed, one per state after {self.states[0]!r}")
        if (torch.diff(self.medians) < 0).any():
            raise ValueError(f"medians must not decrease with damage state, got {self.medians.tolist()}")

    # ------------------------------------------------------------------
    def form(self):
        im = self.parents[0].name
        medians = ", ".join(f"{s} {m:g}" for s, m in zip(self.states[1:], self.medians.tolist()))
        betas = self.betas.tolist()
        beta = (f"{betas[0]:g}" if len(set(betas)) == 1 else
                ", ".join(f"{s} {b:g}" for s, b in zip(self.states[1:], betas)))
        return (f"P(DS >= k) = Phi(ln({im} / median_k) / beta_k): medians {medians}; "
                f"beta {beta}{self.parameter_sets()}")

    # ------------------------------------------------------------------
    def distribution(self, im: torch.Tensor) -> torch.Tensor:
        """P(DS = k | im) for each damage state k (the EvaluateDensity capability).

        im : (N,) intensity
        Returns (N, K)
        """
        if (im < 0).any() or not torch.isfinite(im).all():
            raise ValueError(f"{self.parents[0].name} must be finite and nonnegative")
        exceed = FRAGILITY_FORMS[self.curve](im, self.medians, self.betas)     # (N, K-1): P(DS >= k)

        # Curves with different betas cross somewhere; a worse state is never more likely than a milder one.
        capped = torch.cummin(exceed, dim=1).values
        if (capped < exceed).any():
            self.notes.add("fragility curves cross; P(DS >= k) capped at P(DS >= k-1) where they do")

        ones = torch.ones_like(capped[:, :1])
        zeros = torch.zeros_like(ones)
        return (torch.cat([ones, capped], 1) - torch.cat([capped, zeros], 1)).clamp_min(0)

    def sample(self, Cs_pars):
        """
        Cs_pars : (N, 1) IM

        Returns
        -------
        Cs   : (N, 1) DS, the damage state index
        logp : (N,) log P(DS | IM)
        """
        im = Cs_pars.to(device=self.device, dtype=torch.float64)[:, 0]
        pmf = self.distribution(im)                                          # (N, K)

        # One uniform per sample: DS is the first state whose cumulative probability exceeds it.
        u = torch.rand((len(im), 1), dtype=torch.float64, device=self.device)
        ds = (u >= pmf.cumsum(1)[:, :-1]).sum(1)                             # (N,)
        logp = torch.log(pmf.gather(1, ds[:, None])[:, 0])
        return ds[:, None].to(torch.float64), logp

    def log_prob(self, Cs):
        """
        Cs : (N, 2)
            Cs[:, 0] = DS index
            Cs[:, 1] = IM

        Returns (N,) log P(DS | IM); -inf where DS is not a state index
        """
        Cs = Cs.to(device=self.device, dtype=torch.float64)
        ds, im = Cs[:, 0], Cs[:, 1]
        pmf = self.distribution(im)
        n_states = pmf.shape[1]

        valid = (ds == ds.floor()) & (ds >= 0) & (ds < n_states)
        index = ds.nan_to_num().long().clamp(0, n_states - 1)
        logp = torch.log(pmf.gather(1, index[:, None])[:, 0])
        return torch.where(valid, logp, -torch.inf)
