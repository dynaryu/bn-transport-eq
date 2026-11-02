"""Fragility kernels: intensity -> damage state."""

from __future__ import annotations

from typing import Any

import torch

from . import Probability


def lognormal(im: torch.Tensor, limit_states: list[str], params: list[dict[str, Any]]) -> torch.Tensor:
    """P(DS >= k | im) = Phi(ln(im / median_k) / beta_k), (n, rows, K-1); 0 at im = 0.

    ``params`` is each row's ``parameters``: medians, and one beta or a beta per limit state.
    """
    medians = torch.as_tensor([[p["medians"][s] for s in limit_states] for p in params], dtype=torch.float64)
    if (torch.diff(medians, dim=1) < 0).any():
        raise ValueError(f"medians must not decrease with damage state, got {medians.tolist()}")
    betas = torch.as_tensor([[p["betas"][s] if "betas" in p else p["beta"] for s in limit_states] for p in params],
                            dtype=torch.float64)
    medians, betas = medians.to(im.device), betas.to(im.device)
    return torch.special.ndtr((torch.log(im)[..., None] - torch.log(medians)) / betas)


# Functional forms, by the parameter set's "form". A new curve with an existing form is only
# a parameter set; a new form is a function here plus its branch in schema/functions/fragility.bridge.json.
FRAGILITY_FORMS = {
    "lognormal": lognormal,
}


class Fragility(Probability):
    """Damage state at each bridge, for whichever curve each row's parameter set is."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        forms = {p["form"] for p in self.spec["parameters"]}
        if len(forms) != 1 or not forms <= FRAGILITY_FORMS.keys():
            raise ValueError(f"one implemented fragility form per binding, got {sorted(forms)}")
        self.curve_form = forms.pop()
        self.curve = FRAGILITY_FORMS[self.curve_form]

    def form(self) -> str:
        params = [p["parameters"] for p in self.spec["parameters"]]
        im = self.parents[0].name if len(self.parents) == 1 else self.spec["parameters"][0]["imt"]
        head = f"{self.curve_form}, P(DS >= k) = Phi(ln({im} / median_k) / beta_k)"
        if any(p != params[0] for p in params):
            return f"{head} over {len(params)} rows{self.parameter_sets()}"
        medians = ", ".join(f"{s} {m:g}" for s, m in params[0]["medians"].items())
        betas = params[0].get("betas")
        beta = (", ".join(f"{s} {b:g}" for s, b in betas.items()) if betas else f"{params[0]['beta']:g}")
        return f"{head}: medians {medians}; beta {beta}{self.parameter_sets()}"

    def distribution(self, pars: torch.Tensor) -> torch.Tensor:
        """P(DS = k | im) as (n, rows, K)."""
        im = self.input(pars, "im")
        if (im < 0).any() or not torch.isfinite(im).all():
            raise ValueError(f"{self.spec['parameters'][0]['imt']} must be finite and nonnegative")
        q = self.curve(im, self.spec["states"]["damage"][1:], [p["parameters"] for p in self.spec["parameters"]])
        # Curves with different betas cross somewhere; a worse state is never more likely than a milder one.
        capped = torch.cummin(q, dim=-1).values
        crossed = (capped < q).any(-1)
        if crossed.any():
            self.notes.add(f"fragility curves cross at {int(crossed.any(0).sum())} bridge(s); "
                           "P(DS >= k) capped at P(DS >= k-1) there")
        ones = torch.ones_like(capped[..., :1])
        return (torch.cat([ones, capped], -1) - torch.cat([capped, torch.zeros_like(ones)], -1)).clamp_min(0)

    def sample(self, Cs_pars=None, n_sample=None):
        mass = self.distribution(self._pars(Cs_pars, n_sample))
        # One uniform per bridge and draw: DS = the first state whose cumulative mass exceeds it.
        u = torch.rand(mass.shape[:2] + (1,), dtype=torch.float64, device=self.device)
        child = (u >= mass.cumsum(-1)[..., :-1]).sum(-1)
        logp = torch.log(mass.gather(-1, child[..., None]).squeeze(-1)).sum(1)
        return child.to(torch.float64), logp

    def log_prob(self, Cs: torch.Tensor) -> torch.Tensor:
        Cs = Cs.to(device=self.device, dtype=torch.float64)
        child = Cs[:, :len(self.childs)]
        mass = self.distribution(Cs[:, len(self.childs):])
        valid = ((child == child.floor()) & (child >= 0) & (child < mass.shape[-1])).all(1)
        index = child.nan_to_num().long().clamp(0, mass.shape[-1] - 1)
        logp = torch.log(mass.gather(-1, index[..., None]).squeeze(-1)).sum(1)
        return torch.where(valid, logp, -torch.inf)
