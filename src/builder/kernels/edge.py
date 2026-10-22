"""Bridge states -> edge state."""

from __future__ import annotations

import torch

from . import Probability


class EdgeOpen(Probability):
    """Open only if every bridge on the edge is usable; an edge with no bridge is open."""

    def evaluate(self, pars: torch.Tensor) -> torch.Tensor:
        out = [(pars[:, r["usable"]] == 1).all(1) if r["usable"] else
               torch.ones(len(pars), dtype=torch.bool, device=self.device)
               for r in self.spec["input_columns"]]
        return torch.stack(out, 1).to(torch.float64)
