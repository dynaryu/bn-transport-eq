"""Bridge states -> edge state."""

from __future__ import annotations

import torch

from . import Probability


class EdgeOpen(Probability):
    """Open only if every bridge on the edge is usable; an edge with no bridge is open."""

    def form(self) -> str:
        rows = self.spec["input_columns"]
        if len(rows) != 1:
            return f"1 if every bridge on the edge is usable, else 0, for {len(rows)} edges"
        if not rows[0]["usable"]:
            return "1: the edge has no bridge"
        names = [self.parents[i].name for i in rows[0]["usable"]]
        return f"1 if {' and '.join(names)} {'is' if len(names) == 1 else 'are'} 1, else 0"

    def evaluate(self, pars: torch.Tensor) -> torch.Tensor:
        out = [(pars[:, r["usable"]] == 1).all(1) if r["usable"] else
               torch.ones(len(pars), dtype=torch.bool, device=self.device)
               for r in self.spec["input_columns"]]
        return torch.stack(out, 1).to(torch.float64)
