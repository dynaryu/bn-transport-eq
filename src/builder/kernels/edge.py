"""Bridge damage -> edge state."""

from __future__ import annotations

import torch

from . import Probability


class EdgeOpen(Probability):
    """Whether an edge is open, given the damage state of each bridge on it (deterministic).

        EdgeOpen = True if every bridge on the edge is in one of usable_states, else False;
                   True for an edge with no bridge
    """

    def __init__(self, childs, parents, usable_states, device="cpu"):
        """
        childs        : [Variable EdgeOpen], values ["False", "True"] (state indices 0, 1)
        parents       : [Variable DS] of each bridge on the edge, values = the damage states;
                        none for an edge without a bridge
        usable_states : the damage states (labels) in which a bridge stays open: the closure policy
        """
        super().__init__(childs, parents, device)
        assert len(childs) == 1, "EdgeOpen: one EdgeOpen child"
        self.usable_states = list(usable_states)

        states = list(parents[0].values) if parents else []
        if any(list(p.values) != states for p in parents):
            raise ValueError("every bridge on an edge must be on the same damage scale")
        unknown = sorted(set(self.usable_states) - set(states)) if parents else []
        if unknown:
            raise ValueError(f"usable states {unknown} are not damage states: {states}")
        # Open (1) or not (0) for each DS index: the lookup table evaluate() reads.
        self.open_if = self.tensor([float(s in self.usable_states) for s in states])      # (K,)

    def form(self):
        if not self.parents:
            return "True: the edge has no bridge"
        names = " and ".join(p.name for p in self.parents)
        return (f"True if {names} {'is' if len(self.parents) == 1 else 'are'} "
                f"{', '.join(self.usable_states)}, else False{self.parameter_sets()}")

    def evaluate(self, Cs_pars):
        """
        Cs_pars : (N, n_bridges) DS index of each bridge on the edge (N, 0 for none)
        Returns (N, 1) EdgeOpen index: 1 (True) open, 0 (False) closed
        """
        if not self.parents:
            return torch.ones((len(Cs_pars), 1), dtype=torch.float64, device=self.device)
        ds = Cs_pars
        if ((ds != ds.floor()) | (ds < 0) | (ds >= len(self.open_if))).any():
            raise ValueError("invalid damage state index")
        return self.open_if[ds.long()].prod(dim=1, keepdim=True)
