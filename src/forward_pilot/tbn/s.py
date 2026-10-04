"""Service state of every origin, given which bridges are usable.

Deterministic: once the bridge states are fixed, reachability is settled. All the
uncertainty lives upstream, so this is a degenerate kernel and must not be
represented as a near-degenerate table.

**One node with many children, on purpose.** A separate node per origin would
re-run the same reachability pass once per origin -- 72 identical graph searches
over the same 129 bridge states. One search answers every origin at once, so the
whole set is one node whose `childs` are the per-origin service variables.

That departs from tbnpy's usual one-key-per-variable convention, which matters
only for `sample_evidence` matching evidence columns to node keys; forward
sampling is unaffected. Worth knowing before attaching evidence to this node.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ..engine import _reachable_nodes

CUT_OFF, CONNECTED = 0, 1


class S:
    """Per-origin binary service. Parents = every X; childs = one var per origin."""

    def __init__(
        self,
        childs,
        parents,
        network: Mapping[str, Any],
        edge_order: Sequence[str],
        origins: Sequence[str],
        destinations: Sequence[str],
        device: str = "cpu",
    ):
        assert len(childs) == len(origins), "one child variable per origin"
        assert len(parents) == len(edge_order), "one parent per bridge"
        self.childs = childs
        self.parents = parents
        self.device = device
        self.network = network
        self.edge_items = [(e, network["edges"][e]) for e in edge_order]
        self.origins = list(origins)
        self.destinations = list(destinations)
        self.directed = bool(network["directed"])

    def _service(self, X: torch.Tensor) -> torch.Tensor:
        """(n, n_edges) usability -> (n, n_origins) service.

        Reuses the engine's own reachability so the two backends cannot drift
        apart: the question "did this change the answer?" stays answerable.
        """
        X = X.to(self.device).long()
        usable = X.detach().cpu().numpy().astype(bool)
        out = torch.empty((X.shape[0], len(self.origins)), dtype=torch.long, device=self.device)
        for i, row in enumerate(usable):
            reached = _reachable_nodes(
                self.network["nodes"], self.edge_items, row, self.destinations, self.directed
            )
            out[i] = torch.tensor(
                [CONNECTED if o in reached else CUT_OFF for o in self.origins],
                dtype=torch.long, device=self.device,
            )
        return out

    def sample(self, Cs_pars):
        Cs = self._service(Cs_pars)
        return Cs, torch.zeros(Cs.shape[0], device=self.device)

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        n_childs = len(self.childs)
        claimed = Cs[:, :n_childs].long()
        expected = self._service(Cs[:, n_childs:])
        consistent = (claimed == expected).all(dim=1)
        zero = torch.zeros(Cs.shape[0], device=self.device)
        return torch.where(consistent, zero, torch.full_like(zero, -float("inf")))
