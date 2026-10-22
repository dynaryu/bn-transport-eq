"""Implementations named by the functions' `implementation` field: tbnpy probability objects.

Each is a class built as ``cls(childs, parents, spec, device=..., directory=...)``
for one binding: its ``childs`` are every output variable of the binding's rows,
its ``parents`` every input variable they read. ``spec`` is that binding resolved
by ``bn_builder``: the rows, each row's parameter values, which parent columns
each row's input ports read, the states of discrete ports, and whole tables.

The methods are the capabilities its function declares, named by
``load.CAPABILITY_METHOD``: ``sample(Cs_pars=None, n_sample=None)`` for
SampleForward, returning (childs, log probability), and ``log_prob(Cs)`` for
EvaluateLikelihood, with ``Cs`` the childs' columns then the parents'. A discrete
variable holds the index into its quantity's states. Everything is PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class Notes(list):
    """What a kernel wants a reader of the result to know. tbnpy deep-copies its probability
    objects before sampling; the copies keep appending to this same list."""

    def __deepcopy__(self, memo: dict) -> Notes:
        return self

    def add(self, text: str) -> None:
        if text not in self:
            self.append(text)


class Probability:
    """Shared plumbing. A deterministic subclass defines ``evaluate(pars)`` only."""

    def __init__(self, childs: list, parents: list, spec: dict[str, Any], device: str = "cpu",
                 directory: Path | str | None = None) -> None:
        self.childs = childs
        self.parents = parents
        self.spec = spec
        self.device = torch.device(device)
        self.directory = Path(directory) if directory else None
        self.notes = Notes()

    def tensor(self, values: Any) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float64, device=self.device)

    def column(self, attr: str) -> torch.Tensor:
        """One attribute across the rows, shaped (1, rows) to broadcast over samples."""
        return self.tensor([row[attr] for row in self.spec["rows"]])[None, :]

    def input(self, pars: torch.Tensor, port: str) -> torch.Tensor:
        """(n, rows) for a port with exactly one parent per row (via self or a foreign key)."""
        return torch.stack([pars[:, r[port][0]] for r in self.spec["input_columns"]], dim=1)

    def _pars(self, Cs_pars: torch.Tensor | None, n_sample: int | None) -> torch.Tensor:
        if Cs_pars is None:
            return torch.empty((n_sample, 0), dtype=torch.float64, device=self.device)
        return Cs_pars.to(device=self.device, dtype=torch.float64)

    def sample(self, Cs_pars: torch.Tensor | None = None, n_sample: int | None = None):
        pars = self._pars(Cs_pars, n_sample)
        return self.evaluate(pars), torch.zeros(len(pars), dtype=torch.float64, device=self.device)

    def log_prob(self, Cs: torch.Tensor) -> torch.Tensor:
        Cs = Cs.to(device=self.device, dtype=torch.float64)
        n_child = len(self.childs)
        expected = self.evaluate(Cs[:, n_child:])
        valid = torch.isclose(Cs[:, :n_child], expected, rtol=1e-9, atol=0.0, equal_nan=True).all(dim=1)
        return torch.where(valid, 0.0, -torch.inf).to(torch.float64)
