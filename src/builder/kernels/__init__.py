"""Implementations named by the functions' `implementation` field: tbnpy probability objects.

Each kernel is a class written like tbnpy's own examples: its constructor takes
``childs``, ``parents`` and the distribution's parameters by name, and its
``sample`` and ``log_prob`` follow tbnpy's protocol:

    sample(Cs_pars)  Cs_pars (N, n_parents)  ->  Cs (N, n_childs), logp (N,)
    log_prob(Cs)     Cs (N, n_childs + n_parents), childs first  ->  logp (N,)

A root (no parents) is sampled with ``sample(n_sample=N)`` instead. A discrete
variable holds the index into its ``Variable.values``. Everything is PyTorch.

The generated model.json is a tbnpy spec: each probability object is a
``custom`` node naming its class here, its childs and parents, and its
constructor's arguments (``kwargs``), all plain JSON. ``tbnpy.compile`` builds
the objects from it, so a kernel's constructor is the whole interface: what the
spec records is exactly what the constructor takes. The node's ``metadata``
(function, role, parameter sets) reaches the object as ``spec_metadata``.

Every kernel inherits ``tbnpy.Probability``, so it prints as the distribution it
defines, and ``print`` adds each variable's domain and the kernel's notes.
"""

from __future__ import annotations

from typing import Any

import torch
from tbnpy.probability import Probability as TbnpyProbability


class Notes(list):
    """What a kernel wants a reader of the result to know. tbnpy deep-copies its probability
    objects before sampling; the copies keep appending to this same list."""

    def __deepcopy__(self, memo: dict) -> Notes:
        return self

    def add(self, text: str) -> None:
        if text not in self:
            self.append(text)


class Probability(TbnpyProbability):
    """What every kernel shares.

    A deterministic kernel defines ``evaluate(Cs_pars)`` only, returning the childs
    (N, n_childs) as a function of the parents. ``sample`` then returns it with log
    probability 0, and ``log_prob`` is 0 where the childs equal it and -inf elsewhere.
    A stochastic kernel defines its own ``sample`` and ``log_prob``.
    """

    def __init__(self, childs: list, parents: list = (), device: str = "cpu") -> None:
        self.childs = list(childs)
        self.parents = list(parents)
        self.device = torch.device(device)
        self.spec_metadata: dict[str, Any] = {}      # tbnpy.compile sets it from the node's metadata
        self.notes = Notes()

    def tensor(self, values: Any) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float64, device=self.device)

    # ------------------------------------------------------------------
    # Readings (repr and print come from tbnpy.Probability)
    def form(self) -> str:
        return self.spec_metadata.get("function", "")

    def form_details(self) -> list[str]:
        return [f"note: {note}" for note in self.notes]

    def parameter_sets(self) -> str:
        """`` [set]`` naming the parameter set the object was built from, for the end of a form; else empty."""
        name = self.spec_metadata.get("parameter_set")
        return f" [{name}]" if name else ""

    # ------------------------------------------------------------------
    # Deterministic kernels
    def evaluate(self, Cs_pars: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} defines sample and log_prob itself")

    def _parent_columns(self, Cs_pars: torch.Tensor | None, n_sample: int | None) -> torch.Tensor:
        """The parent columns as float64 on this device; (n_sample, 0) for a root."""
        if Cs_pars is None:
            return torch.empty((n_sample, 0), dtype=torch.float64, device=self.device)
        return Cs_pars.to(device=self.device, dtype=torch.float64)

    def sample(self, Cs_pars: torch.Tensor | None = None, n_sample: int | None = None):
        Cs_pars = self._parent_columns(Cs_pars, n_sample)
        Cs = self.evaluate(Cs_pars)
        return Cs, torch.zeros(len(Cs_pars), dtype=torch.float64, device=self.device)

    def log_prob(self, Cs: torch.Tensor) -> torch.Tensor:
        Cs = Cs.to(device=self.device, dtype=torch.float64)
        n_childs = len(self.childs)
        expected = self.evaluate(Cs[:, n_childs:])
        valid = torch.isclose(Cs[:, :n_childs], expected, rtol=1e-9, atol=0.0, equal_nan=True).all(dim=1)
        return torch.where(valid, 0.0, -torch.inf).to(torch.float64)
