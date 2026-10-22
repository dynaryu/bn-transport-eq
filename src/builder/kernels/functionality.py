"""Damage state -> component functionality."""

from __future__ import annotations

from typing import Any

import torch

from . import Probability


class UsableFromDamage(Probability):
    """1 if the bridge's damage state is one its policy lists as usable, else 0."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        states = self.spec["states"]["damage"]
        self.policy = self.tensor([[float(s in p["usable_states"]) for s in states]
                                   for p in self.spec["parameters"]])          # (rows, K)

    def evaluate(self, pars: torch.Tensor) -> torch.Tensor:
        damage = self.input(pars, "damage")
        if ((damage != damage.floor()) | (damage < 0) | (damage >= self.policy.shape[1])).any():
            raise ValueError("invalid damage state index")
        return self.policy[torch.arange(len(self.policy), device=self.device)[None, :], damage.long()]
