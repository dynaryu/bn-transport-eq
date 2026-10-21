"""Damage state -> component functionality."""

from __future__ import annotations

import numpy as np

from . import Call


class UsableFromDamage:
    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        states = call.quantities["damage"]["states"]
        usable = np.isin(states, call.params["usable_states"]).astype(int)   # 1 open, 0 closed
        return {"usable": usable[np.asarray(call.inputs["damage"], int)]}
