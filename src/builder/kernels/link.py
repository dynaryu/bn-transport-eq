"""Component states -> link state."""

from __future__ import annotations

import numpy as np

from . import Call


class Series:
    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        out = np.ones((call.n, len(call.rows)), dtype=int)
        for j, usable in enumerate(call.inputs["usable"]):      # (n, k) per link; k may be 0
            if usable.shape[1]:
                out[:, j] = np.all(usable == 1, axis=1)
        return {"open": out}
