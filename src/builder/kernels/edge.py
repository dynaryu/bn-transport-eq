"""Bridge states -> edge state."""

from __future__ import annotations

import numpy as np

from . import Call


class EdgeOpen:
    """Open only if every bridge on the edge is usable; an edge with no bridge is open."""

    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        out = np.ones((call.n, len(call.rows)), dtype=int)
        for j, usable in enumerate(call.inputs["usable"]):      # (n, k) per edge; k may be 0
            if usable.shape[1]:
                out[:, j] = np.all(usable == 1, axis=1)
        return {"open": out}
