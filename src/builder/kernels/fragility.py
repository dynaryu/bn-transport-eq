"""Fragility kernels: intensity -> damage state."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from . import Call


class LognormalFragility:
    """P(DS >= k | im) = Phi(ln(im / median_k) / beta) over the output's states after the first."""

    @staticmethod
    def _exceedance(call: Call) -> np.ndarray:
        states = call.quantities["damage"]["states"]
        p = call.params
        medians = np.asarray([p["medians"][s] for s in states[1:]], float)
        if np.any(np.diff(medians) < 0):
            raise ValueError(f"medians must not decrease with damage state, got {medians.tolist()}")
        im = np.asarray(call.inputs["im"], float)
        if np.any(im < 0) or np.any(~np.isfinite(im)):
            raise ValueError(f"{p['imt']} must be finite and nonnegative")
        with np.errstate(divide="ignore"):
            z = (np.log(im)[..., None] - np.log(medians)) / p["beta"]
        return ndtr(z)                               # (n, rows, K-1); 0 at im = 0

    def evaluate_density(self, call: Call) -> dict[str, np.ndarray]:
        """P(DS = k | im) as (n, rows, K)."""
        q = self._exceedance(call)
        ones = np.ones(q.shape[:-1] + (1,))
        zeros = np.zeros_like(ones)
        return {"damage": np.concatenate([ones, q], -1) - np.concatenate([q, zeros], -1)}

    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        # One uniform per bridge and draw: DS = number of limit states it falls under.
        q = self._exceedance(call)
        u = call.rng.random(q.shape[:-1] + (1,))
        return {"damage": (u < q).sum(-1)}
