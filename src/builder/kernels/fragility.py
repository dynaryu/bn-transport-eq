"""Fragility kernels: intensity -> damage state."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr

from . import Call


def lognormal(im: np.ndarray, limit_states: list[str], p: dict[str, Any]) -> np.ndarray:
    """P(DS >= k | im) = Phi(ln(im / median_k) / beta_k), (n, rows, K-1); 0 at im = 0. ``p`` is the set's ``parameters``."""
    medians = np.asarray([p["medians"][s] for s in limit_states], float)
    if np.any(np.diff(medians) < 0):
        raise ValueError(f"medians must not decrease with damage state, got {medians.tolist()}")
    betas = np.asarray([p["betas"][s] for s in limit_states] if "betas" in p else p["beta"], float)
    with np.errstate(divide="ignore"):
        return ndtr((np.log(im)[..., None] - np.log(medians)) / betas)


# Functional forms, by the parameter set's "form". A new curve with an existing form is only
# a parameter set; a new form is a function here plus its branch in schema/functions/fragility.bridge.json.
FRAGILITY_FORMS = {
    "lognormal": lognormal,
}


class Fragility:
    """Damage state at each bridge, for whichever curve the parameter set is."""

    @staticmethod
    def _exceedance(call: Call) -> np.ndarray:
        p = call.params
        im = np.asarray(call.inputs["im"], float)
        if np.any(im < 0) or np.any(~np.isfinite(im)):
            raise ValueError(f"{p['imt']} must be finite and nonnegative")
        q = FRAGILITY_FORMS[p["form"]](im, call.quantities["damage"]["states"][1:], p["parameters"])
        # Curves with different betas cross somewhere; a worse state is never more likely than a milder one.
        capped = np.minimum.accumulate(q, axis=-1)
        crossed = np.any(capped < q, axis=-1)
        if crossed.any():
            call.warnings.append(f"fragility curves cross in {crossed.mean():.1%} of draws at "
                                 f"{crossed.any(0).sum()} bridge(s); P(DS >= k) capped at P(DS >= k-1)")
        return capped

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
