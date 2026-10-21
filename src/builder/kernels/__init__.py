"""Implementations named by the functions' `implementation` field.

Each is a class whose methods are the capabilities its function declares, named
by ``load.CAPABILITY_METHOD``: ``sample_forward(call)`` for SampleForward,
``evaluate_density(call)`` for EvaluateDensity. A method returns
``{output port name: array of shape (n_samples, n_rows)}``; a discrete output is
the index into its quantity's states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Call:
    """Everything one invocation sees: the rows sharing one parameter set, and their inputs.

    ``inputs[name]`` is an (n, rows) array when each row has exactly one parent
    (via self or a foreign key), and a list with one (n, k) array per row when a
    row may have several (via a reverse key or ``*``). ``parents[name]`` names
    those parents' row ids in the same order.
    """

    n: int
    rows: list[dict[str, Any]]
    inputs: dict[str, Any]
    parents: dict[str, list[list[str]]]
    params: dict[str, Any] | None
    tables: dict[str, list[dict[str, Any]]]
    quantities: dict[str, dict[str, Any]]    # port name -> quantity record
    rng: np.random.Generator
    warnings: list[str] = field(default_factory=list)   # what the kernel wants a reader of the result to know

    def column(self, attr: str, dtype: Any = float) -> np.ndarray:
        """One attribute across this call's rows, shaped (1, rows) to broadcast over samples."""
        return np.asarray([row[attr] for row in self.rows], dtype=dtype)[None, :]
