"""Uniform interface every knowledge asset implements.

An asset declares its capabilities in its ``asset.ttl`` or, for a single-file asset,
its ``<name>.asset.json``; this module defines what each declared capability obliges
the wrapper to provide. The conformance harness checks the two agree, because a
capability claimed but not implemented produces a wrong posterior rather than an
error.

Implement only the methods your asset genuinely supports. Leave the rest alone --
the base raises NotImplementedError, which the harness reports as a mismatch
against the declaration.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Capability names, matching cap: individuals in ontology/capabilities.ttl
EVALUATE_DENSITY = "EvaluateDensity"
SAMPLE_FORWARD = "SampleForward"
EVALUATE_LIKELIHOOD = "EvaluateLikelihood"
BATCH_CLASSIFY = "BatchClassify"
GRADIENT = "Gradient"
NORMALISING_CONSTANT = "NormalisingConstant"

CAPABILITY_METHOD = {
    EVALUATE_DENSITY: "evaluate_density",
    SAMPLE_FORWARD: "sample_forward",
    EVALUATE_LIKELIHOOD: "evaluate_likelihood",
    BATCH_CLASSIFY: "batch_classify",
    GRADIENT: "gradient",
    NORMALISING_CONSTANT: "normalising_constant",
}


class DataUnavailable(RuntimeError):
    """The asset is registered, but the numbers it needs are not in the repository.

    Raise this from construction rather than returning a placeholder. The conformance
    harness reads it as "this asset can do nothing yet", which conforms only where the
    declaration claims no capability either.
    """


class Asset(ABC):
    """One factor's computational backend.

    Subclasses set the class attributes and override only the capability methods
    they support. ``target`` and ``given`` name variables in the ontology, and must
    match the asset's ``cap:targetVariable`` / ``cap:givenVariable`` declarations.
    """

    asset_id: str = ""
    version: str = "0.0.0"
    target: tuple[str, ...] = ()
    given: tuple[str, ...] = ()
    deterministic: bool = False

    # ---- single-file assets ------------------------------------------------

    @classmethod
    def from_asset_file(cls, path: str | Path, **options: Any) -> "Asset":
        """Bind the wrapper to the one asset a ``<name>.asset.json`` declares.

        Override this where several single-file assets share the wrapper. The
        instance returned must answer for that file's asset alone, so that the
        harness tests the declaration it was handed and not a neighbour's.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not bind to a single-file asset; define from_asset_file()"
        )

    # ---- capability methods ------------------------------------------------

    def evaluate_density(
        self, target: Mapping[str, Any], given: Mapping[str, Any]
    ) -> np.ndarray:
        """p(target | given), pointwise."""
        raise NotImplementedError

    def sample_forward(
        self, given: Mapping[str, Any], n: int, rng: np.random.Generator
    ) -> Mapping[str, np.ndarray]:
        """Draw ``n`` values of target given parent values.

        Returns one array per target variable. Where the asset declares
        ``cap:Joint``, row i of every array must belong to the same draw.
        """
        raise NotImplementedError

    def evaluate_likelihood(
        self, target: Mapping[str, Any], given: Mapping[str, Any]
    ) -> np.ndarray:
        """L(given) = p(target = t | given), as a function of the parents.

        This is the backward direction and is generally NOT normalised over
        ``given``. Return the unnormalised function; do not rescale it here.
        """
        raise NotImplementedError

    def batch_classify(self, given: Mapping[str, np.ndarray]) -> np.ndarray:
        """Classify many parent-state samples at once."""
        raise NotImplementedError

    def gradient(
        self, target: Mapping[str, Any], given: Mapping[str, Any]
    ) -> Mapping[str, np.ndarray]:
        raise NotImplementedError

    def normalising_constant(self, given: Mapping[str, Any]) -> float:
        raise NotImplementedError

    # ---- conformance support ----------------------------------------------

    def implemented_capabilities(self) -> set[str]:
        """Which capability methods this class actually overrides."""
        found = set()
        for name, method in CAPABILITY_METHOD.items():
            own = getattr(type(self), method, None)
            base = getattr(Asset, method, None)
            if own is not None and own is not base:
                found.add(name)
        return found

    def smoke_inputs(self) -> dict[str, Any]:
        """Minimal valid inputs for the conformance harness to call each method.

        Override this. Return a dict with keys ``target`` and ``given`` holding
        values the asset accepts. The harness uses it to prove each declared
        capability runs, not merely that the method exists.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define smoke_inputs() so conformance can run it"
        )
