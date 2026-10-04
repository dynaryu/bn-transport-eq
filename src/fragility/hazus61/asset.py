"""Registry capabilities for one asset, with batches of intensity scenarios."""
import numpy as np

from src.wrappers.base import Asset
from .engine import evaluate, match


class HazusAsset(Asset):
    version = "0.1.0"
    target = ("AssetDamageState",)
    given = ("LocalIntensity",)
    deterministic = False
    family = ""
    states = ()

    def __init__(self, hazard_mode=None, application="modified", bridge_pgd_policy=None, vintage_policy="table_7_1"):
        self.hazard_mode = hazard_mode or ("ground_failure" if self.family == "roadway" else "ground_shaking")
        if self.hazard_mode not in {"ground_shaking", "ground_failure"} or (self.family == "roadway" and self.hazard_mode != "ground_failure"):
            raise ValueError("unsupported hazard_mode for this family")
        self.options = dict(hazard_mode=self.hazard_mode, application=application,
                            bridge_pgd_policy=bridge_pgd_policy, vintage_policy=vintage_policy)

    def match(self, asset, hazard=None):
        return match(self.family, asset, self.hazard_mode, hazard, self.options["vintage_policy"])

    def evaluate(self, given):
        return evaluate(self.family, given, **self.options)

    def distribution(self, given):
        result = self.evaluate(given)
        if "probabilities" not in result:
            raise ValueError(f"unresolved asset: {result['match_status']}; candidates={result['candidate_classes']}; missing={result['missing_fields']}")
        return np.asarray(result["probabilities"])

    def evaluate_density(self, target, given):
        probabilities = self.distribution(given)
        requested = np.atleast_1d(target["AssetDamageState"])
        if requested.ndim != 1 or len(requested) not in (1, len(probabilities)):
            raise ValueError("damage state must be scalar or match the intensity batch")
        if not all(s in self.states for s in requested):
            raise ValueError(f"damage states must be named values in {self.states}")
        requested = np.broadcast_to(requested, (len(probabilities),))
        return np.asarray([row[self.states.index(s)] for row, s in zip(probabilities, requested)])

    def evaluate_likelihood(self, target, given):
        return self.evaluate_density(target, given)

    def sample_forward(self, given, n=1, rng=None):
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng() if rng is None else rng
        probabilities = self.distribution(given)
        if len(probabilities) == 1:
            probabilities = np.repeat(probabilities, n, axis=0)
        elif n not in (1, len(probabilities)):
            raise ValueError("batched parents require n=1 or n equal to the batch size")
        uniforms = rng.random(len(probabilities))
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0  # sampling-only roundoff guard, not a CPD repair
        indices = (uniforms[:, None] >= cumulative).sum(axis=1)
        return {"AssetDamageState": np.asarray(self.states)[indices]}

    def smoke_inputs(self):
        if self.family == "roadway":
            given = {"asset": {"hazus_roadway_class": "HRD1"}, "LocalIntensity": {"pgd_in": 12}}
        else:
            given = {"asset": {"hazus_bridge_class": "HWB17", "num_spans": 3, "skew_angle_deg": 20,
                               "bridge_length_m": 90, "span_width_m": 12},
                     "LocalIntensity": {"sa_1p0_g": 0.43, "pgd_in": 4}}
        return {"given": given, "target": {"AssetDamageState": "none"}, "n": 4, "rng": np.random.default_rng(42)}
