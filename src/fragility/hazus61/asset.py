"""Registry capabilities for one asset, with batches of intensity scenarios."""
from pathlib import Path

import numpy as np

from src.wrappers.base import Asset
from src.wrappers.record import read_asset_file
from .classify import normalize_asset
from .data import folder
from .engine import evaluate, match


class HazusAsset(Asset):
    version = "0.1.0"
    target = ("AssetDamageState",)
    given = ("LocalIntensity",)
    deterministic = False
    family = ""
    states = ()
    asset_class = None  # the one Hazus class answered for, when bound to an asset file

    def __init__(self, hazard_mode=None, application="modified", bridge_pgd_policy=None, vintage_policy="table_7_1"):
        self.hazard_mode = hazard_mode or ("ground_failure" if self.family == "roadway" else "ground_shaking")
        if self.hazard_mode not in {"ground_shaking", "ground_failure"} or (self.family == "roadway" and self.hazard_mode != "ground_failure"):
            raise ValueError("unsupported hazard_mode for this family")
        self.options = dict(hazard_mode=self.hazard_mode, application=application,
                            bridge_pgd_policy=bridge_pgd_policy, vintage_policy=vintage_policy)

    @classmethod
    def from_asset_file(cls, path, **options):
        """Answer for the single class one ``<class>.asset.json`` declares.

        The file states its own application policy, so a class whose modified curve
        the source cannot supply is offered as base curves by its declaration and
        not by a fallback here. An explicit ``application`` still overrides it.
        """
        path = Path(path).resolve()
        if path.parent != folder(cls.family).resolve():
            raise ValueError(f"{path.name} is not one of this wrapper's assets in {folder(cls.family)}")
        record = read_asset_file(path)
        parameters = record["parameters"]
        if parameters["asset_family"] != cls.family:
            raise ValueError(f"{path.name} is a {parameters['asset_family']} asset, not {cls.family}")
        if "application" in parameters:
            options.setdefault("application", parameters["application"])
        model = cls(**options)
        model.asset_id = record["assetId"]
        model.asset_class = parameters["asset_class"]
        return model

    def _class_key(self):
        return "hazus_roadway_class" if self.family == "roadway" else "hazus_bridge_class"

    def _own(self, asset):
        """Asset metadata with the bound class stated; a contradicting class is refused."""
        if self.asset_class is None:
            return asset
        asset = normalize_asset({} if asset is None else asset, self.family)
        stated = asset.get(self._class_key())
        if stated not in (None, self.asset_class):
            raise ValueError(f"{self.asset_id} supplies {self.asset_class} only, but the asset metadata states {stated}")
        return {**asset, self._class_key(): self.asset_class}

    def match(self, asset, hazard=None):
        return match(self.family, self._own(asset), self.hazard_mode, hazard, self.options["vintage_policy"])

    def evaluate(self, given):
        if self.asset_class is not None:
            if given.get("class_probabilities") is not None:
                raise ValueError(f"{self.asset_id} supplies {self.asset_class} only; a class mixture needs the family wrapper")
            given = {**given, "asset": self._own(given.get("asset"))}
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
        if self.asset_class is not None and self.family == "highway_bridge":
            # No class here: the binding supplies it. Zero skew, because the shaking
            # modifiers can cross thresholds at an angle and that is an error by policy.
            given = {"asset": {"num_spans": 3, "skew_angle_deg": 0},
                     "LocalIntensity": {"sa_1p0_g": 0.43, "sa_0p3_g": 0.8}}
        elif self.asset_class is not None:
            given = {"asset": {}, "LocalIntensity": {"pgd_in": 12}}
        elif self.family == "roadway":
            given = {"asset": {"hazus_roadway_class": "HRD1"}, "LocalIntensity": {"pgd_in": 12}}
        else:
            given = {"asset": {"hazus_bridge_class": "HWB17", "num_spans": 3, "skew_angle_deg": 20,
                               "bridge_length_m": 90, "span_width_m": 12},
                     "LocalIntensity": {"sa_1p0_g": 0.43, "pgd_in": 4}}
        return {"given": given, "target": {"AssetDamageState": "none"}, "n": 4, "rng": np.random.default_rng(42)}
