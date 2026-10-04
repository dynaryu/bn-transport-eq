"""GMPE as a lognormal kernel: HazardEvent -> LocalIntensity at one site.

A ground-motion prediction equation says

    ln Y = mu(rupture, distance, site) + eps,    eps ~ N(0, sigma)

so the factor it supplies is a lognormal conditional density. Every capability this
asset declares follows from that one form, and all of the modelling content sits in
``mu``, ``sigma`` and the coefficient table behind them.

What is implemented here is the *framework*: the contexts, the validation rules,
log-base and unit discipline, the period/damping grid policy, the extrapolation
policy, the lognormal kernel, and a registry of the 30 benchmark model families from
``params/registry.json``.

What is **not** implemented is any published median function. No authoritative
coefficient table for any of the 30 is present in this repository, so constructing
one of them raises :class:`CoefficientsUnavailable` naming exactly what source
material is missing, rather than returning a number. That is the rule the benchmark
specification puts first and the one this repository's registry puts first too: a
fabricated coefficient produces a confident wrong answer, which is worse than a
stop. See ``SOURCE.md`` for the per-model gap and ``README.md`` for the rationale.

The default ``model_id`` is a synthetic verification kernel, clearly labelled as such,
which exists only so the framework above can be tested without inventing anybody's
coefficients. It must never be used for hazard assessment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.wrappers.base import Asset

_HERE = Path(__file__).parent
_REGISTRY_PATH = _HERE / "params" / "registry.json"
_COEFFS_DIR = _HERE / "params" / "coeffs"
_VERIFICATION_PATH = _HERE / "params" / "verification-synthetic.json"

LN10 = math.log(10.0)
RESOLVE = "RESOLVE"
VERIFICATION_ID = "_verification"


# ---- errors ---------------------------------------------------------------
#
# Three distinct failure modes, kept distinct because they call for different
# actions: go and find a document, fix the call, or decide to extrapolate.


class CoefficientsUnavailable(RuntimeError):
    """The model is registered but no authoritative coefficient table is present."""


class ValidationError(ValueError):
    """The call is inadmissible: a missing input, or one outside a published grid."""


class OutOfScope(ValidationError):
    """Outside the calibration envelope, with extrapolation not explicitly allowed."""


# ---- log base and units ---------------------------------------------------


class LogBase(str, Enum):
    """Preserved from the publication. Converted only at the API boundary."""

    LN = "ln"
    LOG10 = "log10"

    def to_ln(self, x: Any) -> Any:
        return x * LN10 if self is LogBase.LOG10 else x


# Base unit per dimension: m/s^2, m/s, m. Conversion within a dimension only -- an
# acceleration model cannot be asked for a displacement.
_DIMENSION = {
    "g": "acceleration", "cm/s2": "acceleration", "m/s2": "acceleration",
    "cm/s": "velocity", "m/s": "velocity",
    "cm": "displacement", "m": "displacement",
}
_TO_BASE = {
    "g": 9.80665, "cm/s2": 0.01, "m/s2": 1.0,
    "cm/s": 0.01, "m/s": 1.0,
    "cm": 0.01, "m": 1.0,
}

_IMT_DIMENSION = {
    "PGA": "acceleration", "SA": "acceleration", "PSA": "acceleration",
    "PGV": "velocity", "PSV": "velocity",
    "SD": "displacement",
}
_DEFAULT_OUT_UNITS = {"acceleration": "g", "velocity": "cm/s", "displacement": "cm"}
_SPECTRAL_IMTS = {"SA", "PSA", "PSV", "SD"}


def unit_factor(frm: str, to: str) -> float:
    """Multiplicative factor taking a value in ``frm`` to ``to``.

    Raises across dimensions rather than guessing a pseudo-spectral conversion:
    PSV -> PSA depends on the period and is a modelling decision, not a unit change.
    """
    for u in (frm, to):
        if u not in _DIMENSION:
            raise ValidationError(f"unknown unit {u!r}; known: {sorted(_DIMENSION)}")
    if _DIMENSION[frm] != _DIMENSION[to]:
        raise ValidationError(
            f"cannot convert {frm} ({_DIMENSION[frm]}) to {to} ({_DIMENSION[to]}). "
            "Crossing dimensions is a modelling decision, not a unit conversion."
        )
    return _TO_BASE[frm] / _TO_BASE[to]


# ---- contexts -------------------------------------------------------------
#
# The three common contexts from the benchmark specification. Two additions, both
# forced by the registry rather than invented:
#
#   magnitude / magnitude_scale -- several of the 30 are not on Mw (Musson is ML,
#       others are RESOLVE). Silently feeding Mw into them changes the model, so a
#       non-Mw model reads `magnitude` and states its scale.
#   event_type -- the subduction models branch on interface/intraslab/crustal, which
#       the caller must supply. It is never inferred from depth here; any depth rule
#       belongs inside one specific model's kernel.


@dataclass
class RuptureContext:
    mw: Optional[float] = None
    magnitude: Optional[float] = None
    magnitude_scale: Optional[str] = None
    mechanism: Optional[str] = None
    event_type: Optional[str] = None
    depth_km: Optional[float] = None
    dip_deg: Optional[float] = None
    ztor_km: Optional[float] = None
    width_km: Optional[float] = None
    aftershock: Optional[bool] = None
    tectonic_region: Optional[str] = None


@dataclass
class DistanceContext:
    rjb_km: Optional[float] = None
    rrup_km: Optional[float] = None
    rhypo_km: Optional[float] = None
    repi_km: Optional[float] = None
    rx_km: Optional[float] = None
    rc_km: Optional[float] = None


@dataclass
class SiteContext:
    vs30_mps: Optional[float] = None
    site_class: Optional[str] = None
    z1p0_km: Optional[float] = None
    z2p5_km: Optional[float] = None


@dataclass
class GMPEOutput:
    median: Any
    ln_median: Any
    sigma_total: Optional[float] = None
    tau_inter: Optional[float] = None
    phi_intra: Optional[float] = None
    units: str = "g"
    component: Optional[str] = None
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


_CONTEXT_TYPES = {
    "rupture": RuptureContext,
    "distance": DistanceContext,
    "site": SiteContext,
}


def _coerce(kind: str, value: Any):
    """Accept a context object or a plain dict; reject unknown keys.

    Unknown keys raise rather than being dropped. A dropped ``Mw`` (wrong case) would
    reappear as a missing-required-input error pointing at the wrong thing.
    """
    cls = _CONTEXT_TYPES[kind]
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        known = {f.name for f in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValidationError(
                f"{kind} context got unknown field(s) {sorted(unknown)}; "
                f"known fields: {sorted(known)}"
            )
        return cls(**dict(value))
    raise ValidationError(f"{kind} must be a {cls.__name__} or a mapping, got {type(value).__name__}")


# ---- registry -------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One registry entry. ``raw`` is the registry dict; the properties are the parts
    the framework validates against."""

    model_id: str
    raw: Mapping[str, Any]

    @property
    def label(self) -> str:
        return self.raw.get("label", self.model_id)

    @property
    def kind(self) -> str:
        return self.raw.get("kind", "gmpe")

    @property
    def imts(self) -> tuple[str, ...]:
        return tuple(self.raw.get("imts", ()))

    @property
    def damping(self) -> tuple[float, ...]:
        return tuple(self.raw.get("damping", ()))

    @property
    def required(self) -> Mapping[str, Sequence[str]]:
        return self.raw.get("required", {})

    @property
    def distance_field(self) -> Optional[str]:
        f = self.raw.get("distance", {}).get("field")
        return None if f in (None, RESOLVE) else f

    @property
    def coefficient_status(self) -> str:
        return self.raw.get("coefficients", {}).get("status", "missing")

    @property
    def coefficients_needed(self) -> str:
        return self.raw.get("coefficients", {}).get("needed", "unspecified")

    @property
    def reproduction(self) -> Mapping[str, Any]:
        """Per-model completeness, from the benchmark spec's coefficient appendix.

        `ANCHOR_BRANCH_REPRODUCIBLE` means one branch (normally PGA) can be evaluated.
        It does NOT license the spectral periods: those coefficients are a separate
        table, and generating them from the anchor is the error the appendix's
        completeness gate exists to prevent.
        """
        return self.raw.get("reproduction", {})

    @property
    def anchor_branch(self) -> Optional[str]:
        """The branch that is numerically reproducible today, if any."""
        if self.coefficient_status != "anchor_only":
            return None
        return self.reproduction.get("branch")

    def unresolved(self) -> list[str]:
        """Dotted paths whose value is still RESOLVE.

        These are the fields the catalogue summary does not pin down -- log base,
        component definition, magnitude scale and the like. Each one is a silent
        error waiting to happen, so they are surfaced rather than defaulted.
        """
        found: list[str] = []

        def walk(node: Any, path: str) -> None:
            if node == RESOLVE or (isinstance(node, str) and node.endswith("- RESOLVE")):
                found.append(path)
            elif isinstance(node, Mapping):
                for k, v in node.items():
                    walk(v, f"{path}.{k}" if path else str(k))
            elif isinstance(node, (list, tuple)):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(dict(self.raw), "")
        return sorted(found)


def load_registry(path: Path | str = _REGISTRY_PATH) -> dict[str, ModelSpec]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: ModelSpec(k, v) for k, v in doc["models"].items()}


def registry() -> dict[str, ModelSpec]:
    return load_registry()


def coefficient_status() -> dict[str, str]:
    """How complete each model's coefficients are.

    None of them is `present`: `present` additionally requires a coefficient file
    with provenance and a registered kernel, so every model still refuses.
    """
    return {mid: s.coefficient_status for mid, s in load_registry().items()}


def reproduction_status() -> dict[str, dict[str, Any]]:
    """The completeness gate, per model.

    Reading order for anyone about to implement one: this says whether a branch can
    be evaluated at all, `coefficient_status` says whether this repo holds the data,
    and `ModelSpec.unresolved()` says what the catalogue never pinned down.
    """
    out = {}
    for mid, spec in sorted(load_registry().items(), key=lambda kv: kv[1].raw["benchmark_id"]):
        out[mid] = {
            "benchmark_id": spec.raw["benchmark_id"],
            "status": spec.reproduction.get("status"),
            "branch": spec.reproduction.get("branch"),
            "spectral": spec.reproduction.get("spectral_coefficients"),
            "coefficients": spec.coefficient_status,
            "n_unresolved": len(spec.unresolved()),
        }
    return out


def missing_sources() -> list[dict[str, Any]]:
    """Per-model report of what source material has to be obtained.

    This is the deliverable the benchmark specification asks for when a coefficient
    set is unavailable: say exactly what is missing instead of producing a number.
    """
    out = []
    for mid, spec in sorted(load_registry().items(), key=lambda kv: kv[1].raw["benchmark_id"]):
        if spec.coefficient_status == "present":
            continue
        out.append({
            "benchmark_id": spec.raw["benchmark_id"],
            "model_id": mid,
            "label": spec.label,
            "douglas_refs": spec.raw.get("douglas_refs", []),
            "status": spec.coefficient_status,
            "needed": spec.coefficients_needed,
            "unresolved_fields": spec.unresolved(),
        })
    return out


def coverage_matrix() -> list[dict[str, Any]]:
    """Which common-context fields each of the 30 requires, plus its coefficient state.

    The cross-model view the specification asks for before implementation starts: it
    is what shows that the common contexts have to carry explicit rupture geometry,
    five distance metrics, both continuous and categorical sites, an event-type
    enumeration and non-Mw magnitude scales.
    """
    rows = []
    for mid, spec in sorted(load_registry().items(), key=lambda kv: kv[1].raw["benchmark_id"]):
        req = spec.required
        rows.append({
            "benchmark_id": spec.raw["benchmark_id"],
            "model_id": mid,
            "kind": spec.kind,
            "tectonic_types": spec.raw.get("tectonic_types", []),
            "magnitude_scale": spec.raw.get("magnitude", {}).get("scale"),
            "distance_metric": spec.raw.get("distance", {}).get("metric"),
            "site_parameterization": spec.raw.get("site", {}).get("parameterization"),
            "imts": list(spec.imts),
            "damping": list(spec.damping),
            "component": spec.raw.get("component"),
            "log_base": spec.raw.get("log_base"),
            "required_rupture": list(req.get("rupture", [])),
            "required_distance": list(req.get("distance", [])),
            "required_site": list(req.get("site", [])),
            "required_other": list(req.get("path", [])) + list(req.get("other", [])),
            "coefficients": spec.coefficient_status,
            "n_unresolved": len(spec.unresolved()),
        })
    return rows


# ---- median kernels -------------------------------------------------------


class MedianKernel:
    """Computes mu and the aleatory terms in the model's OWN log base and units.

    Conversion to ln and to the caller's units happens once, in :class:`Gmpe`. A
    kernel that converted internally would make the published form unrecognisable
    against the paper, which is where review of these implementations happens.
    """

    log_base: LogBase = LogBase.LN
    units: str = "g"
    component: Optional[str] = None
    period_grid: Optional[tuple[str, ...]] = None

    def mu_sigma(
        self, *, imt: str, period_s: Optional[float], damping: float,
        rupture: RuptureContext, distance: DistanceContext, site: SiteContext,
        other: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Optional[float]]]:
        raise NotImplementedError


class _VerificationKernel(MedianKernel):
    """SYNTHETIC. Not a GMPE, and not an approximation of one.

    Its coefficients are invented, as ``params/verification-synthetic.json`` states in
    its own first field. It exists so the framework can be exercised end to end --
    log10 input, cm/s2 input, a real period grid, tau/phi composition -- without
    fabricating coefficients for any of the 30 published families.
    """

    def __init__(self, path: Path = _VERIFICATION_PATH) -> None:
        self.doc = json.loads(Path(path).read_text(encoding="utf-8"))
        self.log_base = LogBase(self.doc["log_base"])
        self.units = self.doc["units"]
        self.component = self.doc["component"]
        self.coeffs = self.doc["coefficients"]
        self.period_grid = tuple(self.coeffs)

    @staticmethod
    def _key(imt: str, period_s: Optional[float]) -> str:
        return "PGA" if imt == "PGA" else f"{period_s:g}"

    def mu_sigma(self, *, imt, period_s, damping, rupture, distance, site, other):
        c = self.coeffs[self._key(imt, period_s)]
        m = rupture.mw
        r = distance.rhypo_km
        if site.site_class not in c["c"]:
            raise ValidationError(
                f"site_class {site.site_class!r} not in the synthetic kernel's classes "
                f"{sorted(c['c'])}"
            )
        mu = c["a"] * m + c["b"] * r - math.log10(r) + c["c"][site.site_class]
        return mu, {"tau": c["tau"], "phi": c["phi"], "sigma": None}


# No published median function is registered. Each of the 30 enters here only once an
# authoritative coefficient table exists in params/coeffs/ with recorded provenance.
_KERNELS: dict[str, type[MedianKernel]] = {}


# ---- the asset ------------------------------------------------------------


class Gmpe(Asset):
    """One GMPE, as a lognormal factor supplying LocalIntensity at a single site."""

    asset_id = "hazard/gmpe"
    version = "0.1.0"
    target = ("LocalIntensity",)
    given = ("HazardEvent",)
    deterministic = False

    def __init__(
        self,
        model_id: str = VERIFICATION_ID,
        *,
        allow_extrapolation: bool = False,
        registry_path: Path | str = _REGISTRY_PATH,
    ) -> None:
        self.model_id = model_id
        self.allow_extrapolation = allow_extrapolation
        self._registry = load_registry(registry_path)

        if model_id == VERIFICATION_ID:
            self._kernel: MedianKernel = _VerificationKernel()
            self.spec = ModelSpec(VERIFICATION_ID, self._verification_spec(self._kernel.doc))
            return

        if model_id not in self._registry:
            raise ValidationError(
                f"unknown model_id {model_id!r}. Registered: {sorted(self._registry)}"
            )

        self.spec = self._registry[model_id]
        if self.spec.coefficient_status != "present" or model_id not in _KERNELS:
            raise CoefficientsUnavailable(self._gap_report(self.spec))
        self._kernel = _KERNELS[model_id]()

    # ---- construction helpers -------------------------------------------

    @staticmethod
    def _verification_spec(doc: Mapping[str, Any]) -> dict[str, Any]:
        """Put the synthetic kernel into registry shape so one validation path serves both."""
        return {
            "benchmark_id": "--", "kind": "verification", "label": doc["label"],
            "region": "none", "tectonic_types": [],
            "magnitude": doc["magnitude"], "distance": doc["distance"],
            "site": {"parameterization": "categorical", "classes": doc["site_classes"],
                     "fields": ["site_class"]},
            "imts": doc["imts"], "damping": doc["damping"],
            "periods": {"grid": list(doc["coefficients"])},
            "component": doc["component"], "log_base": doc["log_base"],
            "internal_units": doc["units"],
            "required": {"rupture": ["mw"], "distance": [doc["distance"]["field"]],
                         "site": ["site_class"]},
            "coefficients": {"status": "present", "needed": "none - synthetic"},
            "WARNING": doc["WARNING"],
        }

    @staticmethod
    def _gap_report(spec: ModelSpec) -> str:
        repro = spec.reproduction
        lines = [
            f"{spec.label} ({spec.model_id}) is registered but cannot predict: "
            f"coefficients are {spec.coefficient_status}.",
            "",
            f"Reproduction status: {repro.get('status', 'unknown')}"
            + (f" (branch: {repro['branch']})" if repro.get("branch") else ""),
        ]
        # An anchor branch being available is a different situation from nothing being
        # available, and the message has to say which -- otherwise it sends someone
        # looking for a table that is already in hand.
        if spec.coefficient_status == "anchor_only":
            lines += [
                f"  An anchor branch ({repro.get('branch')}) IS reproducible from the "
                "spec appendix. What is missing is the spectral table, and no "
                "MedianKernel is implemented for this model yet.",
                "  Implementing the anchor branch is unblocked; off-anchor periods must "
                "still be refused rather than interpolated from it.",
            ]
        elif spec.coefficient_status == "partial":
            lines.append(
                "  Not even the anchor branch is complete. "
                + str(repro.get("incomplete_note", ""))
            )
        lines += [
            "",
            f"Source material needed: {spec.coefficients_needed}",
            f"Douglas (2011) sections: {', '.join(spec.raw.get('douglas_refs', [])) or 'n/a'}",
        ]
        unresolved = spec.unresolved()
        if unresolved:
            lines += [
                "",
                "Still unresolved -- read these off the original paper, do not assume them:",
                *(f"  - {p}" for p in unresolved),
            ]
        lines += [
            "",
            "Supply params/coeffs/<model_id>.json with recorded provenance, register a "
            "MedianKernel, and set coefficients.status to 'present'. Coefficients must "
            "never be reconstructed from figures or recalled from memory.",
        ]
        return "\n".join(lines)

    # ---- validation ------------------------------------------------------

    def _validate(
        self, imt: str, period_s: Optional[float], damping: float,
        rupture: RuptureContext, distance: DistanceContext, site: SiteContext,
        other: Mapping[str, Any],
    ) -> list[str]:
        spec, warns = self.spec, []

        if spec.kind == "benchmark":
            raise ValidationError(
                f"{spec.label} does not supply a median. {spec.coefficients_needed}"
            )

        if imt not in spec.imts:
            raise ValidationError(f"{spec.label} supports IMTs {list(spec.imts)}, got {imt!r}")

        # Damping is a published grid, not a continuum. Substituting 5% for an
        # unlisted ratio is the single easiest way to misuse the damping-diverse
        # models in this set (Chen & Yu, Berge-Thierry, Cauzzi & Faccioli).
        if spec.damping and not any(abs(damping - d) < 1e-12 for d in spec.damping):
            raise ValidationError(
                f"{spec.label} publishes damping {list(spec.damping)}, got {damping}. "
                "No interpolation rule is implemented."
            )

        # Specific diagnostics first, generic missing-input check last. Both raise
        # ValidationError, but "Rrup is not a substitute for Rhypo" tells the caller
        # what to do and a bare "rhypo_km missing" does not.
        self._check_distance_metric(distance)
        self._check_site_parameterization(site)
        self._check_required(rupture, distance, site, other)
        warns += self._check_period(imt, period_s)
        warns += self._check_envelope(rupture, distance)
        return warns

    def _check_required(self, rupture, distance, site, other) -> None:
        """Every declared required field must be present. Nothing is defaulted.

        A missing Vs30 or site class is a validation error, never an assumed value --
        the caller has to state the assumption so it is visible in their code.
        """
        sources = {"rupture": rupture, "distance": distance, "site": site}
        missing = []
        for group, names in self.spec.required.items():
            for name in names:
                if name == RESOLVE:
                    raise ValidationError(
                        f"{self.spec.label} has an unresolved required {group} field. "
                        "Read it off the original paper before calling."
                    )
                if group in sources:
                    if getattr(sources[group], name, None) is None:
                        missing.append(f"{group}.{name}")
                elif other.get(name) is None:
                    missing.append(f"other.{name}")
        if missing:
            raise ValidationError(
                f"{self.spec.label} requires {missing}, which were not supplied. "
                "Supply them explicitly; this asset does not infer missing inputs."
            )

    def _check_distance_metric(self, distance: DistanceContext) -> None:
        """No heuristic distance substitution.

        Rjb and Rrup differ by the rupture geometry, Rhypo and Repi by the depth. A
        model fitted on one metric evaluated with another is a different model, so if
        the required metric is absent the call fails even when a near relative is to
        hand.
        """
        want = self.spec.distance_field
        if want is None or getattr(distance, want, None) is not None:
            return
        supplied = [f.name for f in fields(distance) if getattr(distance, f.name) is not None]
        raise ValidationError(
            f"{self.spec.label} requires {want}; supplied {supplied or 'nothing'}. "
            "Distance metrics are not interchangeable and are not converted here. "
            "Compute the required metric from the rupture geometry, or use the model's "
            "own published fallback if it defines one."
        )

    def _check_site_parameterization(self, site: SiteContext) -> None:
        """A categorical-site model is never handed a Vs30 and left to guess.

        Where the source publishes a Vs30 -> class mapping (Ambraseys, Ghasemi,
        Lussou) the registry records its exact boundaries, but applying it belongs
        with that model's kernel and coefficient set. Until then the caller maps
        deliberately and the class is explicit in their code.
        """
        s = self.spec.raw.get("site", {})
        param = s.get("parameterization")
        # "none" means the published equation has no site term at all (Musson), so
        # there is no class to demand.
        if param in (None, RESOLVE, "none", "fixed_condition", "vs30_continuous", "vs30_banded"):
            return
        if site.site_class is None and site.vs30_mps is not None:
            classes = s.get("classes", [])
            raise ValidationError(
                f"{self.spec.label} is a {param} site model and needs site_class, but only "
                f"vs30_mps={site.vs30_mps} was given. Published classes: {classes}. "
                "Map it yourself and pass site_class -- the mapping is not applied silently."
            )

    def _check_period(self, imt: str, period_s: Optional[float]) -> list[str]:
        if imt not in _SPECTRAL_IMTS:
            return []
        if period_s is None:
            raise ValidationError(f"{imt} is spectral and needs period_s")
        grid = self._kernel.period_grid
        if grid is None:
            raise ValidationError(
                f"{self.spec.label} has no resolved period grid, so no period can be "
                "validated. Interpolation is not permitted without a documented rule."
            )
        if f"{period_s:g}" not in grid:
            raise ValidationError(
                f"period {period_s} s is not on the published grid "
                f"{[g for g in grid if g != 'PGA']}. Interpolation is model-specific and "
                "is only allowed where the source permits it."
            )
        return []

    def _check_envelope(self, rupture, distance) -> list[str]:
        """Calibration-range policy: refuse by default, warn when told to extrapolate."""
        warns: list[str] = []
        mspec = self.spec.raw.get("magnitude", {})
        m = rupture.mw if rupture.mw is not None else rupture.magnitude
        warns += self._band("magnitude", m, mspec.get("min"), mspec.get("max"))

        dspec = self.spec.raw.get("distance", {})
        fld = self.spec.distance_field
        r = getattr(distance, fld, None) if fld else None
        warns += self._band(
            f"distance ({dspec.get('metric', '?')})", r, dspec.get("min_km"), dspec.get("max_km")
        )
        return warns

    def _band(self, what: str, value, lo, hi) -> list[str]:
        if value is None:
            return []
        out = []
        for bound, limit, cmp in (("below", lo, value.__lt__), ("above", hi, value.__gt__)):
            if isinstance(limit, (int, float)) and cmp(limit):
                msg = (
                    f"{what} {value} is {bound} the stated calibration limit {limit} "
                    f"for {self.spec.label}"
                )
                if not self.allow_extrapolation:
                    raise OutOfScope(
                        msg + ". Pass allow_extrapolation=True to proceed with a warning."
                    )
                out.append("EXTRAPOLATION: " + msg)
        return out

    # ---- prediction ------------------------------------------------------

    def predict(
        self,
        imt: str = "PGA",
        period_s: Optional[float] = None,
        damping: float = 0.05,
        rupture: Any = None,
        distance: Any = None,
        site: Any = None,
        other: Optional[Mapping[str, Any]] = None,
        units: Optional[str] = None,
    ) -> GMPEOutput:
        """Median and aleatory terms for one scenario at one site."""
        rup = _coerce("rupture", rupture)
        dis = _coerce("distance", distance)
        sit = _coerce("site", site)
        oth = dict(other or {})

        warns = self._validate(imt, period_s, damping, rup, dis, sit, oth)
        out_units = units or _DEFAULT_OUT_UNITS[_IMT_DIMENSION[imt]]

        mu_native, sig_native = self._kernel.mu_sigma(
            imt=imt, period_s=period_s, damping=damping,
            rupture=rup, distance=dis, site=sit, other=oth,
        )

        # Two independent conversions, deliberately separated.
        #   log base: scales both mu and the aleatory terms.
        #   units:    a multiplicative rescaling of Y, so it SHIFTS ln Y and leaves
        #             its spread alone. Applying it to sigma is a classic silent bug.
        base = self._kernel.log_base
        ln_median = base.to_ln(mu_native) + math.log(unit_factor(self._kernel.units, out_units))
        tau = base.to_ln(sig_native.get("tau")) if sig_native.get("tau") is not None else None
        phi = base.to_ln(sig_native.get("phi")) if sig_native.get("phi") is not None else None
        sigma = base.to_ln(sig_native.get("sigma")) if sig_native.get("sigma") is not None else None

        sigma_derived = False
        if sigma is None and tau is not None and phi is not None:
            sigma = math.sqrt(tau * tau + phi * phi)
            sigma_derived = True

        if self.spec.raw.get("WARNING"):
            warns.append("SYNTHETIC: " + self.spec.raw["WARNING"].split(".")[0] + ".")

        return GMPEOutput(
            median=np.exp(ln_median), ln_median=ln_median,
            sigma_total=sigma, tau_inter=tau, phi_intra=phi,
            units=out_units, component=self._kernel.component,
            warnings=tuple(warns),
            provenance={
                "model_id": self.model_id, "asset_version": self.version,
                "model_label": self.spec.label, "imt": imt, "period_s": period_s,
                "damping": damping, "distance_metric": self.spec.raw.get("distance", {}).get("metric"),
                "component": self._kernel.component,
                "native_log_base": base.value, "native_units": self._kernel.units,
                "sigma_total_derived_from_tau_phi": sigma_derived,
                "extrapolated": any(w.startswith("EXTRAPOLATION") for w in warns),
                "spatial_correlation": None,
            },
        )

    def _from_given(self, given: Mapping[str, Any]) -> GMPEOutput:
        return self.predict(
            imt=given.get("imt", "PGA"),
            period_s=given.get("period_s"),
            damping=given.get("damping", 0.05),
            rupture=given.get("rupture"),
            distance=given.get("distance"),
            site=given.get("site"),
            other=given.get("other"),
            units=given.get("units"),
        )

    def _sigma_or_raise(self, out: GMPEOutput) -> float:
        if out.sigma_total is None:
            raise ValidationError(
                f"{self.spec.label} published no aleatory term here, so the conditional "
                "distribution is undefined. A median alone cannot be sampled or scored."
            )
        return float(out.sigma_total)

    # ---- capabilities ----------------------------------------------------

    def sample_forward(
        self, given: Mapping[str, Any], n: int = 1, rng: np.random.Generator | None = None
    ) -> Mapping[str, np.ndarray]:
        """Draw ``n`` intensities for ONE site.

        Draws are independent. This asset carries no spatial correlation model, which
        is why asset.ttl declares cap:Marginal -- see README.md. Calling it per site
        and stacking the results imposes independence across sites and will understate
        the probability of widespread damage.
        """
        out = self._from_given(given)
        sigma = self._sigma_or_raise(out)
        rng = np.random.default_rng() if rng is None else rng
        eps = rng.standard_normal(n)
        return {"LocalIntensity": np.exp(out.ln_median + sigma * eps)}

    def evaluate_density(
        self, target: Mapping[str, Any], given: Mapping[str, Any]
    ) -> np.ndarray:
        """Lognormal p(LocalIntensity | scenario), pointwise, in the output units."""
        out = self._from_given(given)
        sigma = self._sigma_or_raise(out)
        y = np.asarray(target["LocalIntensity"], dtype=float)
        dens = np.zeros_like(y, dtype=float)
        ok = y > 0.0  # the lognormal has no mass at or below zero
        z = (np.log(y[ok]) - out.ln_median) / sigma
        dens[ok] = np.exp(-0.5 * z * z) / (y[ok] * sigma * math.sqrt(2.0 * math.pi))
        return dens

    def evaluate_likelihood(
        self, target: Mapping[str, Any], given: Mapping[str, Any]
    ) -> np.ndarray:
        """L(scenario) = p(LocalIntensity = y | scenario), read over the parents.

        The same lognormal ordinate as ``evaluate_density``, and deliberately a
        separate method: this is the backward message, unnormalised over the parents,
        and that is what makes inference from a recorded intensity possible. Returned
        unrescaled.
        """
        return self.evaluate_density(target, given)

    # ---- metadata --------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Region, regime, ranges, component, site model, IMTs, periods, damping."""
        raw = self.spec.raw
        return {
            "model_id": self.model_id, "label": self.spec.label,
            "asset_id": self.asset_id, "asset_version": self.version,
            "region": raw.get("region"), "tectonic_types": raw.get("tectonic_types", []),
            "magnitude": raw.get("magnitude", {}), "distance": raw.get("distance", {}),
            "site": raw.get("site", {}), "mechanism": raw.get("mechanism", {}),
            "imts": list(self.spec.imts), "periods": raw.get("periods", {}),
            "damping": list(self.spec.damping), "component": raw.get("component"),
            "log_base": raw.get("log_base"), "internal_units": raw.get("internal_units"),
            "coefficients": raw.get("coefficients", {}),
            "spatial_correlation": None,
            "unresolved_fields": self.spec.unresolved(),
            "douglas_refs": raw.get("douglas_refs", []),
        }

    def required_inputs(self, imt: str = "PGA", period: Optional[float] = None) -> dict[str, list[str]]:
        """Which rupture/distance/site fields the caller must supply."""
        req = {k: list(v) for k, v in self.spec.required.items()}
        if imt in _SPECTRAL_IMTS:
            req.setdefault("call", []).append("period_s")
        return req

    def validity(self) -> dict[str, Any]:
        """The calibration envelope, and what is known not to be pinned down."""
        raw = self.spec.raw
        return {
            "magnitude": raw.get("magnitude", {}), "distance": raw.get("distance", {}),
            "periods": raw.get("periods", {}), "damping": list(self.spec.damping),
            "tectonic_types": raw.get("tectonic_types", []), "region": raw.get("region"),
            "site": raw.get("site", {}),
            "allow_extrapolation": self.allow_extrapolation,
            "unresolved_fields": self.spec.unresolved(),
            "notes": raw.get("implementation_notes", []),
        }

    # ---- conformance -----------------------------------------------------

    def smoke_inputs(self) -> dict[str, Any]:
        return {
            "given": {
                "imt": "PGA", "damping": 0.05,
                "rupture": {"mw": 6.0},
                "distance": {"rhypo_km": 20.0},
                "site": {"site_class": "rock"},
            },
            "target": {"LocalIntensity": np.array([0.05, 0.10, 0.20])},
        }
