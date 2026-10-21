"""Scenario and ground-motion kernels: everything that produces a Rupture, an IM or its residual."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import Call

_EARTH_RADIUS_KM = 6371.0


class ScenarioFixed:
    """The Event row, copied into a degenerate Rupture node."""

    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        fields = list(call.quantities["rupture"]["fields"])
        out = np.zeros((call.n, len(call.rows)), dtype=[(f, float) for f in fields])
        for j, row in enumerate(call.rows):
            for f in fields:
                out[f][:, j] = row[f]
        return {"rupture": out}


def great_circle_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def campbell_2003_ln_median(mw: np.ndarray, r_km: np.ndarray, c: dict[str, float]) -> np.ndarray:
    """ln median SA in g, term for term as the EMA dataset script implements it, for the coefficients given."""
    r = np.maximum(r_km, 1e-9)
    f1 = c["c2"] * mw + c["c3"] * (8.5 - mw) ** 2
    radius = np.sqrt(r**2 + (c["c7"] * np.exp(c["c8"] * mw)) ** 2)
    f2 = c["c4"] * np.log(radius) + (c["c5"] + c["c6"] * mw) * r
    near = c["c7"] * (np.log(r) - np.log(70.0))
    f3 = np.where(r <= 70.0, 0.0,
                  np.where(r <= 130.0, near, near + c["c8"] * (np.log(r) - np.log(130.0))))
    return c["c1"] + f1 + f2 + f3


def campbell_2003(mw: np.ndarray, r_km: np.ndarray, p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Campbell (2003) form: ln median in g, and aleatory sigma from the set's magnitude-dependent model."""
    s = p["sigma"]
    return (campbell_2003_ln_median(mw, r_km, p["coefficients"]),
            np.where(mw < s["m_break"], s["a"] + s["b"] * mw, s["sigma_above"]))


# Functional forms, by the parameter set's "form". A new GMM with an existing form is only
# a parameter set; a new form is a function here plus its branch in schema/functions/hazard.gmm.json.
GMM_FORMS = {
    "campbell-2003": campbell_2003,
}


class Gmm:
    """ln IM = ln median(Mw, R) + sigma * eps at each bridge, for whichever GMM the parameter set is."""

    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        p = call.params
        event = call.inputs["rupture"][0]          # (n, 1): the singleton Event, shared by every bridge
        if event.shape[1] != 1:
            raise ValueError(f"expected one Rupture parent, got {event.shape[1]}")
        event = event[:, 0]
        mw = event["mw"][:, None]
        lo, hi = p["magnitude_range"]
        if np.any((mw < lo) | (mw > hi)):
            raise ValueError(f"Mw outside this parameter set's range [{lo}, {hi}]")
        if p.get("distance_metric", "hypocentral") != "hypocentral":
            raise ValueError(f"distance metric {p['distance_metric']!r} is not implemented")

        repi = great_circle_km(event["lat"][:, None], event["lon"][:, None],
                               call.column("lat"), call.column("lon"))
        r = np.sqrt(repi**2 + event["depth_km"][:, None] ** 2)
        ln_median, aleatory = GMM_FORMS[p["form"]](mw, r, p)
        sigma = np.sqrt(aleatory**2 + p["sigma_epistemic"] ** 2)
        eps = call.rng.standard_normal(ln_median.shape)
        return {"im": np.exp(ln_median + sigma * eps)}


class ShakeMap:
    """ln IM = ln median (grid.xml, e.g. PSA10) + sigma (uncertainty.xml, e.g. STDPSA10) * epsilon."""

    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        from .. import shakemap   # imported late: only this kernel fetches data

        # shakemap.load caches the parsed grid, and re-asks which version is current
        # every few minutes, so a long-running server picks up a revised map.
        p = call.params
        grid = shakemap.load(p["event_id"], p.get("source"), p.get("version"), p["imt"])
        ln_sa, sigma, inside = grid.sample(call.column("lat")[0], call.column("lon")[0])
        if not inside.all():
            outside = [row["id"] for row, ok in zip(call.rows, inside) if not ok]
            raise ValueError(
                f"{len(outside)} site(s) lie outside ShakeMap {grid.event_id} v{grid.version}, "
                f"e.g. {outside[:3]}; outside the map means not estimated, not zero"
            )
        meta = grid.meta
        call.warnings.append(
            f"Ground motion from USGS ShakeMap {grid.event_id} version {grid.version} "
            f"({meta.get('originator')}, {meta.get('map_status')}, via {meta.get('source')})."
        )
        if not meta.get("seismic_stations"):
            call.warnings.append(
                "This ShakeMap has no seismic station recordings, so its shaking is a ground-motion-model "
                "estimate (constrained by felt reports at most) with the larger uncertainty that implies."
            )
        call.warnings.extend(grid.warnings)
        epsilon = np.asarray(call.inputs["epsilon"], float)
        return {"im": np.exp(ln_sa[None, :] + sigma[None, :] * epsilon)}


def jayaram_baker_2009_range_km(period_s: float, vs30_clustered: bool) -> float:
    """Range b of rho(h) = exp(-3h / b): case 1 (eq. 17) or, with Vs30 clustering, case 2 (eq. 18) below 1 s."""
    if period_s < 1.0:
        return 40.7 - 15.0 * period_s if vs30_clustered else 8.5 + 17.2 * period_s
    return 22.0 + 3.7 * period_s


def correlation_matrix(lat: np.ndarray, lon: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    if params["model"] == "independent":
        return np.eye(len(lat))
    distance = great_circle_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    b = jayaram_baker_2009_range_km(params["period_s"], params["vs30_clustered"])
    return np.exp(-3.0 * distance / b)


class SpatialCorrelationField:
    """epsilon = L z over every site in the call: z iid (optionally truncated) normal, L L' = R."""

    def sample_forward(self, call: Call) -> dict[str, np.ndarray]:
        from scipy.stats import truncnorm

        p = call.params
        lat, lon = call.column("lat")[0], call.column("lon")[0]
        shape = (call.n, len(call.rows))
        t = p["truncation_level"]
        z = (call.rng.standard_normal(shape) if t is None
             else truncnorm.rvs(-t, t, size=shape, random_state=call.rng))
        if p["model"] == "independent":
            call.warnings.append("Ground-motion residuals are drawn independently at each site: no spatial "
                                 "correlation, so joint outcomes across the network are likely understated.")
            return {"epsilon": z}
        # A symmetric root rather than Cholesky: sites at one location have correlation 1,
        # which makes R singular, and the eigendecomposition takes that in its stride.
        values, vectors = np.linalg.eigh(correlation_matrix(lat, lon, p))
        root = vectors * np.sqrt(np.clip(values, 0.0, None))
        return {"epsilon": z @ root.T}
