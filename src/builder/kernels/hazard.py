"""Ground-motion kernels: everything that produces an IM or its residual.

A field drawn jointly over its sites (``Gmm``, ``SpatialCorrelationField``) is one
object owning every site, so the dependence between sites lives inside one object
and is never an implicit product of independent objects.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributions import LogNormal, Normal

from . import Probability

_EARTH_RADIUS_KM = 6371.0


def _t(x: Any) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def _sites(n: int) -> str:
    return f"{n} site" if n == 1 else f"{n} sites"


def great_circle_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> torch.Tensor:
    lat1, lon1, lat2, lon2 = (torch.deg2rad(_t(v)) for v in (lat1, lon1, lat2, lon2))
    a = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))


def campbell_2003_ln_median(mw: Any, r_km: Any, c: dict[str, Any]) -> torch.Tensor:
    """ln median SA in g, term for term as the EMA dataset script implements it, for the coefficients given.

    Coefficients may be numbers or tensors (one per row), broadcasting with ``mw`` and ``r_km``.
    """
    mw, r = _t(mw), _t(r_km).clamp_min(1e-9)
    c = {k: _t(v) for k, v in c.items()}
    f1 = c["c2"] * mw + c["c3"] * (8.5 - mw) ** 2
    radius = torch.sqrt(r**2 + (c["c7"] * torch.exp(c["c8"] * mw)) ** 2)
    f2 = c["c4"] * torch.log(radius) + (c["c5"] + c["c6"] * mw) * r
    near = c["c7"] * (torch.log(r) - torch.log(_t(70.0)))
    f3 = torch.where(r <= 70.0, 0.0,
                     torch.where(r <= 130.0, near, near + c["c8"] * (torch.log(r) - torch.log(_t(130.0)))))
    return c["c1"] + f1 + f2 + f3


def campbell_2003(mw: torch.Tensor, r_km: torch.Tensor, params: list[dict[str, Any]]):
    """Campbell (2003) form for each row's set: ln median in g, and aleatory sigma (magnitude-dependent)."""
    c = {k: _t([p["coefficients"][k] for p in params]) for k in params[0]["coefficients"]}
    s = {k: _t([p["sigma"][k] for p in params]) for k in params[0]["sigma"]}
    return (campbell_2003_ln_median(mw, r_km, c),
            torch.where(mw < s["m_break"], s["a"] + s["b"] * mw, s["sigma_above"]))


# Functional forms, by the parameter set's "form". A new GMM with an existing form is only
# a parameter set; a new form is a function here plus its branch in schema/functions/hazard.gmm.json.
GMM_FORMS = {
    "campbell-2003": campbell_2003,
}


def fixed_event(table: list[dict[str, Any]]) -> dict[str, Any]:
    """The one Event row, with the fields a GMM needs."""
    if len(table) != 1:
        raise ValueError(f"a GMM needs exactly one Event row, got {len(table)}")
    event = table[0]
    missing = [f for f in ("mw", "lat", "lon", "depth_km") if event.get(f) is None]
    if missing:
        raise ValueError(f"Event {event['id']!r} lacks {', '.join(missing)}, which the GMM needs")
    return event


# ======================================================================
class Gmm(Probability):
    """Ground motion at n sites from one fixed earthquake, by a ground-motion model.

        ln IM_i = ln_median_i + sigma_i * eps_i,    eps_i ~ N(0, 1), independent between sites

    A root: the earthquake is fixed data, not a parent.
    """

    def __init__(self, childs, parents, event, gmm, sites, device="cpu"):
        """
        childs  : [Variable IM_1, ..., Variable IM_n], one per site, in g
        parents : [] -- this node does not have parents: the EQ scenario is a fixed parameter
        event   : the earthquake, {"id", "mw", "lat", "lon", "depth_km"}
        gmm     : the ground-motion model, one parameter set for every site: its "form" (a key of
                  GMM_FORMS), "coefficients", aleatory "sigma", "sigma_epistemic",
                  "magnitude_range" and "distance_metric"
        sites   : [{"id", "lat", "lon"}], one per child, in the same order
        """
        super().__init__(childs, parents, device)
        assert not parents, "Gmm is a root: the EQ scenario is a fixed parameter, not a parent"
        assert len(sites) == len(childs), "one site per child"
        if gmm["form"] not in GMM_FORMS:
            raise ValueError(f"GMM form {gmm['form']!r} is not implemented; have {sorted(GMM_FORMS)}")
        if gmm.get("distance_metric", "hypocentral") != "hypocentral":
            raise ValueError("only the hypocentral distance metric is implemented")
        low, high = gmm["magnitude_range"]
        if not low <= event["mw"] <= high:
            raise ValueError(f"Mw {event['mw']} outside the ground-motion set's magnitude range [{low}, {high}]")
        self.event = dict(event)
        self.gmm_form = gmm["form"]

        # Hypocentral distance from the event to each site, in km.
        lat, lon = _t([s["lat"] for s in sites]), _t([s["lon"] for s in sites])
        r = torch.sqrt(great_circle_km(event["lat"], event["lon"], lat, lon) ** 2 + _t(event["depth_km"]) ** 2)

        # GMM parameters: the ln median and total sigma at each site.
        ln_median, sigma_aleatory = GMM_FORMS[gmm["form"]](_t(event["mw"]), r, [gmm])
        sigma = torch.sqrt(sigma_aleatory**2 + _t(gmm["sigma_epistemic"]) ** 2)
        self.ln_median = self.tensor(ln_median.expand(len(childs)))          # (n,)
        self.sigma = self.tensor(sigma.expand(len(childs)))                  # (n,)
        self.notes.add("hazard.gmm draws each Bridge's residual independently: i.e. no spatial correlation")

    # ------------------------------------------------------------------
    def form(self):
        sigma = (f"{self.sigma[0].item():.3g}" if torch.allclose(self.sigma, self.sigma[0]) else
                 f"{self.sigma.min().item():.3g} to {self.sigma.max().item():.3g}")
        return (f"lognormal at {_sites(len(self.childs))}: ln IM = ln median + sigma * eps, sigma {sigma}, "
                f"eps independent{self.parameter_sets()}")

    def form_details(self):
        e = self.event
        return [f"ln median by {self.gmm_form} for event {e['id']} (Mw {e['mw']:g}, {e['depth_km']:g} km deep, "
                f"at ({e['lat']:g}, {e['lon']:g}); hypocentral distance)"] + super().form_details()

    # ------------------------------------------------------------------
    def distribution(self, n_sample: int) -> LogNormal:
        """IM at each site, for n_sample samples: LogNormal of shape (n_sample, n) (the EvaluateDensity capability)."""
        shape = (n_sample, len(self.childs))
        return LogNormal(self.ln_median.expand(shape), self.sigma.expand(shape))

    def sample(self, Cs_pars=None, n_sample=None):
        """
        n_sample : number of samples (a root: no Cs_pars)

        Returns
        -------
        Cs   : (n_sample, n) IM at each site
        logp : (n_sample,) log density of each sample, summed over the sites
        """
        if n_sample is None:
            n_sample = len(Cs_pars)
        dist = self.distribution(n_sample)

        Cs = dist.sample()
        logp = dist.log_prob(Cs).sum(1)
        return Cs, logp

    def log_prob(self, Cs):
        """
        Cs : (N, n) IM at each site (no parent columns: a root)

        Returns (N,) log density, summed over the sites; -inf where an IM is not positive and finite
        """
        Cs = Cs.to(device=self.device, dtype=torch.float64)[:, :len(self.childs)]
        dist = self.distribution(len(Cs))

        valid = (Cs > 0).all(1) & torch.isfinite(Cs).all(1)
        logp = dist.log_prob(Cs.clamp_min(torch.finfo(Cs.dtype).tiny)).sum(1)
        return torch.where(valid, logp, -torch.inf)


# ======================================================================
class ShakeMap(Probability):
    """Ground motion at one site from a USGS ShakeMap, given the site's residual eps (deterministic).

        ln IM = ln_median + sigma * eps

    ln_median is the map's median at the site (grid.xml, e.g. PSA10) and sigma its uncertainty
    (uncertainty.xml, e.g. STDPSA10). The residual eps is drawn for every site jointly by a
    SpatialCorrelationField.

    A site off the map is given the map's value at the nearest point of its edge, sigma included. USGS sizes
    a map to reach where the shaking it predicts has faded to light (about MMI IV), and shaking keeps falling
    with distance, so that is an upper bound on the site's shaking, not zero and not a guess at it. A map that
    reaches none of the sites (``sites``, every site of the binding) is refused: it is some other region's.
    """

    def __init__(self, childs, parents, event_id, imt, lat, lon, source=None, version=None, sites=None,
                 device="cpu"):
        """
        childs   : [Variable IM], in g
        parents  : [Variable eps], the site's standardised residual
        event_id : the ShakeMap's USGS event id
        imt      : the intensity measure, e.g. "SA(1.0)"
        lat, lon : the site, in degrees
        source   : where to fetch the map ("shakecast" or "comcat"); None for shakemap.load's default
        version  : the map version; None for the current one
        sites    : [[lat, lon], ...] of every site on this map, to refuse a map that reaches none of them
        """
        from .. import shakemap   # imported late: only this kernel fetches data

        super().__init__(childs, parents, device)
        assert len(childs) == 1 and len(parents) == 1, "ShakeMap: one IM child, one eps parent"

        # shakemap.load caches the parsed grid, and re-asks which version is current
        # every few minutes, so a long-running server picks up a revised map.
        grid = shakemap.load(event_id, source, version, imt)
        ln_median, sigma, inside = grid.sample([lat], [lon])
        self.outside_km = 0.0
        if not inside[0]:
            if sites is not None and not grid.sample([s[0] for s in sites], [s[1] for s in sites])[2].any():
                raise ValueError(f"ShakeMap {grid.event_id} v{grid.version} reaches none of the {len(sites)} sites; "
                                 f"it maps another region's shaking")
            edge_lat, edge_lon = grid.nearest_on_map([lat], [lon])
            ln_median, sigma, _ = grid.sample(edge_lat, edge_lon)
            self.outside_km = float(great_circle_km(lat, lon, float(edge_lat[0]), float(edge_lon[0])))
            self.notes.add(f"{childs[0].name}'s site lies {self.outside_km:.0f} km off ShakeMap {grid.event_id} "
                           f"v{grid.version}: it is given the map's value at the nearest edge, an upper bound, since "
                           f"USGS maps reach to where shaking has faded to light and it falls further with distance.")
        self.ln_median = float(ln_median[0])
        self.sigma = float(sigma[0])
        self.source = f"ShakeMap {grid.event_id} v{grid.version}, {imt}"

        meta = grid.meta
        self.notes.add(f"Ground motion from USGS ShakeMap {grid.event_id} version {grid.version} "
                       f"({meta.get('originator')}, {meta.get('map_status')}, via {meta.get('source')}).")
        if not meta.get("seismic_stations"):
            self.notes.add("This ShakeMap has no seismic station recordings, so its shaking is a ground-motion-model "
                           "estimate (constrained by felt reports at most) with the larger uncertainty that implies.")
        for warning in grid.warnings:
            self.notes.add(warning)

    def form(self):
        eps = self.parents[0].name
        return (f"ln IM = {self.ln_median:.4g} + {self.sigma:.3g} * {eps}, from {self.source}")

    def evaluate(self, Cs_pars):
        """
        Cs_pars : (N, 1) eps
        Returns (N, 1) IM
        """
        eps = Cs_pars[:, :1]
        return torch.exp(self.ln_median + self.sigma * eps)


# ======================================================================
def jayaram_baker_2009_range_km(period_s: float, vs30_clustered: bool) -> float:
    """Range b of rho(h) = exp(-3h / b): case 1 (eq. 17) or, with Vs30 clustering, case 2 (eq. 18) below 1 s."""
    if period_s < 1.0:
        return 40.7 - 15.0 * period_s if vs30_clustered else 8.5 + 17.2 * period_s
    return 22.0 + 3.7 * period_s


def correlation_matrix(lat: torch.Tensor, lon: torch.Tensor, range_km: float | None) -> torch.Tensor:
    """R_ij = exp(-3 h_ij / range_km), h_ij the distance between sites i and j; the identity without a range."""
    if range_km is None:
        return torch.eye(len(lat), dtype=torch.float64)
    distance = great_circle_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return torch.exp(-3.0 * distance / range_km)


class SpatialCorrelationField(Probability):
    """Standardised ground-motion residuals eps at n sites, drawn jointly.

        eps = V sqrt(L) z,    z_i ~ N(0, 1) iid, optionally truncated at +/- truncation
        V L V' = R,           R_ij = exp(-3 h_ij / range_km)       (Jayaram & Baker 2009)

    h_ij is the distance between sites i and j; without range_km the sites are independent
    (R = I). A symmetric root rather than Cholesky: co-located sites have correlation 1,
    which makes R singular, and the eigendecomposition takes that in its stride. The log
    probability is the density on the field's support, z = L^-1/2 V' eps over the directions
    with positive variance.
    """

    _TOL = 1e-10

    def __init__(self, childs, parents, sites, correlation, device="cpu"):
        """
        childs      : [Variable eps_1, ..., Variable eps_n], one per site
        parents     : [] (a root)
        sites       : [{"id", "lat", "lon"}], one per child, in the same order
        correlation : the correlation model's parameter set: "model" ("independent", or
                      "jayaram-baker-2009" with "period_s" and "vs30_clustered", which set the
                      range), and "truncation_level" (null for none)
        """
        super().__init__(childs, parents, device)
        assert not parents and len(sites) == len(childs), "a root, with one site per child"
        self.source = correlation["model"]
        self.truncation = correlation["truncation_level"]
        self.range_km = (None if correlation["model"] == "independent" else
                         jayaram_baker_2009_range_km(correlation["period_s"], correlation["vs30_clustered"]))
        if self.range_km is None:
            self.notes.add("Ground-motion residuals are drawn independently at each site: no spatial "
                           "correlation, so joint outcomes across the network are likely understated.")

        lat, lon = _t([s["lat"] for s in sites]), _t([s["lon"] for s in sites])
        values, vectors = torch.linalg.eigh(correlation_matrix(lat, lon, self.range_km))
        self.support = values > self._TOL                                    # directions with positive variance
        self.values = values.clamp_min(0.0).to(self.device)                  # L, (n,)
        self.vectors = vectors.to(self.device)                               # V, (n, n)

        normal = Normal(self.tensor(0.0), self.tensor(1.0))
        t = self.truncation
        self.bounds = None if t is None else (normal.cdf(self.tensor(-t)), normal.cdf(self.tensor(t)))

    # ------------------------------------------------------------------
    def form(self):
        n = _sites(len(self.childs))
        truncated = "" if self.truncation is None else f", truncated at +/-{self.truncation:g}"
        if self.range_km is None:
            return f"independent standard normals at {n}{truncated}{self.parameter_sets()}"
        return (f"standard normals at {n}, correlated by {self.source}: "
                f"rho(h) = exp(-3h / {self.range_km:.3g} km){truncated}{self.parameter_sets()}")

    # ------------------------------------------------------------------
    def _z(self, n_sample: int) -> torch.Tensor:
        """(n_sample, n) independent standard normals, truncated if asked."""
        shape = (n_sample, len(self.childs))
        if self.bounds is None:
            return torch.randn(shape, dtype=torch.float64, device=self.device)
        lo, hi = self.bounds
        u = lo + (hi - lo) * torch.rand(shape, dtype=torch.float64, device=self.device)
        return torch.special.ndtri(u)

    def sample(self, Cs_pars=None, n_sample=None):
        """
        n_sample : number of samples (a root: no Cs_pars)

        Returns
        -------
        Cs   : (n_sample, n) eps at each site
        logp : (n_sample,) log density of each sample
        """
        if n_sample is None:
            n_sample = len(Cs_pars)
        z = self._z(n_sample)
        eps = z @ (self.vectors * self.values.sqrt()).T
        return eps, self.log_prob(eps)

    def log_prob(self, Cs):
        """
        Cs : (N, n) eps at each site (no parent columns: a root)

        Returns (N,) log density on the field's support; -inf outside a truncation
        """
        eps = Cs.to(device=self.device, dtype=torch.float64)[:, :len(self.childs)]
        lam = self.values[self.support]
        z = (eps @ self.vectors[:, self.support]) / lam.sqrt()
        logp = -0.5 * z**2 - 0.5 * torch.log(self.tensor(2 * torch.pi))
        if self.bounds is not None:
            logp = torch.where(z.abs() <= self.truncation + 1e-9,
                               logp - torch.log(self.bounds[1] - self.bounds[0]), -torch.inf)
        return logp.sum(1) - 0.5 * torch.log(lam).sum()
