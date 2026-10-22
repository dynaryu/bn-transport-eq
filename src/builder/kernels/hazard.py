"""Ground-motion kernels: everything that produces an IM or its residual.

Each hazard object owns its binding's whole field, every bridge at once, so the
dependence between sites lives inside one object (the residual field's
correlation) and is never an implicit product of independent objects.
"""

from __future__ import annotations

from typing import Any

import torch

from . import Probability

_EARTH_RADIUS_KM = 6371.0


def _t(x: Any) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


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


class Gmm(Probability):
    """ln IM = ln median(Mw, R) + sigma * eps at each bridge, for whichever GMM each row's set is.

    A root: the event is fixed data, not a parent. Residuals are independent between
    bridges, as the Campbell (2003) sets give only a total sigma.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        params = self.spec["parameters"]
        forms = {p["form"] for p in params}
        if len(forms) != 1 or not forms <= GMM_FORMS.keys():
            raise ValueError(f"one implemented GMM form per binding, got {sorted(forms)}")
        if any(p.get("distance_metric", "hypocentral") != "hypocentral" for p in params):
            raise ValueError("only the hypocentral distance metric is implemented")
        event = fixed_event(self.spec["tables"]["Event"])
        mw = self.tensor(event["mw"])
        bounds = self.tensor([p["magnitude_range"] for p in params])
        if ((mw < bounds[:, 0]) | (mw > bounds[:, 1])).any():
            raise ValueError(f"Mw {event['mw']} outside a ground-motion set's magnitude range")
        repi = great_circle_km(event["lat"], event["lon"], self.column("lat"), self.column("lon"))
        r = torch.sqrt(repi**2 + self.tensor(event["depth_km"]) ** 2)          # (1, rows)
        ln_median, aleatory = GMM_FORMS[forms.pop()](mw, r, params)
        self.loc = ln_median.to(self.device)
        self.scale = torch.sqrt(aleatory**2 + self.tensor([p["sigma_epistemic"] for p in params]) ** 2).to(self.device)

    def distribution(self, n: int) -> torch.distributions.LogNormal:
        shape = (n, len(self.childs))
        return torch.distributions.LogNormal(self.loc.expand(shape), self.scale.expand(shape))

    def sample(self, Cs_pars=None, n_sample=None):
        d = self.distribution(n_sample if Cs_pars is None else len(Cs_pars))
        child = d.sample()
        return child, d.log_prob(child).sum(1)

    def log_prob(self, Cs: torch.Tensor) -> torch.Tensor:
        child = Cs.to(device=self.device, dtype=torch.float64)[:, :len(self.childs)]
        valid = (child > 0).all(1) & torch.isfinite(child).all(1)
        logp = self.distribution(len(child)).log_prob(child.clamp_min(torch.finfo(child.dtype).tiny)).sum(1)
        return torch.where(valid, logp, -torch.inf)


class ShakeMap(Probability):
    """ln IM = ln median (grid.xml, e.g. PSA10) + sigma (uncertainty.xml, e.g. STDPSA10) * epsilon."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from .. import shakemap   # imported late: only this kernel fetches data

        # shakemap.load caches the parsed grid, and re-asks which version is current
        # every few minutes, so a long-running server picks up a revised map.
        p = self.spec["parameters"][0]
        grid = shakemap.load(p["event_id"], p.get("source"), p.get("version"), p["imt"])
        rows = self.spec["rows"]
        ln_im, sigma, inside = grid.sample([r["lat"] for r in rows], [r["lon"] for r in rows])
        if not inside.all():
            outside = [row["id"] for row, ok in zip(rows, inside) if not ok]
            raise ValueError(f"{len(outside)} site(s) lie outside ShakeMap {grid.event_id} v{grid.version}, "
                             f"e.g. {outside[:3]}; outside the map means not estimated, not zero")
        self.ln_im, self.sigma = self.tensor(ln_im)[None, :], self.tensor(sigma)[None, :]
        meta = grid.meta
        self.notes.add(f"Ground motion from USGS ShakeMap {grid.event_id} version {grid.version} "
                       f"({meta.get('originator')}, {meta.get('map_status')}, via {meta.get('source')}).")
        if not meta.get("seismic_stations"):
            self.notes.add("This ShakeMap has no seismic station recordings, so its shaking is a ground-motion-model "
                           "estimate (constrained by felt reports at most) with the larger uncertainty that implies.")
        for w in grid.warnings:
            self.notes.add(w)

    def evaluate(self, pars: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.ln_im + self.sigma * self.input(pars, "epsilon"))


def jayaram_baker_2009_range_km(period_s: float, vs30_clustered: bool) -> float:
    """Range b of rho(h) = exp(-3h / b): case 1 (eq. 17) or, with Vs30 clustering, case 2 (eq. 18) below 1 s."""
    if period_s < 1.0:
        return 40.7 - 15.0 * period_s if vs30_clustered else 8.5 + 17.2 * period_s
    return 22.0 + 3.7 * period_s


def correlation_matrix(lat: torch.Tensor, lon: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    if params["model"] == "independent":
        return torch.eye(len(lat), dtype=torch.float64)
    distance = great_circle_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    b = jayaram_baker_2009_range_km(params["period_s"], params["vs30_clustered"])
    return torch.exp(-3.0 * distance / b)


class SpatialCorrelationField(Probability):
    """epsilon = V sqrt(L) z over every site: z iid (optionally truncated) normal, V L V' = R.

    A symmetric root rather than Cholesky: sites at one location have correlation 1,
    which makes R singular, and the eigendecomposition takes that in its stride. The
    log probability is then the density on the field's support: z = L^-1/2 V' epsilon
    over the directions with positive variance.
    """

    _TOL = 1e-10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        params = self.spec["parameters"]
        if any(p != params[0] for p in params):
            raise ValueError("one residual-field set per binding: the field is drawn jointly over its rows")
        self.params = params[0]
        if self.params["model"] == "independent":
            self.notes.add("Ground-motion residuals are drawn independently at each site: no spatial "
                           "correlation, so joint outcomes across the network are likely understated.")
        values, vectors = torch.linalg.eigh(correlation_matrix(self.column("lat")[0], self.column("lon")[0],
                                                               self.params))
        self.support = values > self._TOL
        self.values = values.clamp_min(0.0).to(self.device)
        self.vectors = vectors.to(self.device)
        t = self.params["truncation_level"]
        normal = torch.distributions.Normal(self.tensor(0.0), self.tensor(1.0))
        self.bounds = None if t is None else (normal.cdf(self.tensor(-t)), normal.cdf(self.tensor(t)))

    def _z(self, n: int) -> torch.Tensor:
        shape = (n, len(self.childs))
        if self.bounds is None:
            return torch.randn(shape, dtype=torch.float64, device=self.device)
        lo, hi = self.bounds
        u = lo + (hi - lo) * torch.rand(shape, dtype=torch.float64, device=self.device)
        return torch.special.ndtri(u)

    def sample(self, Cs_pars=None, n_sample=None):
        z = self._z(n_sample if Cs_pars is None else len(Cs_pars))
        epsilon = z @ (self.vectors * self.values.sqrt()).T
        return epsilon, self._log_density(epsilon)

    def _log_density(self, epsilon: torch.Tensor) -> torch.Tensor:
        lam = self.values[self.support]
        z = (epsilon @ self.vectors[:, self.support]) / lam.sqrt()
        logp = -0.5 * z**2 - 0.5 * torch.log(self.tensor(2 * torch.pi))
        if self.bounds is not None:
            t = self.params["truncation_level"]
            logp = torch.where(z.abs() <= t + 1e-9, logp - torch.log(self.bounds[1] - self.bounds[0]), -torch.inf)
        return logp.sum(1) - 0.5 * torch.log(lam).sum()

    def log_prob(self, Cs: torch.Tensor) -> torch.Tensor:
        return self._log_density(Cs.to(device=self.device, dtype=torch.float64)[:, :len(self.childs)])
