"""USGS ShakeMap as a hazard input: find events, fetch grids, sample at sites.

A ShakeMap is the ground motion an earthquake actually produced, estimated on a
regular latitude/longitude grid from recordings, felt reports and a ground-motion
model. Its ``grid.xml`` carries the median PSA at 1 s (``PSA10``, in %g) and its
``uncertainty.xml`` the log standard deviation (``STDPSA10``, in ln units) at each
grid point. Those two numbers are exactly what the pilot's ``SA1`` node needs, so
a ShakeMap replaces the scenario-plus-GMPE path rather than adding to it.

Two sources, one reader:

* ``ComcatSource`` -- the USGS earthquake catalogue (ComCat). Searches events
  that have a ShakeMap and downloads the preferred one. Downloads are cached on
  disk in the same layout ShakeCast uses, so a later run needs no network.
* ``ShakeCastSource`` -- the data directory of a running USGS ShakeCast server
  (v4). ShakeCast already polls ComCat and saves every ShakeMap version as
  ``<data>/<event_id>/<event_id>-<version>/grid.xml`` (and ``uncertainty.xml``
  when its ``eq_pref_products`` include it, as the default configuration does).
  Reading that folder means the pilot follows whatever ShakeCast has received,
  with no second poller. ShakeCast's own root is ``SHAKECAST_USER_DIRECTORY``,
  else ``~/.shakecast``.

Standard library and numpy only.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

COMCAT_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USER_AGENT = "bn-on-demand-pilot (forward BN pilot; ShakeMap hazard input)"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = Path(
    os.environ.get("FORWARD_PILOT_SHAKEMAP_CACHE")
    or ROOT / "experiments" / "forward-bn-pilot" / "shakemaps"
)

# grid.xml unit strings for spectral acceleration, by ShakeMap generation.
_TO_G = {"%g": 0.01, "pctg": 0.01, "g": 1.0}


class ShakeMapError(ValueError):
    """A ShakeMap could not be found, fetched or read."""


class UnknownEvent(ShakeMapError):
    """The catalogue has no event with that id."""


# ---- the grid --------------------------------------------------------------


@dataclass(frozen=True)
class ShakeMapGrid:
    """PSA(1.0 s) on a regular grid, rows north to south, columns west to east."""

    event_id: str
    version: str
    meta: dict[str, Any]
    lon_min: float
    lat_max: float
    dlon: float
    dlat: float
    ln_psa10: np.ndarray                     # (nlat, nlon), ln of SA in g
    std_psa10: np.ndarray | None = None      # (nlat, nlon), ln units; None if no uncertainty.xml
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.ln_psa10.shape

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(lon_min, lat_min, lon_max, lat_max) of the grid points."""
        nlat, nlon = self.shape
        return (self.lon_min, self.lat_max - (nlat - 1) * self.dlat,
                self.lon_min + (nlon - 1) * self.dlon, self.lat_max)

    def sample(self, lats: Sequence[float], lons: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear (ln median SA1 in g, sigma, inside-grid flag) at each site.

        Interpolation is done on ln SA, the scale ShakeMap's medians and
        uncertainties live on. A site outside the grid gets NaN and ``False``;
        the caller decides what that means rather than this method guessing.
        """
        lats, lons = np.asarray(lats, float), np.asarray(lons, float)
        nlat, nlon = self.shape
        row = (self.lat_max - lats) / self.dlat
        col = (lons - self.lon_min) / self.dlon
        inside = (row >= 0) & (row <= nlat - 1) & (col >= 0) & (col <= nlon - 1)
        r0 = np.clip(np.floor(row).astype(int), 0, max(nlat - 2, 0))
        c0 = np.clip(np.floor(col).astype(int), 0, max(nlon - 2, 0))
        fr, fc = np.clip(row - r0, 0, 1), np.clip(col - c0, 0, 1)

        def interp(field_: np.ndarray) -> np.ndarray:
            r1, c1 = np.minimum(r0 + 1, nlat - 1), np.minimum(c0 + 1, nlon - 1)
            top = field_[r0, c0] * (1 - fc) + field_[r0, c1] * fc
            bottom = field_[r1, c0] * (1 - fc) + field_[r1, c1] * fc
            return top * (1 - fr) + bottom * fr

        ln_sa = np.where(inside, interp(self.ln_psa10), np.nan)
        sigma = (np.where(inside, interp(self.std_psa10), np.nan)
                 if self.std_psa10 is not None else np.where(inside, 0.0, np.nan))
        return ln_sa, sigma, inside


def _parse_grid_xml(path: Path, wanted: str) -> tuple[dict[str, Any], dict[str, float], np.ndarray, str]:
    """Header attributes, grid spec, the ``wanted`` field as (nlat, nlon), and its units.

    The header is parsed as XML; the data block, which is most of a ~30 MB file,
    is read as one whitespace-separated array, which is far faster than parsing
    it element by element.
    """
    text = path.read_text(encoding="utf-8")
    start, end = text.find("<grid_data>"), text.find("</grid_data>")
    if start < 0 or end < 0:
        raise ShakeMapError(f"{path} is not a ShakeMap grid: no <grid_data> block")
    header = ET.fromstring(text[:start] + "</shakemap_grid>")
    strip = lambda tag: tag.rsplit("}", 1)[-1]          # drop the XML namespace
    root_attrs = dict(header.attrib)
    event_attrs, spec, fields = {}, {}, {}
    for child in header:
        name = strip(child.tag)
        if name == "event":
            event_attrs = dict(child.attrib)
        elif name == "grid_specification":
            spec = {k: float(v) for k, v in child.attrib.items()}
        elif name == "grid_field":
            fields[child.attrib["name"].upper()] = (int(child.attrib["index"]) - 1, child.attrib.get("units", ""))
    if wanted not in fields:
        raise ShakeMapError(f"{path.name} has no {wanted} field; it carries {sorted(fields)}")

    data = np.array(text[start + len("<grid_data>"):end].split(), dtype=float)
    nlat, nlon = int(spec["nlat"]), int(spec["nlon"])
    if data.size != nlat * nlon * len(fields):
        raise ShakeMapError(f"{path.name}: expected {nlat}x{nlon} rows of {len(fields)} values, got {data.size} values")
    table = data.reshape(nlat * nlon, len(fields))

    # Do not trust the row order to a convention: check it against the LAT/LON columns.
    lat, lon = table[:, fields["LAT"][0]], table[:, fields["LON"][0]]
    if not (np.isclose(lat[0], spec["lat_max"], atol=1e-3) and np.isclose(lon[0], spec["lon_min"], atol=1e-3)
            and lon[1] > lon[0] and lat[nlon] < lat[0]):
        raise ShakeMapError(f"{path.name}: grid rows are not ordered north-west first as expected")
    index, units = fields[wanted]
    meta = {**root_attrs, "event": event_attrs, "grid": spec}
    return meta, spec, table[:, index].reshape(nlat, nlon), units


def read_grid(directory: Path | str) -> ShakeMapGrid:
    """Read ``grid.xml`` (required) and ``uncertainty.xml`` (if present) from one folder."""
    directory = Path(directory)
    grid_path = directory / "grid.xml"
    if not grid_path.is_file():
        raise ShakeMapError(f"no grid.xml in {directory}")
    meta, spec, psa, units = _parse_grid_xml(grid_path, "PSA10")
    scale = _TO_G.get(units.strip().lower())
    if scale is None:
        raise ShakeMapError(f"grid.xml gives PSA10 in {units!r}; expected %g, pctg or g")
    # Zero shaking is legal in a grid; floor it so the log stays finite and tiny.
    ln_psa = np.log(np.maximum(psa * scale, 1e-6))

    warnings: list[str] = []
    std = None
    unc_path = directory / "uncertainty.xml"
    if unc_path.is_file():
        _, unc_spec, std, _ = _parse_grid_xml(unc_path, "STDPSA10")
        if (unc_spec["nlat"], unc_spec["nlon"]) != (spec["nlat"], spec["nlon"]):
            raise ShakeMapError("uncertainty.xml and grid.xml are on different grids")
    else:
        warnings.append(
            "This ShakeMap has no uncertainty.xml, so ground-motion variability is omitted "
            "and bridge failure probabilities are understated."
        )

    event = meta.get("event", {})
    return ShakeMapGrid(
        event_id=meta.get("event_id") or event.get("event_id", directory.parent.name),
        version=str(meta.get("shakemap_version", "")),
        meta={
            "event_id": meta.get("event_id"),
            "version": meta.get("shakemap_version"),
            "originator": meta.get("shakemap_originator"),
            "map_status": meta.get("map_status"),
            "process_timestamp": meta.get("process_timestamp"),
            "magnitude": _float(event.get("magnitude")),
            "depth_km": _float(event.get("depth")),
            "lat": _float(event.get("lat")),
            "lon": _float(event.get("lon")),
            "time": event.get("event_timestamp"),
            "description": event.get("event_description"),
            "seismic_stations": _int(event.get("seismic_stations")),
            "intensity_observations": _int(event.get("intensity_observations")),
            "directory": str(directory),
        },
        # Spacing from the bounds, not the "nominal" attributes: those are rounded
        # (0.0167 for 1/60 degree), which drifts ~2 km across a 600-row grid.
        lon_min=spec["lon_min"], lat_max=spec["lat_max"],
        dlon=(spec["lon_max"] - spec["lon_min"]) / (spec["nlon"] - 1),
        dlat=(spec["lat_max"] - spec["lat_min"]) / (spec["nlat"] - 1),
        ln_psa10=ln_psa, std_psa10=std, warnings=warnings,
    )


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    return None if number is None else int(number)


# ---- where grids come from ---------------------------------------------------


def _version_dirs(event_dir: Path) -> list[tuple[int, Path]]:
    """``<id>-<version>`` folders holding a grid.xml, oldest first (ShakeCast layout)."""
    found = []
    for child in event_dir.iterdir() if event_dir.is_dir() else []:
        match = re.fullmatch(re.escape(event_dir.name) + r"-(\d+)", child.name)
        if match and (child / "grid.xml").is_file():
            found.append((int(match.group(1)), child))
    return sorted(found)


class ShakeCastSource:
    """ShakeMaps a ShakeCast server has already downloaded."""

    name = "shakecast"

    def __init__(self, root: Path | str | None = None):
        user_dir = Path(root or os.environ.get("SHAKECAST_USER_DIRECTORY") or Path.home() / ".shakecast")
        # Accept either ShakeCast's user directory or its data/ folder.
        self.data_dir = user_dir / "data" if (user_dir / "data").is_dir() else user_dir

    def events(self) -> list[dict[str, Any]]:
        """Every event with at least one grid, newest version of each."""
        if not self.data_dir.is_dir():
            raise ShakeMapError(f"no ShakeCast data directory at {self.data_dir}")
        out = []
        for event_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            versions = _version_dirs(event_dir)
            if versions:
                out.append(_summary_from_grid_header(versions[-1][1] / "grid.xml", event_dir.name))
        return out

    def directory(self, event_id: str, version: int | None = None) -> Path:
        versions = _version_dirs(self.data_dir / event_id)
        if not versions:
            raise ShakeMapError(f"ShakeCast has no ShakeMap for {event_id!r} in {self.data_dir}")
        if version is None:
            return versions[-1][1]
        for number, path in versions:
            if number == int(version):
                return path
        raise ShakeMapError(f"ShakeCast has {event_id!r} versions {[n for n, _ in versions]}, not {version}")


def _summary_from_grid_header(path: Path, event_id: str) -> dict[str, Any]:
    head = path.read_text(encoding="utf-8")[:4000]
    attr = lambda name: (re.search(rf'\b{name}="([^"]*)"', head) or [None, None])[1]
    return {"event_id": event_id, "version": attr("shakemap_version"), "magnitude": _float(attr("magnitude")),
            "time": attr("event_timestamp"), "place": attr("event_description"),
            "lat": _float(attr("lat")), "lon": _float(attr("lon")), "source": "shakecast"}


class ComcatSource:
    """The USGS earthquake catalogue, with downloads cached in ShakeCast's layout."""

    name = "comcat"

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE, timeout: float = 60):
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

    def _get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{COMCAT_QUERY}?{urlencode(params)}"
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # The catalogue answers an unknown event id with 404: that is "no such
            # event", not an outage, and saying otherwise sends a reader astray.
            if exc.code == 404 and "eventid" in params:
                raise UnknownEvent(f"USGS has no event with id {params['eventid']!r}") from exc
            raise ShakeMapError(f"the USGS catalogue refused the request: {exc}") from exc
        except OSError as exc:
            raise ShakeMapError(f"could not reach the USGS catalogue: {exc}") from exc

    def search(
        self,
        bbox: tuple[float, float, float, float],
        start: str = "1900-01-01",
        end: str | None = None,
        min_magnitude: float | None = None,
    ) -> list[dict[str, Any]]:
        """Events with a ShakeMap inside (lon_min, lat_min, lon_max, lat_max), newest first."""
        lon_min, lat_min, lon_max, lat_max = bbox
        params: dict[str, Any] = {
            "format": "geojson", "producttype": "shakemap", "orderby": "time",
            "minlatitude": lat_min, "maxlatitude": lat_max,
            "minlongitude": lon_min, "maxlongitude": lon_max, "starttime": start,
        }
        if end:
            params["endtime"] = end
        if min_magnitude is not None:
            params["minmagnitude"] = min_magnitude
        events = []
        for feature in self._get_json(params).get("features", []):
            props = feature["properties"]
            lon, lat, depth = feature["geometry"]["coordinates"]
            events.append({
                "event_id": feature["id"], "magnitude": props.get("mag"),
                "time": datetime.fromtimestamp(props["time"] / 1000, timezone.utc).isoformat(timespec="seconds"),
                "place": props.get("place"), "lat": lat, "lon": lon, "depth_km": depth,
                "mmi": props.get("mmi"), "source": "comcat",
            })
        return events

    def directory(self, event_id: str, version: int | None = None) -> Path:
        """Folder holding the event's preferred ShakeMap, downloading it if not cached.

        With no ``version``, the catalogue is asked which version is current, so
        a revised ShakeMap replaces an older cached one. If the catalogue cannot
        be reached, the newest cached version is used instead.
        """
        cached = _version_dirs(self.cache_dir / event_id)
        if version is not None:
            for number, path in cached:
                if number == int(version):
                    return path
        try:
            detail = self._get_json({"eventid": event_id, "format": "geojson"})
        except UnknownEvent:
            raise
        except ShakeMapError:
            if cached and version is None:
                return cached[-1][1]
            raise
        products = detail.get("properties", {}).get("products", {}).get("shakemap") or []
        if not products:
            raise ShakeMapError(f"USGS has no ShakeMap for event {event_id!r}")
        product = products[0]            # ComCat lists the preferred product first
        current = int(float(product.get("properties", {}).get("version", 1)))
        if version is not None and int(version) != current:
            raise ShakeMapError(f"USGS serves version {current} of {event_id!r}; version {version} is not cached")
        target = self.cache_dir / event_id / f"{event_id}-{current}"
        if (target / "grid.xml").is_file():
            return target

        contents = product.get("contents", {})
        if "download/grid.xml" not in contents:
            raise ShakeMapError(f"the ShakeMap for {event_id!r} has no grid.xml")
        target.mkdir(parents=True, exist_ok=True)
        for name in ("grid.xml", "uncertainty.xml"):
            entry = contents.get(f"download/{name}")
            if entry:
                self._download(entry["url"], target / name)
        (target / "info.json").write_text(json.dumps({
            "event_id": event_id, "version": current, "source": product.get("source"),
            "code": product.get("code"), "update_time": product.get("updateTime"),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2) + "\n", encoding="utf-8")
        return target

    def _download(self, url: str, path: Path) -> None:
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=self.timeout) as response:
                partial.write_bytes(response.read())
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise ShakeMapError(f"could not download {url}: {exc}") from exc
        partial.replace(path)            # never leave a truncated grid where a reader would find it


def get_source(name: str | None = None, **kwargs: Any) -> ComcatSource | ShakeCastSource:
    """``comcat`` (default) or ``shakecast``; ``FORWARD_PILOT_SHAKEMAP_SOURCE`` overrides."""
    chosen = (name or os.environ.get("FORWARD_PILOT_SHAKEMAP_SOURCE") or "comcat").strip().lower()
    if chosen == "comcat":
        return ComcatSource(**kwargs)
    if chosen == "shakecast":
        return ShakeCastSource(**kwargs)
    raise ShakeMapError(f"unknown ShakeMap source {chosen!r}; choose comcat or shakecast")


@lru_cache(maxsize=8)
def _load_cached(directory: str) -> ShakeMapGrid:
    return read_grid(directory)


_RESOLVE_TTL = 300.0     # seconds; a ShakeMap revision is minutes apart at the fastest
_resolved: dict[tuple[str, str, int | None], tuple[float, Path]] = {}


def load(event_id: str, source: str | None = None, version: int | None = None) -> ShakeMapGrid:
    """The ShakeMap grid for one event, parsed once per process per folder.

    Which folder is current is remembered for a few minutes, so one query does
    not ask the catalogue twice, while a revised ShakeMap is still picked up.
    """
    if not event_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(event_id)):
        raise ShakeMapError(f"{event_id!r} is not a USGS event id")
    src = get_source(source)
    key = (src.name, str(event_id), None if version is None else int(version))
    now = time.monotonic()
    hit = _resolved.get(key)
    if hit is None or now - hit[0] > _RESOLVE_TTL:
        hit = (now, src.directory(str(event_id), version))
        _resolved[key] = hit
    return _load_cached(str(hit[1]))


# ---- the network side ----------------------------------------------------------


def edge_sites(network: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """(lat, lon) of each edge's bridge: the midpoint of its end nodes.

    The packages carry one notional bridge per edge with no surveyed location, so
    the midpoint is a stand-in, and a package without node lat/lon cannot be
    used with a ShakeMap at all.
    """
    nodes = network["nodes"]
    missing = sorted(n for n, v in nodes.items() if v.get("lat") is None or v.get("lon") is None)
    if missing:
        raise ShakeMapError(
            f"this package has no geodetic coordinates for nodes {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}, so a ShakeMap cannot be sampled at its bridges"
        )
    sites = {}
    for edge_id, edge in network["edges"].items():
        a, b = nodes[edge["from"]], nodes[edge["to"]]
        sites[edge_id] = ((float(a["lat"]) + float(b["lat"])) / 2, (float(a["lon"]) + float(b["lon"])) / 2)
    return sites


def network_bbox(network: dict[str, Any], margin_deg: float = 1.0) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) around the network's nodes."""
    lats = [float(v["lat"]) for v in network["nodes"].values() if v.get("lat") is not None]
    lons = [float(v["lon"]) for v in network["nodes"].values() if v.get("lon") is not None]
    if not lats:
        raise ShakeMapError("this package has no geodetic coordinates")
    return (min(lons) - margin_deg, min(lats) - margin_deg, max(lons) + margin_deg, max(lats) + margin_deg)


def search_events(
    network: dict[str, Any],
    source: str | None = None,
    *,
    start: str = "1900-01-01",
    end: str | None = None,
    min_magnitude: float | None = None,
) -> list[dict[str, Any]]:
    """Events with a ShakeMap around a network, newest first, from either source.

    ShakeCast keeps only what it has received, so its list is filtered here by
    the same box the catalogue search uses.
    """
    bbox = network_bbox(network)
    src = get_source(source)
    if isinstance(src, ComcatSource):
        return src.search(bbox, start=start, end=end, min_magnitude=min_magnitude)
    lon_min, lat_min, lon_max, lat_max = bbox
    events = [
        e for e in src.events()
        if e.get("lat") is not None and lat_min <= e["lat"] <= lat_max and lon_min <= e["lon"] <= lon_max
        and (min_magnitude is None or (e.get("magnitude") or 0) >= min_magnitude)
    ]
    return sorted(events, key=lambda e: e.get("time") or "", reverse=True)


def site_terms(network: dict[str, Any], hazard: dict[str, Any]) -> tuple[dict[str, tuple[float, float]], list[str], dict[str, Any]]:
    """Per-edge (ln median SA1 in g, ground-motion sigma), warnings, and the map's metadata.

    The shared step for both backends: the analytic engine folds the fragility
    dispersion in afterwards, the tbnpy model hands these to its SA1 roots.
    A bridge outside the ShakeMap's extent is refused rather than given zero
    shaking: the extent is drawn where shaking matters, but "outside" means
    "not estimated", and a number nobody estimated should not enter a result.
    """
    grid = load(hazard.get("event_id", ""), hazard.get("source"), hazard.get("version"))
    sites = edge_sites(network)
    ids = list(sites)
    ln_sa, sigma, inside = grid.sample([sites[e][0] for e in ids], [sites[e][1] for e in ids])
    if not inside.all():
        outside = [e for e, ok in zip(ids, inside) if not ok]
        raise ShakeMapError(
            f"{len(outside)} bridge(s) lie outside the ShakeMap for {grid.event_id} "
            f"(extent {tuple(round(v, 2) for v in grid.extent)}), e.g. {outside[:3]}"
        )
    terms = {e: (float(m), float(s)) for e, m, s in zip(ids, ln_sa, sigma)}

    meta = grid.meta
    warnings = list(grid.warnings)
    warnings.append(
        f"Ground motion from USGS ShakeMap {grid.event_id} version {grid.version} "
        f"({meta.get('originator')}, {meta.get('map_status')}), sampled at arc midpoints; "
        "bridge locations are not surveyed."
    )
    if not meta.get("seismic_stations"):
        warnings.append(
            "This ShakeMap has no seismic station recordings, so its shaking is a ground-motion-model "
            "estimate (constrained by felt reports at most) and carries the larger uncertainty that implies."
        )
    warnings.append(
        "ShakeMap uncertainty is applied per bridge independently; the spatial correlation of "
        "ground-motion residuals is not represented."
    )
    return terms, warnings, meta
