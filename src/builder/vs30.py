"""Fill missing bridge Vs30 from the USGS Global Vs30 Mosaic, keeping the user's own values.

    python -m src vs30 <network>            fill missing Bridge.vs30 in its network.json
    python -m src vs30 <network> --check    only report which bridges lack it

The mosaic (USGS, 2025 update of Heath et al. 2020; public domain) is a
Cloud-Optimized GeoTIFF of median Vs30 in m/s at 30 arc-seconds (about 900 m),
estimated from topographic slope with regional maps merged in. ScienceBase
refuses automated downloads, so it is fetched once by hand into
``.cache/vs30/`` (or ``BN_VS30_GRID``); only the window around the network is read.

A bridge with a ``vs30`` keeps it. Every value written here says where it came from
in ``vs30_source``, so a report can state how much of the network rests on the
proxy. Cells coded as water or ice (600-603) are replaced by the nearest land cell
within a few cells, and the source says so; beyond that the bridge is refused.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .load import NETWORKS, ROOT, rel

DEFAULT_GRID = ROOT / ".cache" / "vs30" / "vs30_mosaic_median_30c.tif"
SOURCE = "usgs-global-vs30-mosaic"
DOWNLOAD = ("https://www.sciencebase.gov/catalog/item/67be4ac3d34e8876fcbfbd89 "
            "(vs30_mosaic_median_30c.tif, about 945 MB)")
NODATA = {600: "ocean", 601: "ice", 602: "glacier", 603: "lake"}
SEARCH_CELLS = 3                      # how far (in cells, ~0.9 km each) to look for land


class Vs30Error(ValueError):
    pass


def grid_path() -> Path:
    path = Path(os.environ.get("BN_VS30_GRID") or DEFAULT_GRID)
    if not path.is_file():
        raise Vs30Error(f"no Vs30 grid at {path}. Download it once from {DOWNLOAD} into "
                        f"{DEFAULT_GRID.parent} (or set BN_VS30_GRID).")
    return path


def sample(lats: list[float], lons: list[float], path: Path | None = None) -> list[tuple[float, str]]:
    """(Vs30 in m/s, note) at each point: the cell's value, or the nearest land cell's."""
    from osgeo import gdal        # GDAL arrives with OpenQuake

    gdal.UseExceptions()
    ds = gdal.Open(str(path or grid_path()))
    x0, dx, _, y0, _, dy = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    cols = np.floor((np.asarray(lons) - x0) / dx).astype(int)
    rows = np.floor((np.asarray(lats) - y0) / dy).astype(int)
    # One window around every point, plus the search margin.
    c0, c1 = max(cols.min() - SEARCH_CELLS, 0), min(cols.max() + SEARCH_CELLS + 1, ds.RasterXSize)
    r0, r1 = max(rows.min() - SEARCH_CELLS, 0), min(rows.max() + SEARCH_CELLS + 1, ds.RasterYSize)
    window = band.ReadAsArray(int(c0), int(r0), int(c1 - c0), int(r1 - r0)).astype(float)

    out = []
    for r, c in zip(rows - r0, cols - c0):
        value = window[r, c]
        if int(value) not in NODATA:
            out.append((float(value), ""))
            continue
        best = None
        for dr in range(-SEARCH_CELLS, SEARCH_CELLS + 1):
            for dc in range(-SEARCH_CELLS, SEARCH_CELLS + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < window.shape[0] and 0 <= cc < window.shape[1] and int(window[rr, cc]) not in NODATA:
                    d = dr * dr + dc * dc
                    if best is None or d < best[0]:
                        best = (d, float(window[rr, cc]))
        if best is None:
            raise Vs30Error(f"the Vs30 grid has only {NODATA[int(value)]} within {SEARCH_CELLS} cells of "
                            f"({lats[len(out)]}, {lons[len(out)]}); give that bridge a vs30 yourself")
        out.append((best[1], f"; cell is {NODATA[int(value)]}, nearest land cell used"))
    return out


def fill(network_id: str, root: Path = NETWORKS, path: Path | None = None) -> dict[str, Any]:
    """Write vs30 and vs30_source for every bridge that lacks a vs30. Returns what it did."""
    file = root / network_id / "network.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    bridges = data["tables"].get("Bridge", [])
    missing = [b for b in bridges if b.get("vs30") is None]
    kept = len(bridges) - len(missing)
    if missing:
        values = sample([b["lat"] for b in missing], [b["lon"] for b in missing], path)
        for b, (vs30, note) in zip(missing, values):
            b["vs30"] = round(vs30, 1)
            b["vs30_source"] = SOURCE + note
        data["assumptions"]["vs30"] = (
            f"Bridges without a measured value take the USGS Global Vs30 Mosaic's median at their "
            f"location (slope-based proxy, ~900 m cells; large uncertainty). vs30_source says which.")
        file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"network": network_id, "file": rel(file), "filled": len(missing), "kept": kept,
            "near_water": sum("nearest land" in b.get("vs30_source", "") for b in missing)}
