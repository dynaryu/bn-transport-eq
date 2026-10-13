"""Tests for the ShakeMap hazard input.

No test touches the network. Grids are written in the real ``grid.xml`` /
``uncertainty.xml`` format, into a ShakeCast-style data directory, with a shaking
field whose logarithm is linear in latitude and longitude -- so bilinear
interpolation on ln SA is exact and a sampled value can be checked to the digit.
"""

import json
from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.special import ndtr

from src.forward_pilot import chat, shakemap
from src.forward_pilot.catalog import load_case
from src.forward_pilot.engine import ForwardQueryError, run_forward_query
from src.forward_pilot.hazard import _median_extensive

NS = 'xmlns="http://earthquake.usgs.gov/eqcenter/shakemap"'


def ln_sa_field(lat, lon):
    """ln SA1 (g): strongest in the south-west, linear so interpolation is exact."""
    return log(0.4) - 0.3 * (lat - 37.0) - 0.2 * (lon - 115.0)


def write_grid(folder, event_id="test001", version=1, extent=(112.0, 35.0, 121.0, 43.0),
               step=0.5, sigma=0.6, units="%g", with_uncertainty=True, reverse_rows=False):
    lon_min, lat_min, lon_max, lat_max = extent
    lons = np.arange(lon_min, lon_max + step / 2, step)
    lats = np.arange(lat_max, lat_min - step / 2, -step)          # north to south, as ShakeMap writes
    if reverse_rows:
        lats = lats[::-1]
    scale = 100.0 if units in ("%g", "pctg") else 1.0
    head = (f'<?xml version="1.0" encoding="UTF-8"?><shakemap_grid {NS} event_id="{event_id}" '
            f'shakemap_id="{event_id}" shakemap_version="{version}" shakemap_originator="test" '
            f'map_status="RELEASED">\n<event event_id="{event_id}" magnitude="6.8" depth="20" lat="37.5" '
            f'lon="115.0" event_timestamp="1966-03-22T08:19:36" event_description="Test, China" '
            f'seismic_stations="0" intensity_observations="0" />\n'
            f'<grid_specification lon_min="{lon_min}" lat_min="{lat_min}" lon_max="{lon_max}" '
            f'lat_max="{lat_max}" nominal_lon_spacing="{step}" nominal_lat_spacing="{step}" '
            f'nlon="{len(lons)}" nlat="{len(lats)}"/>\n')
    rows = [(lat, lon) for lat in lats for lon in lons]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "grid.xml").write_text(
        head + '<grid_field index="1" name="LON" units="dd" />\n<grid_field index="2" name="LAT" units="dd" />\n'
        f'<grid_field index="3" name="PSA10" units="{units}" />\n<grid_data>\n'
        + "\n".join(f"{lon:.4f} {lat:.4f} {exp(ln_sa_field(lat, lon)) * scale:.8g}" for lat, lon in rows)
        + "\n</grid_data>\n</shakemap_grid>\n", encoding="utf-8")
    if with_uncertainty:
        (folder / "uncertainty.xml").write_text(
            head + '<grid_field index="1" name="LON" units="dd" />\n<grid_field index="2" name="LAT" units="dd" />\n'
            '<grid_field index="3" name="STDPSA10" units="ln(g)" />\n<grid_data>\n'
            + "\n".join(f"{lon:.4f} {lat:.4f} {sigma}" for lat, lon in rows)
            + "\n</grid_data>\n</shakemap_grid>\n", encoding="utf-8")
    return folder


@pytest.fixture(autouse=True)
def fresh_cache():
    """Grids, resolved folders and chat results are memoised per process; tests must not share them."""
    shakemap._load_cached.cache_clear()
    shakemap._resolved.clear()
    chat.clear_result_cache()
    yield
    shakemap._load_cached.cache_clear()
    shakemap._resolved.clear()
    chat.clear_result_cache()


@pytest.fixture
def shakecast_dir(tmp_path, monkeypatch):
    """A ShakeCast user directory, selected the way a deployment would select it."""
    root = tmp_path / ".shakecast"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("SHAKECAST_USER_DIRECTORY", str(root))
    monkeypatch.setenv("FORWARD_PILOT_SHAKEMAP_SOURCE", "shakecast")
    return root / "data"


@pytest.fixture(scope="module")
def hebei():
    return load_case("hebei-transport")


# ---- reading a grid --------------------------------------------------------


def test_grid_values_are_read_in_g_and_interpolated_on_ln_sa(tmp_path):
    grid = shakemap.read_grid(write_grid(tmp_path / "ev"))
    lat, lon = 38.37, 116.11                     # off-grid on purpose
    ln_sa, sigma, inside = grid.sample([lat], [lon])
    assert inside[0]
    assert ln_sa[0] == pytest.approx(ln_sa_field(lat, lon), abs=1e-6)
    assert sigma[0] == pytest.approx(0.6)
    assert grid.meta["magnitude"] == 6.8 and grid.version == "1"


def test_older_pctg_units_mean_the_same_as_percent_g(tmp_path):
    a = shakemap.read_grid(write_grid(tmp_path / "a", units="%g"))
    b = shakemap.read_grid(write_grid(tmp_path / "b", units="pctg"))
    assert np.allclose(a.ln_psa10, b.ln_psa10)


def test_a_site_outside_the_grid_is_flagged_not_extrapolated(tmp_path):
    grid = shakemap.read_grid(write_grid(tmp_path / "ev"))
    ln_sa, _, inside = grid.sample([44.0], [116.0])
    assert not inside[0] and np.isnan(ln_sa[0])


def test_rows_in_an_unexpected_order_are_refused(tmp_path):
    with pytest.raises(shakemap.ShakeMapError, match="ordered"):
        shakemap.read_grid(write_grid(tmp_path / "ev", reverse_rows=True))


def test_missing_uncertainty_gives_zero_sigma_and_says_so(tmp_path):
    grid = shakemap.read_grid(write_grid(tmp_path / "ev", with_uncertainty=False))
    _, sigma, _ = grid.sample([38.0], [116.0])
    assert sigma[0] == 0.0
    assert "understated" in grid.warnings[0]


# ---- where grids come from ---------------------------------------------------


def test_shakecast_source_uses_the_newest_version_unless_told(shakecast_dir):
    write_grid(shakecast_dir / "ev1" / "ev1-1", "ev1", 1)
    write_grid(shakecast_dir / "ev1" / "ev1-3", "ev1", 3)
    source = shakemap.ShakeCastSource()
    assert source.directory("ev1").name == "ev1-3"
    assert source.directory("ev1", 1).name == "ev1-1"
    with pytest.raises(shakemap.ShakeMapError, match="not 2"):
        source.directory("ev1", 2)
    with pytest.raises(shakemap.ShakeMapError, match="no ShakeMap"):
        source.directory("nope")


def test_shakecast_events_are_filtered_to_the_network(shakecast_dir, hebei):
    write_grid(shakecast_dir / "near" / "near-1", "near", 1)
    far = write_grid(shakecast_dir / "far" / "far-1", "far", 1)
    text = (far / "grid.xml").read_text(encoding="utf-8").replace('lat="37.5"', 'lat="10.0"')
    (far / "grid.xml").write_text(text, encoding="utf-8")
    assert [e["event_id"] for e in shakemap.search_events(hebei)] == ["near"]


def test_comcat_source_downloads_into_the_shakecast_layout(tmp_path, monkeypatch):
    upstream = write_grid(tmp_path / "upstream", "ev9", 4)
    source = shakemap.ComcatSource(cache_dir=tmp_path / "cache")
    detail = {"properties": {"products": {"shakemap": [{
        "source": "us", "code": "ev9", "updateTime": 0, "properties": {"version": "4"},
        "contents": {f"download/{n}": {"url": (upstream / n).as_uri()} for n in ("grid.xml", "uncertainty.xml")},
    }]}}}
    monkeypatch.setattr(source, "_get_json", lambda params: detail)

    folder = source.directory("ev9")
    assert folder == tmp_path / "cache" / "ev9" / "ev9-4"
    assert (folder / "grid.xml").is_file() and (folder / "uncertainty.xml").is_file()
    assert json.loads((folder / "info.json").read_text())["version"] == 4
    # The cache is ShakeCast's layout, so the ShakeCast reader finds it as-is.
    assert shakemap.ShakeCastSource(tmp_path / "cache").directory("ev9") == folder


def test_comcat_falls_back_to_the_cache_when_offline(tmp_path, monkeypatch):
    write_grid(tmp_path / "cache" / "ev9" / "ev9-2", "ev9", 2)
    source = shakemap.ComcatSource(cache_dir=tmp_path / "cache")

    def offline(params):
        raise shakemap.ShakeMapError("could not reach the USGS catalogue")

    monkeypatch.setattr(source, "_get_json", offline)
    assert source.directory("ev9").name == "ev9-2"
    with pytest.raises(shakemap.ShakeMapError, match="could not reach"):
        source.directory("other")


# ---- ShakeCast first, then the catalogue -----------------------------------


@pytest.fixture
def fallback(tmp_path, monkeypatch):
    """An ``auto`` source over a ShakeCast directory and a catalogue whose only
    event is ``remote1``. Returns (source, shakecast data dir, upstream calls)."""
    data = tmp_path / "sc" / "data"
    data.mkdir(parents=True)
    upstream = write_grid(tmp_path / "upstream", "remote1", 7)
    calls = []

    def catalogue(params):
        calls.append(params)
        if params.get("eventid") not in (None, "remote1"):
            raise shakemap.UnknownEvent(f"USGS has no event with id {params['eventid']!r}")
        if "eventid" not in params:            # a search
            return {"features": [
                {"id": "remote1", "properties": {"mag": 6.0, "time": 0, "place": "catalogue"},
                 "geometry": {"coordinates": [115.0, 37.5, 10.0]}},
                {"id": "both", "properties": {"mag": 5.0, "time": 1000, "place": "catalogue copy"},
                 "geometry": {"coordinates": [115.0, 37.5, 10.0]}}]}
        return {"properties": {"products": {"shakemap": [{
            "source": "us", "code": "remote1", "updateTime": 0, "properties": {"version": "7"},
            "contents": {f"download/{n}": {"url": (upstream / n).as_uri()} for n in ("grid.xml", "uncertainty.xml")},
        }]}}}

    source = shakemap.FallbackSource(shakecast_root=tmp_path / "sc", cache_dir=tmp_path / "cache")
    monkeypatch.setattr(source.comcat, "_get_json", catalogue)
    return source, data, calls


def test_auto_is_the_default_source(monkeypatch):
    monkeypatch.delenv("FORWARD_PILOT_SHAKEMAP_SOURCE", raising=False)
    assert shakemap.get_source().name == "auto"


def test_shakecast_is_used_when_it_has_the_event(fallback):
    source, data, calls = fallback
    write_grid(data / "both" / "both-2", "both", 2)
    assert source.directory("both") == data / "both" / "both-2"
    assert source.last_provider == "shakecast" and calls == []      # no network call at all


def test_the_catalogue_fills_in_what_shakecast_lacks(fallback, tmp_path):
    source, data, _ = fallback
    folder = source.directory("remote1")
    assert folder == tmp_path / "cache" / "remote1" / "remote1-7"
    assert source.last_provider == "comcat"


def test_a_version_shakecast_lacks_is_looked_for_upstream(fallback):
    source, data, _ = fallback
    write_grid(data / "remote1" / "remote1-6", "remote1", 6)
    assert source.directory("remote1").name == "remote1-6"           # ShakeCast's newest
    assert source.directory("remote1", 7).name == "remote1-7"        # asked for, so fetched
    assert source.last_provider == "comcat"


def test_an_event_neither_has_is_reported_by_the_catalogue(fallback):
    source, _, _ = fallback
    with pytest.raises(shakemap.UnknownEvent, match="no event with id 'nope'"):
        source.directory("nope")


def test_without_a_shakecast_directory_auto_is_just_the_catalogue(tmp_path, monkeypatch):
    source = shakemap.FallbackSource(shakecast_root=tmp_path / "absent", cache_dir=tmp_path / "cache")
    write_grid(tmp_path / "cache" / "ev9" / "ev9-1", "ev9", 1)
    monkeypatch.setattr(source.comcat, "_get_json", lambda params: (_ for _ in ()).throw(
        shakemap.ShakeMapError("could not reach the USGS catalogue")))
    assert source.directory("ev9").name == "ev9-1" and source.last_provider == "comcat"


def test_search_merges_both_lists_once_each(fallback):
    source, data, _ = fallback
    write_grid(data / "both" / "both-1", "both", 1)
    events = source.search((112.0, 35.0, 121.0, 43.0))
    assert sorted(e["event_id"] for e in events) == ["both", "remote1"]
    assert next(e for e in events if e["event_id"] == "both")["source"] == "shakecast"


def test_search_still_lists_shakecast_events_when_offline(fallback, monkeypatch):
    source, data, _ = fallback
    write_grid(data / "both" / "both-1", "both", 1)
    monkeypatch.setattr(source.comcat, "_get_json", lambda params: (_ for _ in ()).throw(
        shakemap.ShakeMapError("could not reach the USGS catalogue")))
    assert [e["event_id"] for e in source.search((112.0, 35.0, 121.0, 43.0))] == ["both"]


def test_results_say_which_source_the_map_came_from(shakecast_dir, hebei, monkeypatch):
    monkeypatch.setenv("FORWARD_PILOT_SHAKEMAP_SOURCE", "auto")
    write_grid(shakecast_dir / "ev1" / "ev1-1", "ev1", 1)
    result = run_forward_query(hebei, {"hazard": {"model": "shakemap", "event_id": "ev1"},
                                       "samples": 100, "backend": "analytic"})
    assert any("ShakeMap ev1 version 1" in w and "via the ShakeCast server" in w for w in result["warnings"])


def test_an_unknown_event_is_reported_as_such_not_as_an_outage(tmp_path, monkeypatch):
    """USGS answers a bad id with 404. Calling that unreachable sent a model off
    concluding the event did not exist in the package at all."""
    from urllib.error import HTTPError

    def not_found(request, timeout=None):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(shakemap, "urlopen", not_found)
    write_grid(tmp_path / "cache" / "ev9" / "ev9-1", "ev9", 1)
    with pytest.raises(shakemap.UnknownEvent, match="no event with id 'ev9x'"):
        shakemap.ComcatSource(cache_dir=tmp_path / "cache").directory("ev9x")


def test_a_garbled_event_id_comes_back_with_the_likely_one(shakecast_dir, hebei):
    write_grid(shakecast_dir / "iscgem849480" / "iscgem849480-1", "iscgem849480", 1)
    out = json.loads(chat.run_query_for_chat(
        hebei, [], shakemap_event_id="uscgem849480", samples=100,
        known_events=["iscgem849480", "usp0000hjg"]))
    assert out["refused"] is True
    assert out["did_you_mean"] == ["iscgem849480"]


def test_an_event_id_that_is_not_one_is_refused_before_any_lookup():
    with pytest.raises(shakemap.ShakeMapError, match="not a USGS event id"):
        shakemap.load("../../etc/passwd")


# ---- through the engine ------------------------------------------------------


def test_bridge_failure_uses_the_shakemap_median_and_both_dispersions(shakecast_dir, hebei):
    write_grid(shakecast_dir / "ev1" / "ev1-1", "ev1", 1, sigma=0.6)
    result = run_forward_query(hebei, {"hazard": {"model": "shakemap", "event_id": "ev1"},
                                       "samples": 200, "backend": "analytic"})
    edge_id = "e24"
    edge = hebei["edges"][edge_id]
    a, b = hebei["nodes"][edge["from"]], hebei["nodes"][edge["to"]]
    lat, lon = (a["lat"] + b["lat"]) / 2, (a["lon"] + b["lon"]) / 2
    median, beta = _median_extensive(edge["bridge_class"])
    expected = ndtr((ln_sa_field(lat, lon) - log(median)) / sqrt(0.6**2 + beta**2))

    component = next(c for c in result["components"] if c["edge_id"] == edge_id)
    assert component["p_unusable"] == pytest.approx(expected, abs=1e-6)
    assert component["sa1_g"] == pytest.approx(exp(ln_sa_field(lat, lon)), rel=1e-5)
    assert any("ShakeMap ev1 version 1" in w for w in result["warnings"])
    assert any("no seismic station" in w for w in result["warnings"])


def test_a_network_partly_outside_the_shakemap_is_refused(shakecast_dir, hebei):
    write_grid(shakecast_dir / "small" / "small-1", "small", 1, extent=(114.0, 36.0, 116.0, 38.0))
    with pytest.raises(ForwardQueryError, match="outside the ShakeMap"):
        run_forward_query(hebei, {"hazard": {"model": "shakemap", "event_id": "small"},
                                  "samples": 100, "backend": "analytic"})


def test_a_package_without_coordinates_cannot_take_a_shakemap(shakecast_dir, hebei):
    planar = {**hebei, "nodes": {k: {f: v for f, v in n.items() if f not in ("lat", "lon")}
                                 for k, n in hebei["nodes"].items()}}
    write_grid(shakecast_dir / "ev1" / "ev1-1", "ev1", 1)
    with pytest.raises(ForwardQueryError, match="no geodetic coordinates"):
        run_forward_query(planar, {"hazard": {"model": "shakemap", "event_id": "ev1"},
                                   "samples": 100, "backend": "analytic"})


def test_ema_is_georeferenced_and_offers_shakemaps():
    ema = load_case("ema-highway")
    assert "shakemap" in ema["hazard_models"]
    assert ema["default_hazard"]["model"] == "ema_reference_scenario"     # unchanged default
    lats = [n["lat"] for n in ema["nodes"].values()]
    lons = [n["lon"] for n in ema["nodes"].values()]
    # Eastern Massachusetts and its neighbours, with Boston where Boston is.
    assert 41.5 < min(lats) and max(lats) < 43.0 and -72.5 < min(lons) and max(lons) < -70.4
    boston = ema["nodes"]["n22"]
    assert boston["zone"] == "Boston"
    assert abs(boston["lat"] - 42.36) < 0.1 and abs(boston["lon"] + 71.06) < 0.1


def test_ema_bridges_sample_a_shakemap_covering_the_region(shakecast_dir):
    ema = load_case("ema-highway")
    write_grid(shakecast_dir / "ne1" / "ne1-1", "ne1", 1, extent=(-73.5, 41.0, -69.5, 43.5))
    result = run_forward_query(ema, {"hazard": {"model": "shakemap", "event_id": "ne1"},
                                     "samples": 100, "backend": "analytic"})
    assert len(result["components"]) == len(ema["edges"])


def test_tbn_backend_gets_the_same_site_terms(shakecast_dir, hebei):
    pytest.importorskip("tbnpy")
    from src.forward_pilot.tbn.model import _site_terms

    write_grid(shakecast_dir / "ev1" / "ev1-1", "ev1", 1, sigma=0.6)
    terms = _site_terms(hebei, {"model": "shakemap", "event_id": "ev1"})
    assert terms == shakemap.site_terms(hebei, {"model": "shakemap", "event_id": "ev1"})[0]


# ---- through the chat tool ------------------------------------------------------


def test_naming_an_event_selects_the_shakemap_and_states_it(shakecast_dir, hebei):
    write_grid(shakecast_dir / "ev1" / "ev1-1", "ev1", 1)
    events, runs = [], []
    chat.run_query_for_chat(hebei, runs, on_event=events.append, shakemap_event_id="ev1", samples=100)
    assert runs[0]["query"]["hazard"] == {"model": "shakemap", "event_id": "ev1"}
    assert events[0]["statement"]["given"] == "ShakeMap ev1"
    assert [e["type"] for e in events] == ["query", "computed"]


def test_a_revised_shakemap_is_not_served_the_old_result(shakecast_dir, hebei):
    """Same event id, new map version: a different hazard, so the engine runs again."""
    write_grid(shakecast_dir / "ev1" / "ev1-1", "ev1", 1, sigma=0.6)
    runs = []
    chat.run_query_for_chat(hebei, runs, shakemap_event_id="ev1", samples=100)
    write_grid(shakecast_dir / "ev1" / "ev1-2", "ev1", 2, sigma=0.2)
    shakemap._resolved.clear()                 # as if the few-minute memo had expired
    events = []
    chat.run_query_for_chat(hebei, runs, on_event=events.append, shakemap_event_id="ev1", samples=100)
    assert events[0]["reused"] is False
    assert runs[0]["result"]["components"] != runs[1]["result"]["components"]


def test_prompt_lists_nearby_events_only_when_given_them(hebei):
    events = [{"event_id": "iscgem849480", "time": "1966-03-22T08:19:36", "magnitude": 6.8,
               "place": "34 km NW of Nangong, China"}]
    prompt = chat.system_prompt(hebei, events)
    assert "shakemap_event_id" in prompt and "iscgem849480: 1966-03-22, M6.8" in prompt
    assert "iscgem849480" not in chat.system_prompt(hebei)
