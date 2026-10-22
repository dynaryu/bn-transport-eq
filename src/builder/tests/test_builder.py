"""The schema and instances are consistent, compile, and the kernels compute what they claim."""

from __future__ import annotations

import copy
import json
import shutil
from math import log

import numpy as np
import pytest
from scipy.special import ndtr

from src.builder.compile import compile_assembly, compile_path
from src.builder.kernels import Call
from src.builder.kernels.fragility import Fragility
from src.builder.kernels.hazard import (ShakeMap, SpatialCorrelationField, campbell_2003_ln_median,
                                        great_circle_km, jayaram_baker_2009_range_km)
from src.builder.kernels.edge import EdgeOpen
from src.builder.load import (INSTANCES, NETWORKS, SCHEMA, RegistryError, assembly_path, assembly_paths,
                              load_assembly, load_library, load_network)
from src.inference.sample import forward_sample


@pytest.fixture(scope="module")
def lib():
    return load_library()


def _compile(lib, assembly_id, edit=None, network_edit=None):
    assembly = load_assembly(assembly_path(assembly_id))
    network = load_network(assembly["network"], lib)
    if edit:
        assembly = copy.deepcopy(assembly)
        edit(assembly)
    if network_edit:
        network = copy.deepcopy(network)
        network_edit(network)
    return compile_assembly(assembly, lib, network)


# ---- the instances as committed ---------------------------------------------------


def test_every_assembly_compiles(lib):
    paths = assembly_paths()
    assert len(paths) >= 2
    for path in paths:
        assert compile_path(path, lib).n_nodes > 0


def test_the_schema_diagram_is_current(lib):
    from src.builder import diagram

    assert diagram.is_current(lib=lib), "schema/diagram.md is stale; run: python -m src diagram"


def test_the_hazus_transcription_is_the_one_the_curves_cite(lib):
    import hashlib

    source = lib.parameter_sets["hazus61.HWB1"]["source"]
    text = (SCHEMA.parent / source["transcription"]).read_bytes()
    assert hashlib.sha256(text).hexdigest() == source["transcription_sha256"]


def test_few_parameter_sets_describe_every_bridge(lib):
    c = _compile(lib, "ema-highway/gmm")
    fragility = next(b for b in c.summary() if b["function"] == "fragility.bridge")
    assert fragility["rows"] == 129
    assert fragility["parameter_sets"] == {"hazus61.HWB11": 90, "hazus61.HWB4": 28, "hazus61.HWB2": 11}


# ---- consistency errors --------------------------------------------------------


def test_two_producers_for_one_node_is_refused(lib):
    def add_second_hazard(a):
        a["bindings"].insert(1, {"function": "hazard.shakemap", "parameters": {"event_id": "us7000abcd", "imt": "SA(1.0)"}})

    with pytest.raises(RegistryError, match="already produced by binding"):
        _compile(lib, "ema-highway/gmm", add_second_hazard)


def test_missing_producer_is_named(lib):
    def drop_hazard(a):
        del a["bindings"][1]

    with pytest.raises(RegistryError, match=r"no binding produces SA\(1\.0\) for Bridge bridge-e0001"):
        _compile(lib, "ema-highway/gmm", drop_hazard)


def test_unknown_parameter_set_names_the_rows(lib):
    def unknown_class(network):
        network.tables["Bridge"][0]["hazus_class"] = "HWB99"

    with pytest.raises(RegistryError, match="no parameter set 'hazus61.HWB99'.*bridge-e0001"):
        _compile(lib, "ema-highway/gmm", network_edit=unknown_class)


def test_inline_parameters_are_validated(lib):
    def no_event(a):
        next(b for b in a["bindings"] if b["function"] == "hazard.shakemap")["parameters"]["event_id"] = ""

    with pytest.raises(RegistryError, match="event_id"):
        _compile(lib, "hebei-transport/shakemap", no_event)


def test_network_rows_are_checked(lib, tmp_path):
    shutil.copytree(NETWORKS / "hebei-transport", tmp_path / "hebei-transport")
    path = tmp_path / "hebei-transport" / "network.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tables"]["Bridge"][0]["edge"] = "nowhere"
    data["tables"]["Bridge"][1]["span_count"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RegistryError) as exc:
        load_network("hebei-transport", lib, root=tmp_path)
    text = str(exc.value)
    assert "edge = 'nowhere' is not a Edge row" in text
    assert "unknown attribute 'span_count'" in text


def test_schema_checks_ports_against_plates(tmp_path):
    shutil.copytree(SCHEMA, tmp_path / "schema")
    path = tmp_path / "schema" / "functions" / "fragility.bridge.json"
    fn = json.loads(path.read_text(encoding="utf-8"))
    fn["inputs"][0]["via"] = "edge"               # Bridge.edge points at Edge, not at a Bridge
    fn["capabilities"].append("Gradient")         # not implemented
    path.write_text(json.dumps(fn), encoding="utf-8")
    with pytest.raises(RegistryError) as exc:
        load_library(tmp_path / "schema", INSTANCES)
    text = str(exc.value)
    assert "via 'edge' needs Bridge.edge to be a foreign key to Bridge" in text
    assert "OVERCLAIM Gradient" in text


# ---- a GMM chosen by location, an IM left open --------------------------------------


def _with_gmm_sets(tmp_path, *sets):
    """The library with extra hazard.gmm parameter sets, written to a scratch copy of instances/."""
    shutil.copytree(INSTANCES, tmp_path / "instances")
    campbell = json.loads((INSTANCES / "hazard" / "campbell-2003.json").read_text(encoding="utf-8"))
    campbell["sets"] = list(sets)
    (tmp_path / "instances" / "hazard" / "test-gmm.json").write_text(json.dumps(campbell), encoding="utf-8")
    return load_library(SCHEMA, tmp_path / "instances")


def _campbell_like(lib, set_id, **change):
    """Campbell's numbers under another id: a test fixture for selection, not a ground-motion model."""
    base = lib.parameter_sets["campbell-2003.ema-script-sa1"]
    return {"id": set_id, "label": "TEST ONLY", "values": {**base["values"], **change.pop("values", {})}, **change}


def _gmm_by(*set_ids):
    def edit(a):
        b = next(b for b in a["bindings"] if b["function"] == "hazard.gmm")
        b.clear()
        b.update(function="hazard.gmm", parameter_set_by_location=list(set_ids))
    return edit


def _move_first_bridge_to_beijing(network):
    network.tables["Bridge"][0].update(lat=39.9, lon=116.4)


def test_every_gmm_form_named_in_the_schema_has_a_kernel(lib):
    from src.builder.kernels.hazard import GMM_FORMS

    assert set(lib.functions["hazard.gmm"]["instance_schema"]["properties"]["form"]["enum"]) == set(GMM_FORMS)


def test_each_bridge_gets_the_gmm_whose_region_contains_it(lib, tmp_path):
    china = {"name": "test box around Beijing", "polygon": [[110.0, 35.0], [120.0, 35.0], [120.0, 45.0], [110.0, 45.0]]}
    lib2 = _with_gmm_sets(tmp_path, _campbell_like(lib, "test.china", region=china))
    c = _compile(lib2, "ema-highway/gmm", _gmm_by("campbell-2003.ema-script-sa1", "test.china"),
                 network_edit=_move_first_bridge_to_beijing)
    gmm = next(b for b in c.summary() if b["function"] == "hazard.gmm")
    assert gmm["parameter_sets"] == {"campbell-2003.ema-script-sa1": 128, "test.china": 1}


def test_a_bridge_no_listed_region_covers_is_refused_by_name(lib):
    with pytest.raises(RegistryError, match="no listed parameter set covers Bridge bridge-e0001 "
                                            r"\(campbell-2003.ema-script-sa1: Central and eastern North America\)"):
        _compile(lib, "ema-highway/gmm", _gmm_by("campbell-2003.ema-script-sa1"),
                 network_edit=_move_first_bridge_to_beijing)


def test_a_hazard_on_the_wrong_im_does_not_wire_to_the_curves(lib, tmp_path):
    pga = _campbell_like(lib, "test.pga", values={"imt": "PGA"},
                         region=lib.parameter_sets["campbell-2003.ema-script-sa1"]["region"])
    lib2 = _with_gmm_sets(tmp_path, pga)
    with pytest.raises(RegistryError, match=r"no binding produces SA\(1\.0\) for Bridge bridge-e0001.*"
                                            r"Bridge bridge-e0001 has .*PGA"):
        _compile(lib2, "ema-highway/gmm", _gmm_by("test.pga"))


def test_a_templated_port_needs_a_required_enum_of_quantities(tmp_path):
    shutil.copytree(SCHEMA, tmp_path / "schema")
    path = tmp_path / "schema" / "functions" / "hazard.gmm.json"
    fn = json.loads(path.read_text(encoding="utf-8"))
    fn["instance_schema"]["properties"]["imt"]["enum"].append("SA(9.9)")       # not a quantity
    path.write_text(json.dumps(fn), encoding="utf-8")
    with pytest.raises(RegistryError, match="parameter 'imt' must be an enum of quantity ids"):
        load_library(tmp_path / "schema", INSTANCES)


# ---- kernels compute what they claim -----------------------------------------------

# ln median SA1 (g) from the EMA dataset script's Campbell (2003), as reproduced by the
# forward pilot's hazard._gmpe_ln_sa1 at commit 784e6cc.
CAMPBELL_REFERENCE = [
    (5.0, 5.0, -3.0411925158948376),
    (6.5, 60.0, -3.364020778289433),
    (7.5, 100.0, -2.8219992023780347),
    (8.0, 200.0, -2.9479540856474182),
]


@pytest.mark.parametrize("mw,r,expected", CAMPBELL_REFERENCE)
def test_campbell_reproduces_the_dataset_script(lib, mw, r, expected):
    c = lib.parameter_sets["campbell-2003.ema-script-sa1"]["values"]["coefficients"]
    assert campbell_2003_ln_median(np.array(mw), np.array(r), c) == pytest.approx(expected)


def _call(lib, inputs, rows, params=None, parents=None, tables=None, ports=None, n=None):
    n = n or next(iter(v if not isinstance(v, list) else v[0] for v in inputs.values())).shape[0]
    quantities = {name: lib.quantities[q] for name, q in (ports or {}).items()}
    return Call(n, rows, inputs, parents or {}, params, tables or {}, quantities, np.random.default_rng(7))


def test_fragility_density_and_sampling(lib):
    params = lib.parameter_sets["hazus61.HWB11"]["values"]
    im = np.array([[0.0, 0.5, 1.1, 3.0]])
    call = _call(lib, {"im": im}, [{"id": str(i)} for i in range(4)], params,
                 ports={"im": "SA(1.0)", "damage": "DamageState.HazusBridge"})
    p = Fragility().evaluate_density(call)["damage"][0]
    assert np.allclose(p.sum(-1), 1) and np.all(p >= 0)
    assert p[0, 0] == 1.0                                         # no shaking, no damage
    extensive_or_worse = p[:, 3:].sum(-1)
    assert np.allclose(extensive_or_worse[1:], ndtr(np.log(im[0, 1:] / 1.1) / 0.6))

    big = _call(lib, {"im": np.full((20000, 1), 1.1)}, [{"id": "b"}], params,
                ports={"im": "SA(1.0)", "damage": "DamageState.HazusBridge"})
    ds = Fragility().sample_forward(big)["damage"][:, 0]
    assert np.mean(ds >= 3) == pytest.approx(0.5, abs=0.02)      # the median of Extensive


# ---- fragility: one schema, curves as instances ---------------------------------------


def test_every_fragility_form_named_in_the_schema_has_a_kernel(lib):
    from src.builder.kernels.fragility import FRAGILITY_FORMS

    assert set(lib.functions["fragility.bridge"]["instance_schema"]["properties"]["form"]["enum"]) == set(FRAGILITY_FORMS)


def test_per_state_betas_and_crossing_curves_are_capped_with_a_note(lib):
    hazus = lib.parameter_sets["hazus61.HWB11"]["values"]
    params = {**hazus, "parameters": {"medians": hazus["parameters"]["medians"],
                                      "betas": {"slight": 0.3, "moderate": 0.4, "extensive": 0.9, "complete": 0.9}}}
    im = np.array([[0.05, 1.1, 5.0]])
    call = _call(lib, {"im": im}, [{"id": str(i)} for i in range(3)], params,
                 ports={"im": "SA(1.0)", "damage": "DamageState.HazusBridge"})
    p = Fragility().evaluate_density(call)["damage"][0]
    assert np.allclose(p.sum(-1), 1) and np.all(p >= 0)
    assert p[1, 3:].sum() == pytest.approx(ndtr(0.0))           # Extensive's own beta, at its median
    # Moderate (beta 0.4) and Extensive (beta 0.9) cross at low im: P(DS >= extensive) is capped there.
    assert any("cross" in w for w in call.warnings)


def _library_with(tmp_path, folder, name, data):
    shutil.copytree(INSTANCES, tmp_path / "instances")
    (tmp_path / "instances" / folder).mkdir(exist_ok=True)
    (tmp_path / "instances" / folder / name).write_text(json.dumps(data), encoding="utf-8")
    return load_library(SCHEMA, tmp_path / "instances")


def _curve(set_id, **values):
    base = {"form": "lognormal", "imt": "PGA", "damage_scale": "DamageState.HazusBridge",
            "parameters": {"medians": {"slight": 0.1, "moderate": 0.2, "extensive": 0.3, "complete": 0.4}, "beta": 0.5}}
    return {"function": "fragility.bridge", "source": {"note": "TEST ONLY"},
            "sets": [{"id": set_id, "label": "TEST ONLY", "values": {**base, **values}}]}


def test_a_new_curve_is_an_instance_file_with_no_schema_change(lib, tmp_path):
    lib2 = _library_with(tmp_path, "fragility", "test-curves.json", _curve("test.pga-curve"))
    assert lib2.parameter_sets["test.pga-curve"]["function"] == "fragility.bridge"


def test_a_curve_that_does_not_fit_its_damage_scale_is_refused(tmp_path):
    bad = _curve("test.bad", parameters={"medians": {"minor": 0.1, "collapse": 0.4}, "beta": 0.5})
    with pytest.raises(RegistryError) as exc:
        _library_with(tmp_path, "fragility", "test-curves.json", bad)
    text = str(exc.value)
    assert "test.bad" in text and "'slight' is a required property" in text


def test_a_set_filed_under_another_role_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="role 'fragility', so its sets belong in"):
        _library_with(tmp_path, "hazard", "test-curves.json", _curve("test.misfiled"))


def test_the_gmm_gives_each_bridge_its_own_median_and_sigma(lib):
    """ln SA(1.0) at a bridge is N(Campbell median at its hypocentral distance, total sigma)."""
    c = _compile(lib, "ema-highway/gmm")
    gmm = next(b for b in c.bindings if b.function["id"] == "hazard.gmm")
    samples = forward_sample(c, 20000, seed=3, bindings=c.ancestors({gmm.index}))
    event = c.network.row("ema-reference")
    values = lib.parameter_sets["campbell-2003.ema-script-sa1"]["values"]
    s = values["sigma"]
    sigma = np.hypot(s["a"] + s["b"] * event["mw"] if event["mw"] < s["m_break"] else s["sigma_above"],
                     values["sigma_epistemic"])
    for bridge in c.network.rows("Bridge")[:5]:
        r = np.hypot(great_circle_km(event["lat"], event["lon"], bridge["lat"], bridge["lon"]), event["depth_km"])
        x = np.log(samples[f"SA(1.0)::{bridge['id']}"])
        assert x.mean() == pytest.approx(float(campbell_2003_ln_median(event["mw"], r, values["coefficients"])), abs=0.02)
        assert x.std() == pytest.approx(sigma, rel=0.03)


def test_an_edge_closes_with_any_bridge_and_is_open_without_one(lib):
    usable = [np.array([[1, 1], [1, 0]]), np.empty((2, 0), dtype=int)]
    out = EdgeOpen().sample_forward(_call(lib, {"usable": usable}, [{"id": "a"}, {"id": "b"}], n=2))["open"]
    assert out.tolist() == [[1, 1], [0, 1]]


def _diamond(lib, open_states, target_max=0.5, directed=False):
    """s-a-t is 1 h and s-b-t 2 h at 80 km/h; t is the key destination, s, a and b the origins."""
    nodes = [{"id": x, "key_destination": x == "t"} for x in "sabt"]
    edges = [{"id": eid, "source": u, "target": v, "directed": directed, "length_km": km, "avg_speed_kph": 80.0}
             for eid, u, v, km in (("sa", "s", "a", 40.0), ("at", "a", "t", 40.0), ("sb", "s", "b", 80.0),
                                   ("bt", "b", "t", 80.0))]
    ids = [e["id"] for e in edges]
    rows = [{"id": x} for x in "sab"]
    opened = np.asarray(open_states)
    return _call(lib, {"open": [opened] * len(rows)}, rows, {"target_max": target_max},
                 parents={"open": [ids] * len(rows)}, tables={"Node": nodes, "Edge": edges})


def test_travel_time_to_nearest_judges_each_origin_against_its_intact_time(lib):
    from src.builder.kernels.system import TravelTimeToNearest

    #                 sa at sb bt
    call = _diamond(lib, [[1, 1, 1, 1],      # intact: s 1.0 h (via a)
                          [0, 1, 1, 1],      # sa closed: s goes via b, 2.0 h, 1.0 h late
                          [0, 1, 0, 1]])     # s cut off
    out = TravelTimeToNearest().sample_forward(call)
    assert out["travel_time"][:, 0].tolist() == [1.0, 2.0, np.inf]
    assert out["state"][:, 0].tolist() == [1, 0, 0]                 # 1.0 h late > target_max 0.5 h
    assert out["travel_time"][:, 1].tolist() == [0.5, 0.5, 0.5]      # a reaches t directly throughout
    assert not call.warnings

    lenient = TravelTimeToNearest().sample_forward(_diamond(lib, [[0, 1, 1, 1]], target_max=1.0))
    assert lenient["state"][0, 0] == 1                               # 1.0 h late is within 1.0 h


def test_an_origin_with_no_route_even_intact_is_reported(lib):
    from src.builder.kernels.system import TravelTimeToNearest

    call = _diamond(lib, [[1, 1, 1, 1]], directed=True)               # every edge points away from s, so t->s is impossible
    call.rows.append({"id": "t"})
    call.parents["open"].append(call.parents["open"][0])
    call.inputs["open"].append(call.inputs["open"][0])
    for node in call.tables["Node"]:
        node["key_destination"] = node["id"] == "s"
    out = TravelTimeToNearest().sample_forward(call)
    assert out["state"][0].tolist() == [1, 0, 0, 0]                 # only s itself reaches s
    assert "fail even with every edge open" in call.warnings[0]


def test_every_travel_time_node_reads_every_link(lib):
    c = _compile(lib, "ema-highway/gmm")
    nodes = {n["id"]: n for n in c.nodes()}
    origins = [r["id"] for r in c.network.rows("Node") if not r["key_destination"]]
    assert len(origins) == 72 and "TravelTime::n22" not in nodes              # key destinations are not origins
    assert len(nodes[f"SystemState::{origins[0]}"]["parents"]) == 129


def test_shakemap_kernel_samples_and_refuses_outside(lib, monkeypatch):
    from src.builder import shakemap

    grid = shakemap.ShakeMapGrid(
        event_id="test", version="1", meta={}, lon_min=0.0, lat_max=1.0, dlon=1.0, dlat=1.0,
        ln_im=np.full((2, 2), log(0.4)), std_im=np.full((2, 2), 0.5),
    )
    monkeypatch.setattr(shakemap, "load", lambda *a, **k: grid)
    rows = [{"id": "in", "lat": 0.5, "lon": 0.5}]
    epsilon = np.array([[-1.0], [0.0], [2.0]])
    call = _call(lib, {"epsilon": epsilon}, rows, {"event_id": "test", "imt": "SA(1.0)"})
    sa = ShakeMap().sample_forward(call)["im"][:, 0]
    assert np.allclose(np.log(sa), log(0.4) + 0.5 * epsilon[:, 0])     # deterministic given the residual
    assert any("no seismic station" in w for w in call.warnings)

    outside = _call(lib, {"epsilon": np.zeros((1, 2))}, rows + [{"id": "out", "lat": 5.0, "lon": 5.0}],
                    {"event_id": "test", "imt": "SA(1.0)"})
    with pytest.raises(ValueError, match="outside ShakeMap"):
        ShakeMap().sample_forward(outside)


@pytest.mark.parametrize("period, clustered, expected", [
    (0.0, False, 8.5), (0.0, True, 40.7), (0.5, False, 17.1), (0.5, True, 33.2),
    (1.0, False, 25.7), (1.0, True, 25.7), (2.0, False, 29.4),
])
def test_jayaram_baker_ranges(period, clustered, expected):
    assert jayaram_baker_2009_range_km(period, clustered) == pytest.approx(expected)


def _field(lib, set_id, rows, n=40000, truncation=None):
    values = {**lib.parameter_sets[set_id]["values"], "truncation_level": truncation}
    return _call(lib, {}, rows, values, n=n)


def test_the_field_reproduces_the_correlation_model(lib):
    # 0.08993 deg of latitude is 10.0 km; the third site sits on the first.
    rows = [{"id": "a", "lat": 0.0, "lon": 0.0}, {"id": "b", "lat": 0.08993, "lon": 0.0},
            {"id": "c", "lat": 0.0, "lon": 0.0}]
    call = _field(lib, "jayaram-baker-2009.sa1p0", rows)
    eps = SpatialCorrelationField().sample_forward(call)["epsilon"]
    r = np.corrcoef(eps.T)
    assert np.allclose(eps.std(axis=0), 1.0, atol=0.02)
    assert r[0, 1] == pytest.approx(np.exp(-3 * 10.0 / 25.7), abs=0.02)
    assert r[0, 2] == pytest.approx(1.0, abs=1e-6)                     # co-located: no failure, fully correlated
    assert not call.warnings


def test_independent_sites_are_uncorrelated_and_say_so(lib):
    rows = [{"id": "a", "lat": 0.0, "lon": 0.0}, {"id": "b", "lat": 0.0, "lon": 0.0}]
    call = _field(lib, "independent.any-im", rows)
    eps = SpatialCorrelationField().sample_forward(call)["epsilon"]
    assert abs(np.corrcoef(eps.T)[0, 1]) < 0.03
    assert "independently" in call.warnings[0]


def test_truncation_bounds_the_independent_normals(lib):
    call = _field(lib, "independent.any-im", [{"id": "a", "lat": 0.0, "lon": 0.0}], truncation=2.0)
    eps = SpatialCorrelationField().sample_forward(call)["epsilon"]
    assert np.abs(eps).max() <= 2.0 and eps.std() < 0.9                # truncated at 2 sd: variance ~0.77
