"""Known-value tests. These are the regression floor for the asset.

Two things are under test, and they are different in kind:

  the framework  -- log base, units, grids, required inputs, extrapolation, the
                    lognormal kernel. Exercised through the synthetic verification
                    kernel, whose coefficients are invented on purpose.
  the registry   -- that all 30 families are present, and that the wrapper REFUSES
                    to predict for every one of them while their coefficients are
                    absent. That refusal is a feature, so it gets a test.
"""

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.wrappers.loader import load_asset_module  # noqa: E402

# Loaded by path, not as `model`: every asset's wrapper is called model.py, so a bare
# `from model import ...` picks up whichever asset pytest collected first.
_m = load_asset_module(Path(__file__).resolve().parents[1] / "model.py")

LN10 = _m.LN10
CoefficientsUnavailable = _m.CoefficientsUnavailable
DistanceContext = _m.DistanceContext
Gmpe = _m.Gmpe
LogBase = _m.LogBase
OutOfScope = _m.OutOfScope
RuptureContext = _m.RuptureContext
SiteContext = _m.SiteContext
ValidationError = _m.ValidationError
coefficient_status = _m.coefficient_status
coverage_matrix = _m.coverage_matrix
load_registry = _m.load_registry
missing_sources = _m.missing_sources
reproduction_status = _m.reproduction_status
unit_factor = _m.unit_factor

# The synthetic kernel: log10 Y[cm/s2] = a*M + b*R - log10(R) + c[site]
A_PGA, B_PGA, C_ROCK = 0.50, -0.0030, 0.20
TAU_PGA, PHI_PGA = 0.15, 0.25


@pytest.fixture
def asset():
    return Gmpe()


def given(mw=6.0, rhypo=20.0, site_class="rock", **kw):
    g = {
        "imt": "PGA", "damping": 0.05,
        "rupture": {"mw": mw},
        "distance": {"rhypo_km": rhypo},
        "site": {"site_class": site_class},
    }
    g.update(kw)
    return g


# ---- the framework's arithmetic -------------------------------------------


def test_median_matches_the_hand_computed_synthetic_form(asset):
    """Anchor case, computed independently of the implementation."""
    mw, r = 6.0, 20.0
    log10_cm_s2 = A_PGA * mw + B_PGA * r - math.log10(r) + C_ROCK
    expected_g = (10.0 ** log10_cm_s2) / 980.665

    out = asset.predict(imt="PGA", rupture=RuptureContext(mw=mw),
                        distance=DistanceContext(rhypo_km=r),
                        site=SiteContext(site_class="rock"))
    assert out.units == "g"
    assert out.median == pytest.approx(expected_g, rel=1e-12)
    assert out.ln_median == pytest.approx(math.log(expected_g), rel=1e-12)


def test_log_base_converts_but_units_do_not_touch_sigma(asset):
    """sigma scales with the LOG BASE only.

    Changing units multiplies Y by a constant, which shifts ln Y and leaves its
    spread alone. Scaling sigma by the unit factor too is a classic silent bug, so
    it is pinned here.
    """
    a = asset.predict(imt="PGA", units="g", **_ctx())
    b = asset.predict(imt="PGA", units="cm/s2", **_ctx())

    assert a.tau_inter == pytest.approx(TAU_PGA * LN10, rel=1e-12)
    assert a.phi_intra == pytest.approx(PHI_PGA * LN10, rel=1e-12)
    assert b.tau_inter == pytest.approx(a.tau_inter, rel=1e-12)
    assert b.sigma_total == pytest.approx(a.sigma_total, rel=1e-12)

    # ...while the median does move, by exactly the unit factor.
    assert b.median / a.median == pytest.approx(980.665, rel=1e-9)


def test_sigma_total_is_composed_and_flagged_as_derived(asset):
    out = asset.predict(imt="PGA", **_ctx())
    assert out.sigma_total == pytest.approx(
        math.sqrt((TAU_PGA * LN10) ** 2 + (PHI_PGA * LN10) ** 2), rel=1e-12
    )
    assert out.provenance["sigma_total_derived_from_tau_phi"] is True


def test_unit_conversion_refuses_to_cross_dimensions():
    assert unit_factor("cm/s2", "m/s2") == pytest.approx(0.01)
    assert unit_factor("g", "m/s2") == pytest.approx(9.80665)
    assert unit_factor("cm/s2", "g") == pytest.approx(1.0 / 980.665)
    with pytest.raises(ValidationError, match="Crossing dimensions"):
        unit_factor("cm/s2", "cm/s")


def test_log_base_round_trip():
    assert LogBase.LOG10.to_ln(1.0) == pytest.approx(LN10)
    assert LogBase.LN.to_ln(1.0) == pytest.approx(1.0)


def test_median_decreases_with_distance(asset):
    """Monotonicity sanity check for THIS kernel's form, not asserted universally.

    Saturation and basin terms in real models can break monotonicity, which is why
    the cross-model check is deliberately not an absolute assertion.
    """
    far = [asset.predict(imt="PGA", **_ctx(rhypo=r)).median for r in (5.0, 20.0, 80.0, 150.0)]
    assert far == sorted(far, reverse=True)


# ---- the lognormal kernel -------------------------------------------------


def test_density_integrates_to_one(asset):
    """It is a proper density over Y, which is what makes Z trivial and uninteresting."""
    out = asset.predict(imt="PGA", **_ctx())
    y = np.exp(np.linspace(out.ln_median - 8 * out.sigma_total,
                           out.ln_median + 8 * out.sigma_total, 40001))
    dens = asset.evaluate_density({"LocalIntensity": y}, given())
    assert np.trapezoid(dens, y) == pytest.approx(1.0, abs=1e-4)


def test_density_peaks_at_the_mode_and_is_zero_below_zero(asset):
    out = asset.predict(imt="PGA", **_ctx())
    mode = math.exp(out.ln_median - out.sigma_total ** 2)
    y = np.array([mode * 0.5, mode, mode * 2.0])
    dens = asset.evaluate_density({"LocalIntensity": y}, given())
    assert dens[1] > dens[0] and dens[1] > dens[2]

    edge = asset.evaluate_density({"LocalIntensity": np.array([-1.0, 0.0])}, given())
    assert list(edge) == [0.0, 0.0]


def test_sampling_recovers_the_median_and_sigma(asset):
    out = asset.predict(imt="PGA", **_ctx())
    draws = asset.sample_forward(given(), n=200_000, rng=np.random.default_rng(7))["LocalIntensity"]
    ln = np.log(draws)
    assert ln.mean() == pytest.approx(out.ln_median, abs=0.01)
    assert ln.std() == pytest.approx(out.sigma_total, rel=0.01)


def test_sampling_is_reproducible_given_a_seed(asset):
    a = asset.sample_forward(given(), n=5, rng=np.random.default_rng(42))["LocalIntensity"]
    b = asset.sample_forward(given(), n=5, rng=np.random.default_rng(42))["LocalIntensity"]
    assert np.array_equal(a, b)


def test_likelihood_is_the_same_ordinate_read_over_the_parents(asset):
    """Unnormalised over the parents: the magnitudes consistent with one recorded
    intensity do not integrate to one, which is the point of a backward message."""
    y = {"LocalIntensity": np.array([0.07])}
    vals = [asset.evaluate_likelihood(y, given(mw=m))[0] for m in (5.0, 5.5, 6.0, 6.5, 7.0)]
    assert all(v > 0 for v in vals)
    assert sum(vals) != pytest.approx(1.0)
    assert asset.evaluate_likelihood(y, given())[0] == pytest.approx(
        asset.evaluate_density(y, given())[0]
    )


# ---- validation: the non-negotiable rules ---------------------------------


def test_missing_required_input_is_an_error_not_an_assumption(asset):
    with pytest.raises(ValidationError, match="site.site_class"):
        asset.predict(imt="PGA", rupture=RuptureContext(mw=6.0),
                      distance=DistanceContext(rhypo_km=20.0), site=SiteContext())


def test_distance_metrics_are_not_substituted(asset):
    """The kernel needs Rhypo; supplying Rrup instead must fail, not fall back."""
    with pytest.raises(ValidationError, match="not interchangeable"):
        asset.predict(imt="PGA", rupture=RuptureContext(mw=6.0),
                      distance=DistanceContext(rrup_km=20.0),
                      site=SiteContext(site_class="rock"))


def test_vs30_is_not_silently_mapped_onto_a_categorical_site_model(asset):
    with pytest.raises(ValidationError, match="not applied silently"):
        asset.predict(imt="PGA", rupture=RuptureContext(mw=6.0),
                      distance=DistanceContext(rhypo_km=20.0),
                      site=SiteContext(vs30_mps=800.0))


def test_unpublished_damping_is_refused(asset):
    with pytest.raises(ValidationError, match="No interpolation rule"):
        asset.predict(imt="PGA", damping=0.02, **_ctx())


def test_period_must_lie_on_the_published_grid(asset):
    assert asset.predict(imt="PSA", period_s=0.2, **_ctx()).median > 0
    with pytest.raises(ValidationError, match="not on the published grid"):
        asset.predict(imt="PSA", period_s=0.35, **_ctx())
    with pytest.raises(ValidationError, match="needs period_s"):
        asset.predict(imt="PSA", **_ctx())


def test_unsupported_imt_is_refused(asset):
    with pytest.raises(ValidationError, match="supports IMTs"):
        asset.predict(imt="SD", period_s=1.0, **_ctx())


def test_typo_in_a_context_field_raises_rather_than_vanishing(asset):
    with pytest.raises(ValidationError, match="unknown field"):
        asset.predict(imt="PGA", rupture={"Mw": 6.0},
                      distance={"rhypo_km": 20.0}, site={"site_class": "rock"})


@pytest.mark.parametrize(
    "kw, what",
    [({"mw": 3.0}, "magnitude"), ({"mw": 8.5}, "magnitude"),
     ({"rhypo": 0.5}, "distance"), ({"rhypo": 400.0}, "distance")],
)
def test_outside_the_envelope_refuses_by_default_and_warns_when_allowed(kw, what):
    strict = Gmpe()
    with pytest.raises(OutOfScope, match=what):
        strict.predict(imt="PGA", **_ctx(**kw))

    lenient = Gmpe(allow_extrapolation=True)
    out = lenient.predict(imt="PGA", **_ctx(**kw))
    assert any(w.startswith("EXTRAPOLATION") for w in out.warnings)
    assert out.provenance["extrapolated"] is True


def test_in_scope_prediction_is_not_flagged_as_extrapolated(asset):
    out = asset.predict(imt="PGA", **_ctx())
    assert out.provenance["extrapolated"] is False


# ---- the synthetic fixture must announce itself ---------------------------


def test_every_verification_prediction_carries_a_synthetic_warning(asset):
    out = asset.predict(imt="PGA", **_ctx())
    assert any(w.startswith("SYNTHETIC") for w in out.warnings)


def test_provenance_records_what_the_specification_requires(asset):
    p = asset.predict(imt="PGA", **_ctx()).provenance
    for key in ("model_id", "imt", "damping", "distance_metric", "component",
                "native_log_base", "native_units", "extrapolated"):
        assert key in p
    assert p["spatial_correlation"] is None  # no correlation model, stated every time


# ---- the registry of 30 ---------------------------------------------------


def test_all_thirty_families_are_registered():
    reg = load_registry()
    assert len(reg) == 30
    assert sorted(s.raw["benchmark_id"] for s in reg.values()) == [f"{i:02d}" for i in range(1, 31)]


def test_no_published_model_can_predict_yet():
    """The refusal is the feature.

    Anchor-branch coefficients now exist for 22 of the 30, but `present` additionally
    requires a coefficient file with provenance AND a registered kernel, so every
    model still refuses. This test fails the day the first kernel lands; narrow it to
    the families still waiting rather than deleting it.
    """
    assert all(s != "present" for s in coefficient_status().values())
    for model_id in load_registry():
        with pytest.raises(CoefficientsUnavailable):
            Gmpe(model_id)


def test_the_refusal_says_what_source_material_is_missing():
    with pytest.raises(CoefficientsUnavailable) as exc:
        Gmpe("boore-atkinson-2008")
    msg = str(exc.value)
    assert "Source material needed" in msg
    assert "FLIN" in msg          # the specific table, not a generic complaint
    assert "2.240" in msg         # the catalogue section to look it up in
    assert "never be reconstructed from figures" in msg


def test_the_refusal_distinguishes_an_available_anchor_from_nothing_at_all():
    """Telling someone to find a table they already have wastes their afternoon."""
    anchor = str(pytest.raises(CoefficientsUnavailable, Gmpe, "akkar-bommer-2010").value)
    assert "ANCHOR_BRANCH_REPRODUCIBLE" in anchor
    assert "IS reproducible" in anchor
    assert "must still be refused rather than interpolated" in anchor

    partial = str(pytest.raises(CoefficientsUnavailable, Gmpe, "hong-2009b").value)
    assert "Not even the anchor branch is complete" in partial

    nothing = str(pytest.raises(CoefficientsUnavailable, Gmpe, "epri-1993a-ena").value)
    assert "ORIGINAL_TABLE_REQUIRED" in nothing
    assert "IS reproducible" not in nothing


def test_beyer_bommer_is_registered_as_a_benchmark_not_a_gmpe():
    """Douglas states its form and coefficients are not given, so it cannot ever
    become a median GMPE from the catalogue alone."""
    spec = load_registry()["beyer-bommer-2006"]
    assert spec.kind == "benchmark"
    assert spec.raw["supplies_median"] is False
    assert spec.coefficient_status == "unavailable_by_nature"


def test_unknown_model_id_is_rejected():
    with pytest.raises(ValidationError, match="unknown model_id"):
        Gmpe("boore-atkinson-2012")


def test_subduction_models_require_an_explicit_event_type():
    """Tectonic type is never inferred from depth at the framework level."""
    for model_id in ("youngs-1997", "atkinson-boore-2003", "zhao-2006-family",
                     "mcverry-2006", "hong-2009b"):
        spec = load_registry()[model_id]
        assert spec.raw["event_type"]["required"] is True
        assert "event_type" in spec.required["rupture"]


def test_kanno_depth_branch_is_not_recorded_as_a_tectonic_branch():
    spec = load_registry()["kanno-2006"]
    assert spec.raw["depth_branch"]["threshold_km"] == 30.0
    assert "not a tectonic-type branch" in spec.raw["depth_branch"]["warning"]


def test_non_mw_magnitude_scales_are_preserved():
    assert load_registry()["musson-1994-uk"].raw["magnitude"]["scale"] == "ML"


def test_damping_diverse_models_keep_their_published_ratios():
    reg = load_registry()
    assert reg["chen-yu-2008a"].damping == (0.005, 0.02, 0.07, 0.10, 0.20)
    assert reg["berge-thierry-2003"].damping == (0.05, 0.07, 0.10, 0.20)
    assert reg["cauzzi-faccioli-2008"].damping == (0.05, 0.10, 0.20, 0.30)


def test_unresolved_fields_are_surfaced_rather_than_defaulted():
    """A field the sources never state must stay RESOLVE, not acquire a default."""
    reg = load_registry()
    # Douglas codes Musson's component U (unknown); a third party's belief that it
    # is the larger component is not the authors' statement, so it stays open.
    assert "component" in reg["musson-1994-uk"].unresolved()
    # Grids stay open wherever no source matching Douglas's period count exists.
    open_grids = {m for m, s in reg.items() if "periods.grid" in s.unresolved()}
    assert open_grids == {
        "idriss-2008", "chen-yu-2008a", "beyer-bommer-2006", "danciu-tselentis-2007",
        "bindi-2010", "rupakhety-sigbjornsson-2009", "cotton-2008", "hong-2009b",
        "epri-1993a-ena", "free-1998-scr", "musson-1994-uk", "lussou-2001",
    }
    # Each open grid still says what Douglas does publish about it.
    for mid in open_grids:
        assert "published_grid_summary" in reg[mid].raw["periods"]
    assert sum(len(s.unresolved()) for s in reg.values()) == 20


def test_every_resolved_field_records_its_evidence():
    """A value that replaced RESOLVE without a cited source is a guess in disguise."""
    for mid, spec in load_registry().items():
        prov = spec.raw.get("resolution_provenance", {})
        for path, evidence in prov.items():
            assert "Douglas (2011)" in evidence or "OpenQuake" in evidence, (mid, path)
        # A resolved grid must be the published one, so it has to carry a source.
        grid = spec.raw["periods"].get("grid")
        if isinstance(grid, list):
            assert "periods.grid" in prov, mid
            assert grid == sorted(grid) and len(set(grid)) == len(grid), mid


def test_resolved_grids_match_douglas_period_counts_and_ranges():
    """OpenQuake tables are accepted only where they agree with Douglas Illustration 2.

    Where OpenQuake carries extra rows (Akkar & Bommer's 0.01-0.04 s and 4 s,
    Atkinson & Boore's and Youngs' 4 s) they are implementation extensions, not
    published periods, and must not appear in the grid.
    """
    reg = load_registry()
    expected = {
        "boore-atkinson-2008": (21, 0.01, 10.0), "campbell-bozorgnia-2008": (21, 0.01, 10.0),
        "abrahamson-silva-2008": (22, 0.01, 10.0), "chiou-youngs-2008": (22, 0.01, 10.0),
        "faccioli-2010": (22, 0.05, 20.0), "cauzzi-faccioli-2008": (400, 0.05, 20.0),
        "ambraseys-2005a": (61, 0.05, 2.5), "akkar-bommer-2010": (60, 0.05, 3.0),
        "berge-thierry-2003": (143, 0.029412, 10.0), "ghasemi-2009": (17, 0.05, 3.0),
        "youngs-1997": (11, 0.075, 3.0), "atkinson-boore-2003": (7, 0.04, 3.0),
        "zhao-2006-family": (20, 0.05, 5.0), "kanno-2006": (37, 0.05, 5.0),
        "mcverry-2006": (11, 0.075, 3.0), "lin-lee-2008": (27, 0.01, 5.0),
        "garcia-2005": (15, 0.04, 5.0), "arroyo-2010": (56, 0.04, 5.0),
    }
    for mid, (n, lo, hi) in expected.items():
        grid = reg[mid].raw["periods"]["grid"]
        assert (len(grid), grid[0], grid[-1]) == (n, lo, hi), mid
    assert 4.0 not in reg["akkar-bommer-2010"].raw["periods"]["grid"]
    assert 4.0 not in reg["atkinson-boore-2003"].raw["periods"]["grid"]


def test_faccioli_units_are_per_imt_and_the_catalogue_label_is_flagged():
    """Douglas says 'cm/s2' for a displacement spectrum, which cannot be right.

    The PGA anchor only gives a plausible value read as m/s2, and the displacement
    branch is in cm. Recording one unit for both would be a factor-of-100 error on
    one of them.
    """
    f = load_registry()["faccioli-2010"].raw
    assert f["internal_units"] == {"PGA": "m/s2", "SD": "cm"}
    assert "cm/s2" in f["units_warning"]
    a1, a2, a3, a4, a5 = -1.18, 0.559, -1.624, 0.018, 0.445
    log_pga = a1 + a2 * 6.0 + a3 * math.log10(10.0 + a4 * 10 ** (a5 * 6.0))
    assert 0.05 < 10 ** log_pga / 9.80665 < 0.5      # read as m/s2: 0.13 g


def test_danciu_site_term_is_ordinal_not_binary():
    """One coefficient e multiplies S in {0, 1, 2}; the anchor-only reading was wrong."""
    site = load_registry()["danciu-tselentis-2007"].raw["site"]
    assert [c["S"] for c in site["classes"]] == [0, 1, 2]
    assert [c["name"] for c in site["classes"]] == ["B", "C", "D"]


def test_the_two_flagged_log_base_ambiguities_are_now_resolved():
    """These were the dangerous ones: a form written 'log y' with no base stated.

    Both are log10. Reading either as ln would have been a factor-of-ln(10) error in
    the exponent, which looks entirely plausible in the output.
    """
    reg = load_registry()
    assert reg["akkar-bommer-2010"].raw["log_base"] == "log10"
    assert reg["lussou-2001"].raw["log_base"] == "log10"
    for mid in ("akkar-bommer-2010", "lussou-2001"):
        assert "log_base" not in reg[mid].unresolved()


def test_log_base_and_units_are_resolved_for_all_but_the_named_exceptions():
    reg = load_registry()
    unresolved_base = {m for m, s in reg.items() if "log_base" in s.unresolved()}
    assert unresolved_base == {"chen-yu-2008a"}
    # Beyer & Bommer report only ratios of sigma, which are the same in any base.
    assert reg["beyer-bommer-2006"].raw["log_base"] == "not_applicable"

    unresolved_units = {m for m, s in reg.items() if "internal_units" in s.unresolved()}
    assert unresolved_units == {"chen-yu-2008a", "epri-1993a-ena"}


def test_reproduction_status_matches_the_appendix_matrix():
    """The completeness gate, from the spec appendix section 11.2."""
    counts = Counter(r["status"] for r in reproduction_status().values())
    assert counts == {
        "ANCHOR_BRANCH_REPRODUCIBLE": 22,
        "ORIGINAL_TABLE_REQUIRED": 4,
        "PARTIAL": 3,
        "NOT_A_STANDALONE_GMPE": 1,
    }
    # No model's spectral table arrived with the anchors.
    assert all(
        s.reproduction["spectral_coefficients"] in ("ORIGINAL_TABLE_REQUIRED", "NOT_APPLICABLE")
        for s in load_registry().values()
    )


def test_anchor_branch_is_exposed_only_where_it_is_actually_complete():
    reg = load_registry()
    assert reg["akkar-bommer-2010"].anchor_branch == "PGA"
    # PARTIAL models have no usable anchor, so they must not advertise one.
    for mid in ("youngs-1997", "hong-2009b", "musson-1994-uk"):
        assert reg[mid].coefficient_status == "partial"
        assert reg[mid].anchor_branch is None


def test_partial_models_name_the_specific_coefficient_that_is_missing():
    """Each of the three fails for a different reason, and the reason is the point."""
    reg = load_registry()
    assert "c4" in reg["hong-2009b"].coefficients_needed            # intraslab log10(R) term
    assert "Dahle" in reg["musson-1994-uk"].coefficients_needed      # a second source entirely
    assert "event-type" in reg["youngs-1997"].coefficients_needed    # the Zt term


def test_bindi_rjb_repi_is_a_choice_of_coefficient_set_not_a_fallback():
    """The appendix publishes four separate PGA branches, which settles this."""
    dist = load_registry()["bindi-2010"].raw["distance"]
    assert "fallback" not in dist
    assert dist["alternative"]["field"] == "repi_km"
    assert "never convert between the two" in dist["note"]


def test_newly_resolved_model_specifics():
    reg = load_registry()
    # Component definition, stated outright for exactly one model.
    assert reg["mcverry-2006"].raw["component"] == "larger component"
    # Event-type enumeration, recoverable from an intraslab flag.
    assert reg["lin-lee-2008"].raw["event_type"]["classes"] == ["interface", "intraslab"]
    assert "event_type.classes" not in " ".join(reg["lin-lee-2008"].unresolved())
    # The published depth threshold in the Zhao-family source/path term.
    assert reg["zhao-2006-family"].raw["depth_branch"]["hc_km"] == 15.0
    # Site classes defined by depth to rock, not by Vs30.
    epri = reg["epri-1993a-ena"].raw["site"]
    assert epri["class_definitions"]["intermediate_soil"] == "depth to rock 20-100 m"
    # Five site classes, named after the coefficients that reveal them.
    assert len(reg["zhao-2006-family"].raw["site"]["classes"]) == 5


def test_zhao_family_version_is_recorded_as_probable_not_settled():
    """Coefficient naming points at Zhao (2006), but the appendix does not say so.

    Merging versions of this family is the likeliest way to corrupt it, so the
    inference is recorded as an inference.
    """
    note = load_registry()["zhao-2006-family"].raw["version_note"]
    assert "probable, not settled" in note


def test_suspect_aleatory_labelling_is_preserved_as_a_warning():
    """Douglas's intra/inter naming for Danciu looks transposed.

    Silently 'correcting' it would swap tau and phi, so the doubt is carried
    forward instead of resolved.
    """
    al = load_registry()["danciu-tselentis-2007"].raw["aleatory"]
    assert "transposed" in al["warning"]
    assert "tau_intra_as_labelled" in al


def test_aleatory_vocabularies_are_not_normalised_onto_one_scheme():
    """Bindi publishes event/station, Hong adds orientation, several give sigma only."""
    reg = load_registry()
    assert "sigma_station" in reg["bindi-2010"].raw["aleatory"]["horizontal_rjb"]
    assert "sigma_orientation" in reg["hong-2009b"].raw["aleatory"]["interface_geometric_mean"]
    assert reg["youngs-1997"].raw["aleatory"]["sigma_total"] == "1.45 - 0.1*M"


def test_coverage_matrix_spans_the_heterogeneity_the_set_was_chosen_for():
    rows = coverage_matrix()
    assert len(rows) == 30
    assert {t for r in rows for t in r["tectonic_types"]} == {
        "ACTIVE_SHALLOW_CRUST", "SUBDUCTION_INTERFACE",
        "SUBDUCTION_INTRASLAB", "STABLE_CONTINENTAL",
    }
    assert {"Rjb", "Rrup", "Rhypo", "Repi", "Rc"} <= {r["distance_metric"] for r in rows}
    assert {"PGA", "PSA", "SA", "PSV", "SD"} <= {i for r in rows for i in r["imts"]}
    assert len({d for r in rows for d in r["damping"]}) > 1
    assert {"vs30_continuous", "categorical", "fixed_condition"} <= {
        r["site_parameterization"] for r in rows
    }


def test_missing_sources_reports_every_family_with_what_to_go_and_find():
    report = missing_sources()
    assert len(report) == 30
    assert all(r["needed"] and r["needed"] != "unspecified" for r in report)
    cauzzi = next(r for r in report if r["model_id"] == "cauzzi-faccioli-2008")
    assert "ELECTRONIC SUPPLEMENTARY MATERIAL" in cauzzi["needed"]


# ---- helper ---------------------------------------------------------------


def _ctx(mw=6.0, rhypo=20.0, site_class="rock"):
    return {
        "rupture": RuptureContext(mw=mw),
        "distance": DistanceContext(rhypo_km=rhypo),
        "site": SiteContext(site_class=site_class),
    }
