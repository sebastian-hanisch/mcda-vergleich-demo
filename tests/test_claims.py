"""Jede Zahl der README, gemessen mit den eigenen Experimenten der Demo (200 feste Matrizen, SWEEP_SEEDS). Die Werte sind deterministisch (SplitMix64)."""

import pytest

import vgl_constants as C
import vgl_evaluation as ev

TOL = 0.0006


@pytest.fixture(scope="module")
def exp():
    return {"methods": ev.experiment_methods(), "weights": ev.experiment_weights(), "agreement": ev.experiment_agreement(), "stability": ev.experiment_stability()}


def _row(res, decider, method):
    return next(r for r in res["rows"] if r["decider"] == decider and r["method"] == method)


def _hits(res, decider, methods=C.METHODS):
    return [_row(res, decider, m)["hit"] for m in methods]


def _cell(res, sigma, rho):
    return next(c for c in res["cells"] if c["sigma"] == sigma and c["rho"] == rho)


def test_the_sum_and_promethee_lead_and_the_weight_free_methods_trail(exp):
    res = exp["methods"]
    assert res["matrices"] == 200 and (res["n"], res["m"]) == (12, 5)
    assert _hits(res, "guete_linear") == pytest.approx([0.545, 0.775, 0.79, 0.70, 0.71, 0.78, 0.705, 0.68, 0.475, 0.54], abs=TOL)
    assert [_row(res, "guete_linear", m)["top3"] for m in C.METHODS] == pytest.approx([0.83, 0.985, 0.99, 0.945, 0.935, 0.99, 0.96, 0.975, 0.765, 0.835], abs=TOL)
    assert [round(_row(res, "guete_linear", m)["tau"], 2) for m in C.METHODS] == pytest.approx([0.55, 0.89, 0.88, 0.81, 0.82, 0.89, 0.80, 0.83, 0.49, 0.56], abs=0.0001)


def test_the_hit_rate_depends_on_the_decider(exp):
    res = exp["methods"]
    assert _hits(res, "guete_konkav", ("sum_nom", "prom_v", "ahp_dist", "dea_ce")) == pytest.approx([0.81, 0.83, 0.63, 0.505], abs=TOL)
    assert _hits(res, "guete_cobb", ("sum_nom", "prom_v", "ahp_dist", "dea_ce")) == pytest.approx([0.765, 0.78, 0.57, 0.535], abs=TOL)
    assert _hits(res, "roh_linear", ("sum_nom", "prom_v", "ahp_dist", "dea_ce")) == pytest.approx([1.0, 0.895, 0.535, 0.455], abs=TOL)
    assert _hits(res, "untergrenze", ("sum_nom", "sum_minmax", "prom_v", "ahp_dist", "dea_ce")) == pytest.approx([0.715, 0.715, 0.705, 0.58, 0.515], abs=TOL)
    assert max(_hits(res, "untergrenze")) == pytest.approx(0.715)


def test_the_floor_decider_costs_the_weighted_methods_most(exp):
    res = exp["methods"]
    methods = ("sum_nom", "prom_v", "topsis_vec", "electre", "ahp_dist", "equal", "dea_ce", "smaa")
    assert [_row(res, "guete_linear", m)["regret"] for m in methods] == pytest.approx([0.016, 0.015, 0.029, 0.031, 0.055, 0.101, 0.128, 0.10], abs=TOL)
    assert [_row(res, "untergrenze", m)["regret"] for m in methods] == pytest.approx([0.143, 0.158, 0.156, 0.24, 0.282, 0.121, 0.191, 0.111], abs=TOL)
    weighted = [_row(res, "untergrenze", m)["regret"] for m in C.WEIGHTED]
    assert (min(weighted), max(weighted)) == pytest.approx((0.143, 0.282), abs=TOL)


def test_noisy_weights_cost_the_weighted_methods_hits(exp):
    res = exp["weights"]
    sig = [_cell(res, s, 0.0)["hit"] for s in C.SIGMA_CHOICES]
    assert [h["sum_nom"] for h in sig] == pytest.approx([0.775, 0.75, 0.635, 0.54], abs=TOL)
    assert [h["electre"] for h in sig] == pytest.approx([0.705, 0.67, 0.535, 0.445], abs=TOL)
    assert [h["prom_v"] for h in sig] == pytest.approx([0.78, 0.72, 0.655, 0.58], abs=TOL)
    for m in ("equal", "dea_ce", "smaa"):
        assert [h[m] for h in sig] == pytest.approx([sig[0][m]] * 4, abs=1e-12)
    assert sig[0]["equal"] == pytest.approx(0.545, abs=TOL) and sig[0]["dea_ce"] == pytest.approx(0.475, abs=TOL) and sig[0]["smaa"] == pytest.approx(0.54, abs=TOL)


def test_correlated_criteria_pull_the_methods_together(exp):
    res = exp["weights"]
    corr = _cell(res, 0.0, 0.5)
    assert [corr["hit"][m] for m in ("sum_nom", "prom_v", "electre", "equal", "dea_ce", "smaa")] == pytest.approx([0.84, 0.845, 0.75, 0.755, 0.735, 0.76], abs=TOL)
    assert (ev.hit_spread(_cell(res, 0.0, 0.0)), ev.hit_spread(corr)) == pytest.approx((0.315, 0.11), abs=TOL)
    assert [round(_cell(res, s, 0.0)["tau"]["sum_nom"], 2) for s in C.SIGMA_CHOICES] == pytest.approx([0.89, 0.83, 0.74, 0.65], abs=0.0001)


def test_agreement_among_weighted_methods_is_not_evidence(exp):
    res = exp["agreement"]
    cells = [_cell(res, s, 0.0) for s in C.SIGMA_CHOICES]
    assert [c["weighted_agree"] for c in cells] == pytest.approx([0.345, 0.39, 0.365, 0.33], abs=TOL)
    assert [c["weighted_right"] for c in cells] == pytest.approx([1.0, 0.962, 0.877, 0.758], abs=TOL)
    assert [c["sum_right"] for c in cells] == pytest.approx([0.775, 0.75, 0.635, 0.54], abs=TOL)
    assert [c["sum_right_disagree"] for c in cells] == pytest.approx([0.656, 0.615, 0.496, 0.433], abs=TOL)
    first, corr = cells[0], _cell(res, 0.0, 0.5)
    assert (corr["sum_right"], corr["sum_right_disagree"]) == pytest.approx((0.84, 0.614), abs=TOL)
    assert (first["all_agree"], first["all_right"], round(first["mean_tau"], 2)) == pytest.approx((0.18, 1.0, 0.66), abs=TOL)
    assert (corr["all_agree"], corr["all_right"], round(corr["mean_tau"], 2)) == pytest.approx((0.465, 1.0, 0.82), abs=TOL)


def test_stability_differs_but_is_not_correctness(exp):
    rows = {r["perturb"]: r["stay"] for r in exp["stability"]["rows"]}
    methods = ("topsis_vec", "ahp_dist", "prom_v", "sum_nom", "equal", "electre")
    assert [rows[0.05][m] for m in methods] == pytest.approx([0.945, 0.94, 0.94, 0.885, 0.855, 0.82], abs=TOL)
    assert [rows[0.1][m] for m in methods] == pytest.approx([0.85, 0.875, 0.855, 0.785, 0.765, 0.705], abs=TOL)
    assert min(rows[0.05], key=rows[0.05].get) == "electre" and min(rows[0.1], key=rows[0.1].get) == "electre"
