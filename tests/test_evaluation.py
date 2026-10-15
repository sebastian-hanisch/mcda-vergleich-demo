"""Auswertung: Fall der App (analyse) und Aufbau der vier Experimente auf wenigen Matrizen (die Zahlen selbst stehen in test_claims.py)."""

import numpy as np
import pytest

import vgl_constants as C
import vgl_evaluation as ev
import vgl_score as sc
from entscheidungsmatrix_generator import SplitMix64, make_matrix

SEEDS = (200000, 200001, 200002)


def test_analysis_is_consistent_with_the_methods_and_the_oracle():
    a = ev.analyse(ev.Settings(seed=24))
    x = make_matrix(12, 5, 0.0, SplitMix64(24))
    u = sc.decider_utility("guete_linear", x.goodness, x.signed, x.weights)
    assert a.best == int(np.argmax(u)) and a.u.tolist() == u.tolist()
    assert set(a.picks) == set(C.METHODS) and all(sc.top1(a.scores[m]) == a.picks[m] for m in C.METHODS)
    assert a.hits == sum(a.picks[m] == a.best for m in C.METHODS)
    assert all(a.ranks[m] == 1 for m in C.METHODS if a.picks[m] == a.best) and all(a.ranks[m] > 1 for m in C.METHODS if a.picks[m] != a.best)
    assert all(a.regrets[m] == 0.0 for m in C.METHODS if a.picks[m] == a.best)
    assert a.distinct == len(set(a.picks.values())) and a.all_agree == (a.distinct == 1)
    assert a.tau_matrix.shape == (10, 10) and np.diag(a.tau_matrix) == pytest.approx(np.ones(10)) and (a.tau_matrix == a.tau_matrix.T).all()
    assert a.picks_perturbed is None


def test_exact_weights_are_the_hidden_weights_and_noise_changes_only_the_weighted_methods():
    exact = ev.analyse(ev.Settings(seed=24))
    noisy = ev.analyse(ev.Settings(seed=24, sigma=0.6))
    assert exact.w_hat.tolist() == exact.x.weights.tolist() and not np.allclose(noisy.w_hat, noisy.x.weights)
    for m in ("equal", "dea_ce", "smaa"):
        assert noisy.scores[m].tolist() == exact.scores[m].tolist(), m
    assert any(noisy.scores[m].tolist() != exact.scores[m].tolist() for m in C.WEIGHTED)


def test_perturbation_leaves_the_exact_run_unchanged_and_adds_a_second_set_of_winners():
    plain = ev.analyse(ev.Settings(seed=5))
    pert = ev.analyse(ev.Settings(seed=5, perturb=0.1))
    assert pert.scores.keys() == plain.scores.keys() and all(pert.scores[m].tolist() == plain.scores[m].tolist() for m in C.METHODS)
    assert set(pert.picks_perturbed) == set(C.METHODS)
    b = ev.run_matrix(5, 12, 5, 0.0, 0.0, 0.1)
    assert all(b.scores_perturbed[m].tolist() != b.scores[m].tolist() for m in C.METHODS)


def test_the_floor_decider_marks_excluded_picks_as_full_regret():
    a = ev.analyse(ev.Settings(seed=14, decider="untergrenze"))
    assert a.u[a.best] > C.EXCLUDED_BELOW
    assert sum(r == 1.0 for r in a.regrets.values()) >= 5 and all(a.regrets[m] == 1.0 for m in C.METHODS if a.u[a.picks[m]] <= C.EXCLUDED_BELOW)


def test_agreement_flags():
    a = ev.analyse(ev.Settings(seed=23, sigma=0.6))
    assert a.weighted_agree and a.agreed_right is False and not a.all_agree
    b = ev.analyse(ev.Settings(seed=22, rho=0.5))
    assert b.all_agree and b.weighted_agree and b.agreed_right is True
    c = ev.analyse(ev.Settings(seed=24))
    assert not c.weighted_agree and c.agreed_right is None


def test_mean_off_diagonal():
    assert ev.mean_off_diagonal(np.array([[1.0, 0.5], [0.3, 1.0]])) == pytest.approx(0.4)
    assert ev.mean_off_diagonal(np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])) == pytest.approx(0.4)


def test_formatting_helpers():
    assert ev.de(0.125, 2) == "0,12" or ev.de(0.125, 2) == "0,13"
    assert ev.de(3.0, 1) == "3,0" and ev.pct(0.775) == "77,5" and ev.pct(0.05, 0) == "5"


def test_experiments_have_the_documented_structure_on_a_few_matrices():
    m = ev.experiment_methods(deciders=("guete_linear", "untergrenze"), seeds=SEEDS)
    assert len(m["rows"]) == 20 and m["matrices"] == 3 and all(0.0 <= r["hit"] <= r["top3"] <= 1.0 and 0.0 <= r["regret"] <= 1.0 for r in m["rows"])
    w = ev.experiment_weights(seeds=SEEDS)
    assert [(c["sigma"], c["rho"]) for c in w["cells"]] == [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0), (1.0, 0.0), (0.0, 0.5)]
    assert set(w["cells"][0]["hit"]) == set(C.METHODS)
    g = ev.experiment_agreement(seeds=SEEDS)
    assert len(g["cells"]) == 5 and all(0.0 <= c["weighted_agree"] <= 1.0 and c["all_agree"] <= c["weighted_agree"] for c in g["cells"])
    s = ev.experiment_stability(seeds=SEEDS)
    assert [r["perturb"] for r in s["rows"]] == [0.05, 0.1] and all(0.0 <= v <= 1.0 for r in s["rows"] for v in r["stay"].values())


def test_exact_weights_cell_equals_the_methods_experiment_for_the_linear_decider():
    m = ev.experiment_methods(deciders=("guete_linear",), seeds=SEEDS)
    w = ev.experiment_weights(sigmas=(0.0,), rhos=(0.0,), seeds=SEEDS)
    assert len(w["cells"]) == 1
    assert {r["method"]: r["hit"] for r in m["rows"]} == w["cells"][0]["hit"]


def test_weights_grid_has_no_duplicate_for_rho_zero():
    assert ev._config_grid((0.0, 0.6), (0.0, 0.5)) == [(0.0, 0.0), (0.6, 0.0), (0.0, 0.5)]
