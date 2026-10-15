"""Randfälle (DEMO-PLAYBOOK §1c): kleinste Instanzen, gleiche Alternativen, ein Kriterium ohne Streuung, großes Rauschen, keine Störung."""

import numpy as np
import pytest

import vgl_constants as C
import vgl_evaluation as ev
import vgl_methods as md
import vgl_score as sc
from conftest import MINI_S, MINI_W
from entscheidungsmatrix_generator import SplitMix64


def test_smallest_and_largest_instances_run_for_every_decider():
    for decider in C.DECIDERS:
        a = ev.analyse(ev.Settings(n=C.N_MIN, m=C.M_MIN, decider=decider, sigma=0.6, perturb=0.1, seed=3))
        assert len(a.picks) == 10 and a.tau_matrix.shape == (10, 10) and np.isfinite(a.tau_matrix).all()
        b = ev.analyse(ev.Settings(n=C.N_MAX, m=C.M_MAX, decider=decider, rho=0.5, seed=9))
        assert len(b.picks) == 10 and 0 <= b.best < C.N_MAX


@pytest.mark.parametrize("n", [2, 3])
def test_two_and_three_alternatives(n):
    b = ev.run_matrix(5, n, 3, 0.0, 0.3, 0.05)
    for m in C.METHODS:
        s = b.scores[m]
        assert s.shape == (n,) and np.isfinite(s).all() and np.isfinite(b.scores_perturbed[m]).all(), m
    u, j = ev.judge(b, "guete_linear")
    assert all(0.0 <= r["regret"] <= 1.0 and 1 <= r["rank"] <= n for r in j.values())


def test_identical_alternatives_get_identical_scores_and_the_smaller_number_wins():
    S = np.vstack([MINI_S[0], MINI_S[0], MINI_S[2]])
    for name, f in md.WEIGHTED_FUNCTIONS.items():
        s = f(S, MINI_W)
        assert s[0] == pytest.approx(s[1]), name
    assert sc.top1(md.sum_nom(S, MINI_W)) == 0


def test_a_criterion_without_spread_has_no_effect_on_min_max_summation_and_gives_no_nan():
    S = np.array([[80.0, -1000.0, 10.0], [50.0, -3000.0, 10.0], [20.0, -800.0, 10.0]])
    w = np.array([0.4, 0.4, 0.2])
    for name in ("sum_minmax", "prom_v", "electre", "topsis_vec"):
        assert np.isfinite(md.WEIGHTED_FUNCTIONS[name](S, w)).all(), name
    assert md.sum_minmax(S, w) == pytest.approx(md.sum_minmax(S[:, :2], np.array([0.4, 0.4])))


def test_very_large_noise_keeps_weights_a_distribution():
    w = sc.noisy_weights(np.array([0.5, 0.3, 0.2]), 8.0, SplitMix64(11))
    assert w.sum() == pytest.approx(1.0) and np.isfinite(w).all() and (w >= 0).all()


def test_no_perturbation_means_no_second_set_of_scores():
    assert ev.run_matrix(5, 12, 5, 0.0, 0.0, 0.0).scores_perturbed is None


def test_the_best_of_the_floor_decider_is_never_an_excluded_alternative():
    for seed in range(1, 25):
        a = ev.analyse(ev.Settings(decider="untergrenze", seed=seed))
        assert a.u[a.best] > C.EXCLUDED_BELOW
