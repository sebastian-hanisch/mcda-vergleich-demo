"""Nenn-Spannweite, Normierungen, Platz des Besten, Nutzenverlust, Entscheider, Rauschen und Störung."""

import numpy as np
import pytest

import vgl_constants as C
import vgl_score as sc
from conftest import MINI_D, MINI_NOM, MINI_RAW, MINI_S
from entscheidungsmatrix_generator import SplitMix64


def test_nominal_bounds_flip_the_down_criteria():
    lo, hi = sc.nominal_bounds(4)
    assert lo.tolist() == [0.0, -5000.0, 0.0, -600.0] and hi.tolist() == [100.0, -800.0, 100.0, -5.0]
    assert sc.directions(4).tolist() == [1.0, -1.0, 1.0, -1.0]


def test_normalisations():
    assert sc.nominal_normalise(MINI_S) == pytest.approx(MINI_NOM)
    assert sc.minmax_normalise(MINI_S).ravel() == pytest.approx([1.0, 2000 / 2200, 0.5, 0.0, 0.0, 1.0])
    assert sc.vector_normalise(np.array([[3.0, 0.0], [4.0, 0.0]])).tolist() == [[0.6, 0.0], [0.8, 0.0]]
    assert sc.minmax_normalise(np.array([[1.0, 5.0], [3.0, 5.0]])).tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_order_and_top1_break_ties_by_the_smaller_index_after_rounding_to_nine_places():
    s = np.array([0.5, 1.0, 1.0 + 7e-16, 0.2])
    assert sc.order_of(s).tolist() == [1, 2, 0, 3] and sc.top1(s) == 1
    assert sc.top1(np.array([0.1, 0.2, 0.2 + 1e-8])) == 2


def test_rank_of_best():
    u = np.array([0.1, 0.9, 0.5])
    assert sc.rank_of_best(u, np.array([0.3, 0.8, 0.1])) == 1
    assert sc.rank_of_best(u, np.array([0.9, 0.1, 0.5])) == 3
    assert sc.rank_of_best(u, np.array([0.9, 0.5, 0.1])) == 2


def test_kendall_tau_b_by_hand():
    assert sc.kendall_tau_b(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)
    assert sc.kendall_tau_b(np.array([1, 2, 3]), np.array([3, 2, 1])) == pytest.approx(-1.0)
    assert sc.kendall_tau_b(np.array([1, 1, 1]), np.array([1, 2, 3])) == 0.0
    assert sc.kendall_tau_b(np.array([1, 2, 3, 4]), np.array([1, 3, 2, 4])) == pytest.approx(4 / 6)
    assert sc.kendall_tau_b(np.array([1, 1, 2]), np.array([1, 2, 3])) == pytest.approx(2 / np.sqrt(2 * 3))     # Gleichstand: tau-b statt tau-a


def test_regret():
    u = np.array([1.0, 0.4, 0.0])
    assert sc.regret(u, 0) == 0.0 and sc.regret(u, 1) == pytest.approx(0.6) and sc.regret(u, 2) == pytest.approx(1.0)
    assert sc.regret(np.array([0.5, 0.5]), 1) == 0.0                                                         # keine Spanne: kein Verlust


def test_excluded_alternatives_count_as_full_loss_and_stay_out_of_the_span():
    u = np.array([0.9, 0.3, C.UNDER_FLOOR])
    assert sc.allowed(u).tolist() == [True, True, False]
    assert sc.regret(u, 2) == 1.0
    assert sc.regret(u, 1) == pytest.approx(1.0)                                                             # Spanne nur über 0,9 und 0,3
    assert sc.regret(np.array([0.9, 0.6, 0.2, C.UNDER_FLOOR]), 1) == pytest.approx(0.3 / 0.7)


def test_decider_utilities_on_known_goodness():
    G = np.array([[0.25, 1.0], [0.04, 0.5], [0.09, 0.0]])
    w = np.array([0.5, 0.5])
    assert sc.decider_utility("guete_linear", G, MINI_S, w) == pytest.approx([0.625, 0.27, 0.045])
    assert sc.decider_utility("guete_konkav", G, MINI_S, w) == pytest.approx([0.75, 0.1 + 0.5 * np.sqrt(0.5), 0.15])
    assert sc.decider_utility("guete_cobb", G, MINI_S, w) == pytest.approx([0.5 * np.log(0.25), 0.5 * (np.log(0.04) + np.log(0.5)), 0.5 * (np.log(0.09) + np.log(1e-12))])
    assert sc.decider_utility("roh_linear", G, MINI_S, np.array([0.6, 0.4])) == pytest.approx([0.861, 0.4905, 0.52], abs=5e-4)


def test_floor_decider_excludes_a_criterion_below_the_floor_inclusively():
    G = np.array([[0.1, 0.9], [0.0999, 0.9], [0.5, 0.5]])
    u = sc.decider_utility("untergrenze", G, MINI_S, np.array([0.5, 0.5]))
    assert u[0] == pytest.approx(0.5) and u[1] == C.UNDER_FLOOR and u[2] == pytest.approx(0.5)


def test_unknown_decider_raises():
    with pytest.raises(ValueError):
        sc.decider_utility("unbekannt", np.ones((2, 2)), MINI_S[:2], np.array([0.5, 0.5]))


def test_noise_is_off_at_sigma_zero_and_normalised_otherwise():
    w = np.array([0.5, 0.3, 0.2])
    assert sc.noisy_weights(w, 0.0, SplitMix64(1)).tolist() == w.tolist()
    v = sc.noisy_weights(w, 0.6, SplitMix64(1))
    assert v.sum() == pytest.approx(1.0) and (v > 0).all() and not np.allclose(v, w)
    z = SplitMix64(1).normal(3)
    ref = w * np.exp(0.6 * z)
    assert v == pytest.approx(ref / ref.sum())
    assert sc.noisy_weights(w, 0.6, SplitMix64(1)).tolist() == v.tolist()


def test_perturbation_is_lognormal_and_keeps_signs():
    assert sc.perturb_raw(MINI_RAW, 0.0, SplitMix64(1)).tolist() == MINI_RAW.tolist()
    p = sc.perturb_raw(MINI_RAW, 0.1, SplitMix64(1))
    z = SplitMix64(1).normal(6).reshape(3, 2)
    assert p == pytest.approx(MINI_RAW * np.exp(0.1 * z))
    assert (p > 0).all() and (p * MINI_D > 0).sum() == 3


def test_split_inputs_outputs_and_unit_normalisation():
    X, Y = sc.split_inputs_outputs(MINI_RAW, MINI_D)
    assert X.ravel().tolist() == [1000.0, 3000.0, 800.0] and Y.ravel().tolist() == [80.0, 50.0, 20.0]
    Xn, Yn = sc.normalise_units(X, Y)
    assert Xn.ravel().tolist() == pytest.approx([1000 / 3000, 1.0, 800 / 3000]) and Yn.ravel().tolist() == pytest.approx([1.0, 0.625, 0.25])
