"""DEA (CCR, Kreuzeffizienz) und SMAA (Ziehung, Rangmatrix, erwarteter Rang) an der Mini-Instanz von Hand."""

import numpy as np
import pytest

import vgl_dea as dea
import vgl_smaa as sm
from conftest import MINI_NOM
from entscheidungsmatrix_generator import SplitMix64


def test_ccr_efficiency_with_one_input_and_one_output():
    X = np.array([[1000.0], [3000.0], [800.0]])
    Y = np.array([[80.0], [50.0], [20.0]])
    theta, U, V = dea.dea_ccr(X, Y)
    assert theta == pytest.approx([1.0, (50 / 3000) / 0.08, (20 / 800) / 0.08])
    assert [float(v[0] * X[i, 0]) for i, v in enumerate(V)] == pytest.approx([1.0, 1.0, 1.0])           # Normierung v·x_o = 1
    assert [float(u[0] * Y[i, 0]) for i, u in enumerate(U)] == pytest.approx(theta)


def test_cross_efficiency_equals_theta_with_a_single_input_and_output():
    X = np.array([[1.0], [2.0], [4.0]])
    Y = np.array([[1.0], [3.0], [4.0]])                                                                 # Verhältnisse 1, 1,5, 1
    assert dea.cross_efficiency(X, Y) == pytest.approx([2 / 3, 1.0, 2 / 3])


def test_cross_matrix_by_hand():
    X = np.array([[1.0], [2.0]])
    Y = np.array([[1.0], [1.0]])
    ce = dea.cross_matrix(X, Y, [np.array([2.0]), np.array([1.0])], [np.array([1.0]), np.array([1.0])])
    assert ce.tolist() == [[2.0, 1.0], [1.0, 0.5]]                                                      # (u·y_j) / (v·x_j) mit den Gewichten von k


def test_dea_with_two_outputs_and_two_inputs_is_bounded_by_one():
    rng = np.random.default_rng(4)
    X, Y = rng.random((7, 2)) + 0.1, rng.random((7, 2)) + 0.1
    theta, U, V = dea.dea_ccr(X, Y)
    assert (theta <= 1 + 1e-9).all() and (theta > 0).all() and theta.max() == pytest.approx(1.0)
    ce = dea.cross_efficiency(X, Y)
    assert (ce <= 1 + 1e-9).all() and (ce > 0).all()


def test_weights_are_uniform_over_the_simplex():
    W = sm.sample_weights(4000, 3, SplitMix64(7))
    assert W.shape == (4000, 3) and W.sum(axis=1) == pytest.approx(np.ones(4000)) and (W > 0).all()
    assert W.mean(axis=0) == pytest.approx([1 / 3] * 3, abs=0.01)
    assert W.var(axis=0) == pytest.approx([1 / 18] * 3, abs=0.005)                                      # Dirichlet(1,1,1): Var = (1/3)(2/3)/4
    assert sm.sample_weights(10, 3, SplitMix64(7)).tolist() == W[:10].tolist()                          # Der Strom hängt nur vom Seed ab


def test_rank_matrix_and_expected_rank_on_two_corner_draws():
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    R = sm.rank_matrix(MINI_NOM, W)
    assert R.tolist() == [[0, 1], [1, 2], [2, 0]]                                                       # Zeile = Alternative, Spalte = Ziehung
    assert sm.expected_rank(MINI_NOM, W).tolist() == [1.5, 2.5, 2.0]


def test_rank_ties_go_to_the_smaller_index():
    V = np.array([[0.5, 0.5], [0.5, 0.5], [0.1, 0.1]])
    assert sm.rank_matrix(V, np.array([[0.5, 0.5]])).ravel().tolist() == [0, 1, 2]
