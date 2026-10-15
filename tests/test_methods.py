"""Die sieben gewichteten Verfahren und gleiche Gewichte an der Mini-Instanz von Hand und gegen unabhängige Schleifen auf Zufallsmatrizen."""

import numpy as np
import pytest

import vgl_methods as md
from conftest import MINI_NOM, MINI_S, MINI_W
from entscheidungsmatrix_generator import SplitMix64, make_matrix


def test_weighted_sum_on_the_nominal_range():
    assert md.sum_nom(MINI_S, MINI_W) == pytest.approx([0.861, 0.4905, 0.52], abs=5e-4)
    assert md.sum_nom(MINI_S, MINI_W) == pytest.approx(MINI_NOM @ MINI_W)


def test_weighted_sum_on_min_max():
    assert md.sum_minmax(MINI_S, MINI_W) == pytest.approx([0.9636, 0.3, 0.4], abs=5e-4)


def test_equal_weights_are_the_mean_on_the_nominal_range():
    assert md.equal_weights(MINI_S) == pytest.approx(MINI_NOM.mean(axis=1))
    assert md.equal_weights(MINI_S) == pytest.approx(md.sum_nom(MINI_S, np.array([0.5, 0.5])))


def test_promethee_net_flow_on_the_mini_instance():
    assert md.prom_v(MINI_S, MINI_W) == pytest.approx([0.7636, -0.4, -0.3636], abs=5e-4)
    assert md.prom_v(MINI_S, MINI_W).sum() == pytest.approx(0.0, abs=1e-12)


def test_electre_copeland_numbers():
    assert md.electre(MINI_S, MINI_W).tolist() == [2.0, 0.0, -2.0]


def test_electre_threshold_is_inclusive():
    S = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert md.electre(S, np.array([0.6, 0.4])).tolist() == [1.0, -1.0]          # 0,6 >= c* = 0,6 (inklusive): die erste übertrifft die zweite, nicht umgekehrt
    assert md.electre(S, np.array([0.5, 0.5])).tolist() == [0.0, 0.0]


def test_topsis_nominal_and_vector():
    assert md.topsis_nom(MINI_S, MINI_W) == pytest.approx([0.8345, 0.4927, 0.4652], abs=5e-4)
    assert md.topsis_vec(MINI_S, MINI_W) == pytest.approx([0.9479, 0.3626, 0.4195], abs=5e-4)


def test_topsis_vector_of_identical_alternatives_is_one_half():
    assert md.topsis_vec(np.array([[5.0, -2.0], [5.0, -2.0]]), MINI_W).tolist() == [0.5, 0.5]


def test_ahp_distributive_on_the_mini_instance():
    assert md.ahp_dist(MINI_S, MINI_W) == pytest.approx([0.4748, 0.2516, 0.2735], abs=5e-4)
    assert md.ahp_dist(MINI_S, MINI_W).sum() == pytest.approx(1.0)


def test_weighted_scores_cover_exactly_the_seven_weighted_methods():
    out = md.weighted_scores(MINI_S, MINI_W)
    assert list(out) == ["sum_nom", "sum_minmax", "topsis_vec", "topsis_nom", "prom_v", "electre", "ahp_dist"]
    assert out["sum_nom"] == pytest.approx(md.sum_nom(MINI_S, MINI_W))


# ----- unabhängige Schleifen auf Zufallsmatrizen

def _loop_promethee(N, w, p=0.5):
    n, m = N.shape
    phi = np.zeros(n)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            pi_ab = sum(w[j] * min(1.0, max(0.0, (N[a, j] - N[b, j]) / p)) for j in range(m))
            pi_ba = sum(w[j] * min(1.0, max(0.0, (N[b, j] - N[a, j]) / p)) for j in range(m))
            phi[a] += (pi_ab - pi_ba) / (n - 1)
    return phi


def _loop_electre(N, w, cstar=0.6):
    n, m = N.shape
    out = np.zeros(n)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if sum(w[j] for j in range(m) if N[a, j] >= N[b, j]) >= cstar - 1e-12:
                out[a] += 1
                out[b] -= 1
    return out


def _loop_topsis(S, w):
    n, m = S.shape
    V = np.zeros((n, m))
    for j in range(m):
        norm = sum(S[i, j] ** 2 for i in range(n)) ** 0.5
        for i in range(n):
            V[i, j] = S[i, j] / norm * w[j]
    ideal = [max(V[i, j] for i in range(n)) for j in range(m)]
    anti = [min(V[i, j] for i in range(n)) for j in range(m)]
    out = np.zeros(n)
    for i in range(n):
        dp = sum((V[i, j] - ideal[j]) ** 2 for j in range(m)) ** 0.5
        dm = sum((V[i, j] - anti[j]) ** 2 for j in range(m)) ** 0.5
        out[i] = dm / (dp + dm)
    return out


@pytest.mark.parametrize("seed", [3, 8, 21])
def test_methods_match_independent_loops_on_random_matrices(seed):
    x = make_matrix(8, 4, 0.0, SplitMix64(seed))
    S, w = np.asarray(x.signed), np.asarray(x.weights)
    lo, hi = S.min(axis=0), S.max(axis=0)
    N = (S - lo) / (hi - lo)
    assert md.prom_v(S, w) == pytest.approx(_loop_promethee(N, w), abs=1e-12)
    assert md.electre(S, w) == pytest.approx(_loop_electre(N, w), abs=1e-12)
    assert md.topsis_vec(S, w) == pytest.approx(_loop_topsis(S, w), abs=1e-12)
    assert md.sum_minmax(S, w) == pytest.approx([sum(w[j] * N[i, j] for j in range(4)) for i in range(8)], abs=1e-12)
