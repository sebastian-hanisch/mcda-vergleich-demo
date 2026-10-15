"""Orakel: Verfahrensvergleich gegen Identitäten und Eigenschaften (DEMO-PLAYBOOK §12).

1. Rohwert-Entscheider: Die Summe auf der Nenn-Spannweite trifft immer (Identität).
2. Gleiche Gewichte, DEA und SMAA ändern sich nicht, wenn die gemeldeten Gewichte ändern; DEA ändert sich nicht bei Umskalierung der Kriterien.
3. Dominanz: Eine dominierte Alternative steht bei jedem Verfahren nicht vor ihrem Dominator.
4. Skala: Min-Max-Summe, PROMETHEE und ELECTRE ändern sich bei affiner Umskalierung eines Kriteriums nie, die Summe auf der Nenn-Spannweite schon.
5. Rauschen und Störung: Gewichte bleiben eine Verteilung; die Störung ändert kein Vorzeichen.
6. Übereinstimmung: Ist eine Alternative in allen Kriterien am besten, wählen alle zehn Verfahren sie.
"""

import numpy as np
import pytest

import vgl_constants as C
import vgl_dea as dea
import vgl_evaluation as ev
import vgl_methods as md
import vgl_score as sc
import vgl_smaa as sm
from entscheidungsmatrix_generator import SplitMix64, make_matrix

SEEDS = tuple(range(300000, 300040))


def test_the_sum_on_the_nominal_range_always_hits_the_raw_value_decider():
    for seed in SEEDS:
        a = ev.analyse(ev.Settings(decider="roh_linear", seed=seed))
        assert a.picks["sum_nom"] == a.best and a.regrets["sum_nom"] == 0.0 and a.taus["sum_nom"] == pytest.approx(1.0)


def test_methods_without_weight_knowledge_ignore_the_reported_weights():
    for seed in SEEDS[:15]:
        x = make_matrix(10, 5, 0.0, SplitMix64(seed))
        W = sm.sample_weights(300, 5, SplitMix64(seed))
        args = (np.asarray(x.signed), np.asarray(x.raw), np.asarray(x.directions))
        a = ev.all_scores(*args, np.asarray(x.weights), W)
        b = ev.all_scores(*args, np.full(5, 0.2), W)
        for m in ("equal", "dea_ce", "smaa"):
            assert a[m].tolist() == b[m].tolist(), m
        assert any(a[m].tolist() != b[m].tolist() for m in C.WEIGHTED)


def test_dea_does_not_depend_on_the_unit_of_a_criterion():
    factors = np.array([3.0, 0.01, 7.0, 25.0, 0.2])
    for seed in SEEDS[:15]:
        x = make_matrix(10, 5, 0.0, SplitMix64(seed))
        raw, d = np.asarray(x.raw), np.asarray(x.directions)
        X1, Y1 = sc.normalise_units(*sc.split_inputs_outputs(raw, d))
        X2, Y2 = sc.normalise_units(*sc.split_inputs_outputs(raw * factors, d))
        assert dea.cross_efficiency(X2, Y2) == pytest.approx(dea.cross_efficiency(X1, Y1), abs=1e-7)


def test_a_dominated_alternative_never_stands_before_its_dominator():
    for seed in SEEDS:
        x = make_matrix(8, 4, 0.0, SplitMix64(seed))
        S, raw, d = np.asarray(x.signed).copy(), np.asarray(x.raw).copy(), np.asarray(x.directions)
        S[1] = S[0] - np.abs(S[0]) * 0.1                                                     # Alternative 1 ist in jedem Kriterium schlechter als 0
        raw[1] = np.abs(S[1])
        scores = ev.all_scores(S, raw, d, np.asarray(x.weights), sm.sample_weights(300, 4, SplitMix64(seed)))
        for m, s in scores.items():
            assert s[0] >= s[1] - 1e-12, m
        for m in ("sum_nom", "sum_minmax", "prom_v", "smaa", "equal", "ahp_dist"):
            assert scores[m][0] > scores[m][1], m


def test_affine_rescaling_of_a_criterion_leaves_min_max_methods_unchanged_but_not_the_nominal_sum():
    changed = 0
    for seed in SEEDS[:20]:
        x = make_matrix(8, 4, 0.0, SplitMix64(seed))
        S = np.asarray(x.signed)
        S2 = S.copy()
        S2[:, 1] = S2[:, 1] * 0.37 + 3.0
        w = np.asarray(x.weights)
        for f in (md.sum_minmax, md.prom_v, md.electre):
            assert f(S2, w) == pytest.approx(f(S, w), abs=1e-9)
        changed += int(not np.allclose(md.sum_nom(S2, w), md.sum_nom(S, w)))
    assert changed == 20


def test_noisy_weights_are_a_distribution_and_perturbation_keeps_signs():
    for seed in SEEDS:
        x = make_matrix(8, 5, 0.0, SplitMix64(seed))
        v = sc.noisy_weights(np.asarray(x.weights), 0.6, SplitMix64(seed + 1))
        assert v.sum() == pytest.approx(1.0) and (v > 0).all()
        p = sc.perturb_raw(np.asarray(x.raw), 0.1, SplitMix64(seed + 2))
        assert (np.sign(p * np.asarray(x.directions)) == np.sign(np.asarray(x.signed))).all()


def test_an_alternative_that_is_best_in_every_criterion_is_chosen_by_all_ten_methods():
    for seed in SEEDS:
        x = make_matrix(9, 5, 0.0, SplitMix64(seed))
        S, raw, d = np.asarray(x.signed).copy(), np.asarray(x.raw).copy(), np.asarray(x.directions)
        S[4] = S.max(axis=0) + np.abs(S.max(axis=0)) * 0.05
        raw[4] = np.abs(S[4])
        scores = ev.all_scores(S, raw, d, np.asarray(x.weights), sm.sample_weights(300, 5, SplitMix64(seed)))
        assert {sc.top1(v) for v in scores.values()} == {4}
