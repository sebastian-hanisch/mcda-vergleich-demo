"""Verdikte an von Hand gesetzten Zeilen (DEMO-PLAYBOOK §12): jede Zahl im Text und die Auswahl der Zellen steht hier."""

from dataclasses import replace

import pytest

import vgl_constants as C
import vgl_evaluation as ev


def _rows(over=None):
    over = over or {}
    rows = []
    for d in C.DECIDERS:
        for m in C.METHODS:
            r = {"decider": d, "method": m, "hit": 0.5, "top3": 0.9, "tau": 0.7, "regret": 0.1}
            r.update(over.get((d, m), {}))
            rows.append(r)
    return {"rows": rows, "n": 12, "m": 5, "matrices": 200}


LIN = {"equal": 0.545, "sum_nom": 0.775, "sum_minmax": 0.79, "topsis_vec": 0.70, "topsis_nom": 0.71, "prom_v": 0.78, "electre": 0.705, "ahp_dist": 0.68, "dea_ce": 0.475, "smaa": 0.54}


def test_verdict_methods_names_every_hit_rate_and_the_floor_decider():
    over = {("guete_linear", m): {"hit": h} for m, h in LIN.items()}
    over.update({("untergrenze", m): {"hit": 0.6, "regret": r} for m, r in zip(C.WEIGHTED, (0.143, 0.143, 0.156, 0.156, 0.158, 0.24, 0.282))})
    over[("untergrenze", "prom_v")]["hit"] = 0.715
    kind, text = ev.verdict_methods(_rows(over))
    assert kind == "info"
    assert "trifft die gewichtete Summe den Besten in 77,5 % (Nenn-Spannweite) und 79,0 % (Min-Max), PROMETHEE II in 78,0 %, TOPSIS in 70,0 und 71,0 %, ELECTRE I in 70,5 %, AHP in 68,0 %" in text
    assert "gleiche Gewichte 54,5 %, SMAA ohne Gewichtswissen 54,0 %, DEA-Kreuzeffizienz 47,5 %" in text
    assert "bei 200 Matrizen nicht belastbar" in text
    assert "trifft am häufigsten PROMETHEE II (71,5 %)" in text
    assert "zwischen 14,3 und 28,2 %" in text


def test_verdict_methods_names_all_tied_best_methods():
    over = {("untergrenze", "sum_nom"): {"hit": 0.715}, ("untergrenze", "sum_minmax"): {"hit": 0.715}}
    _, text = ev.verdict_methods(_rows(over))
    assert "am häufigsten Summe (Nenn) und Summe (Min-Max) (71,5 %)" in text


def _cell(sigma, rho, hit=None):
    h = {m: 0.5 for m in C.METHODS}
    h.update(hit or {})
    return {"sigma": sigma, "rho": rho, "hit": h, "tau": {m: 0.7 for m in C.METHODS}}


def test_verdict_weights_compares_exact_noisy_and_correlated_cells():
    cells = [_cell(0.0, 0.0, {"sum_nom": 0.775, "electre": 0.705, "equal": 0.545, "smaa": 0.54, "dea_ce": 0.475, "prom_v": 0.78, "sum_minmax": 0.79}),
             _cell(0.3, 0.0), _cell(0.6, 0.0, {"sum_nom": 0.635, "electre": 0.535}), _cell(1.0, 0.0, {"sum_nom": 0.54, "equal": 0.545}),
             _cell(0.0, 0.5, {"sum_nom": 0.84, "electre": 0.75, "equal": 0.755, "smaa": 0.76, "dea_ce": 0.735, "prom_v": 0.845, "sum_minmax": 0.835, "topsis_vec": 0.835, "topsis_nom": 0.805, "ahp_dist": 0.84})]
    kind, text = ev.verdict_weights({"cells": cells, "decider": "guete_linear", "n": 12, "m": 5, "matrices": 200})
    assert kind == "warning"
    assert "fällt die Summe von 77,5 auf 63,5 % und ELECTRE I von 70,5 auf 53,5 %" in text
    assert "bei Rauschen 1,0 sind es 54,0 % (gleiche Gewichte, die keine Gewichte brauchen: 54,5 %)" in text
    assert "von 31,5 auf 11,0 Punkte" in text


def test_hit_spread_is_the_range_between_best_and_worst():
    assert ev.hit_spread(_cell(0.0, 0.0, {"prom_v": 0.9, "dea_ce": 0.2})) == pytest.approx(0.7)


def _agree(sigma, rho, wa, wr, aa=0.2, ar=1.0, tau=0.66):
    return {"sigma": sigma, "rho": rho, "weighted_agree": wa, "weighted_right": wr, "all_agree": aa, "all_right": ar, "sum_right": 0.7, "sum_right_disagree": 0.6, "mean_tau": tau}


def test_verdict_agreement_reports_range_right_share_and_all_ten():
    cells = [_agree(0.0, 0.0, 0.345, 1.0, aa=0.18, tau=0.66), _agree(0.3, 0.0, 0.39, 0.962), _agree(0.6, 0.0, 0.365, 0.877), _agree(1.0, 0.0, 0.33, 0.758),
             _agree(0.0, 0.5, 0.585, 1.0, aa=0.465, tau=0.82)]
    kind, text = ev.verdict_agreement({"cells": cells, "decider": "guete_linear", "n": 12, "m": 5, "matrices": 200})
    assert kind == "warning"
    assert "in 33,0 bis 39,0 % der Matrizen denselben Sieger" in text
    assert "bei exakten Gewichten in 100,0 % der Beste, bei Rauschen 0,6 in 87,7 %, bei 1,0 in 75,8 %" in text
    assert "Alle zehn wählen denselben Sieger in 18,0 % der Matrizen (bei ρ = 0,5: 46,5 %)" in text
    assert "mittlere τ zwischen den Rangfolgen ist 0,66 (bei ρ = 0,5: 0,82)" in text
    assert "belegt nicht, dass der Sieger richtig ist" in text


def test_verdict_stability_names_most_and_least_stable_method_per_level():
    def row(level, over, base=0.9):
        stay = {m: base for m in C.METHODS}
        stay.update(over)
        return {"perturb": level, "stay": stay, "tau": {m: 0.9 for m in C.METHODS}}
    rows = [row(0.1, {"ahp_dist": 0.875, "electre": 0.705}, base=0.8), row(0.05, {"topsis_vec": 0.945, "electre": 0.82})]
    kind, text = ev.verdict_stability({"rows": rows, "n": 12, "m": 5, "matrices": 200})
    assert kind == "info"
    assert text.index("5 % Rauschen") < text.index("10 % Rauschen")
    assert "bei 5 % Rauschen auf den Rohwerten bleibt der Sieger bei TOPSIS (Vektor) in 94,5 % der Matrizen, bei ELECTRE I in 82,0 %" in text
    assert "bei 10 % Rauschen auf den Rohwerten bleibt der Sieger bei AHP in 87,5 % der Matrizen, bei ELECTRE I in 70,5 %" in text
    assert text.startswith("In dieser Welt:")


# ----- Verdikt des Falls

def _case(**over):
    a = ev.analyse(ev.Settings(seed=24))
    return replace(a, **over)


def test_case_all_agree_and_right():
    a = _case(all_agree=True, weighted_agree=True, agreed_right=True, picks={m: 3 for m in C.METHODS}, best=3)
    kind, text = ev.verdict_case(a)
    assert kind == "info" and "Alle zehn Verfahren wählen A4, und das ist der Beste des Entscheiders" in text


def test_case_all_agree_and_wrong_is_a_warning():
    a = _case(all_agree=True, weighted_agree=True, agreed_right=False, picks={m: 3 for m in C.METHODS}, best=5)
    kind, text = ev.verdict_case(a)
    assert kind == "warning" and "Alle zehn Verfahren wählen A4, aber der Beste des Entscheiders ist A6: Einigkeit ist hier kein Beleg." in text


def test_case_seven_weighted_agree_and_right():
    picks = {m: 2 for m in C.WEIGHTED}
    picks.update({"equal": 1, "dea_ce": 0, "smaa": 2})
    a = _case(all_agree=False, weighted_agree=True, agreed_right=True, picks=picks, best=2)
    kind, text = ev.verdict_case(a)
    assert kind == "info" and "Die sieben gewichteten Verfahren wählen A3, den Besten des Entscheiders; SMAA, DEA und gleiche Gewichte sind anderer Meinung." in text


def test_case_seven_weighted_agree_and_wrong_is_a_warning():
    picks = {m: 2 for m in C.WEIGHTED}
    picks.update({"equal": 7, "dea_ce": 7, "smaa": 7})
    a = _case(all_agree=False, weighted_agree=True, agreed_right=False, picks=picks, best=7)
    kind, text = ev.verdict_case(a)
    assert kind == "warning"
    assert "Die sieben gewichteten Verfahren sind sich einig und wählen A3, aber der Beste des Entscheiders ist A8: Sie teilen dieselben Gewichte und damit denselben Fehler." in text


def test_case_without_agreement_counts_hits_and_winners():
    a = _case(all_agree=False, weighted_agree=False, agreed_right=None, hits=6, distinct=4, best=0)
    kind, text = ev.verdict_case(a)
    assert kind == "info" and "6 von 10 Verfahren wählen den Besten des Entscheiders (A1); es gibt 4 verschiedene Sieger." in text


def test_case_names_the_last_alternative():
    a = _case(all_agree=True, weighted_agree=True, agreed_right=True, picks={m: 11 for m in C.METHODS}, best=11)
    assert "Alle zehn Verfahren wählen A12, und das ist der Beste" in ev.verdict_case(a)[1]
