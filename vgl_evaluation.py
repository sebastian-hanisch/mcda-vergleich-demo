"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte (reine Funktionen). Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import vgl_constants as C
import vgl_dea as dea
import vgl_methods as md
import vgl_score as sc
import vgl_smaa as sm
from entscheidungsmatrix_generator import Matrix, SplitMix64, make_matrix


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    return de(100 * x, digits)


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    rho: float = C.DEFAULT_RHO
    sigma: float = C.DEFAULT_SIGMA
    decider: str = C.DEFAULT_DECIDER
    perturb: float = C.DEFAULT_PERTURB
    seed: int = C.DEFAULT_SEED


# ----------------------------------------------------------------------------- eine Matrix

def all_scores(S: np.ndarray, raw: np.ndarray, d: np.ndarray, w_hat: np.ndarray, W: np.ndarray) -> dict[str, np.ndarray]:
    """Wertung aller zehn Verfahren (höher = besser). Gleiche Gewichte, DEA und SMAA kennen w_hat nicht; SMAA wertet mit dem negativen erwarteten Rang."""
    out = {"equal": md.equal_weights(S)}
    out.update(md.weighted_scores(S, w_hat))
    X, Y = sc.normalise_units(*sc.split_inputs_outputs(raw, d))
    out["dea_ce"] = dea.cross_efficiency(X, Y)
    out["smaa"] = -sm.expected_rank(sc.nominal_normalise(S), W)
    return {name: out[name] for name in C.METHODS}


@dataclass
class Bundle:
    """Eine Matrix mit den Wertungen aller Verfahren, ohne und (bei Störung) mit gestörten Rohwerten."""
    x: Matrix
    w_hat: np.ndarray
    scores: dict
    scores_perturbed: dict | None


def run_matrix(seed: int, n: int, m: int, rho: float, sigma: float, perturb: float) -> Bundle:
    x = make_matrix(n, m, rho, SplitMix64(seed))
    w_hat = sc.noisy_weights(x.weights, sigma, SplitMix64(seed + C.OFFSET_NOISE))
    W = sm.sample_weights(C.NS_DRAWS, m, SplitMix64(seed + C.OFFSET_SMAA))
    scores = all_scores(x.signed, x.raw, x.directions, w_hat, W)
    after = None
    if perturb > 0.0:
        raw2 = sc.perturb_raw(x.raw, perturb, SplitMix64(seed + C.OFFSET_PERTURB))
        after = all_scores(raw2 * x.directions, raw2, x.directions, w_hat, W)
    return Bundle(x, w_hat, scores, after)


def judge(b: Bundle, decider: str) -> tuple[np.ndarray, dict]:
    """(Nutzen des Entscheiders, je Verfahren: Wahl, Platz des Besten, tau zum Nutzen, Nutzenverlust)."""
    u = sc.decider_utility(decider, b.x.goodness, b.x.signed, b.x.weights)
    ok = sc.allowed(u)
    out = {}
    for name, s in b.scores.items():
        pick = sc.top1(s)
        out[name] = {"pick": pick, "rank": sc.rank_of_best(u, s), "tau": sc.kendall_tau_b(s[ok], u[ok]), "regret": sc.regret(u, pick)}
    return u, out


def method_tau_matrix(scores: dict) -> np.ndarray:
    """tau-b zwischen den Rangfolgen je zweier Verfahren (Reihenfolge C.METHODS)."""
    return np.array([[sc.kendall_tau_b(scores[a], scores[c]) for c in C.METHODS] for a in C.METHODS])


def mean_off_diagonal(M: np.ndarray) -> float:
    k = len(M)
    return float((M.sum() - np.trace(M)) / (k * (k - 1)))


@dataclass
class Analysis:
    settings: Settings
    x: Matrix
    w_hat: np.ndarray
    u: np.ndarray                # Nutzen des Entscheiders
    best: int                    # sein Bester (kleinste Nummer bei Gleichstand)
    scores: dict
    picks: dict                  # Wahl je Verfahren
    ranks: dict                  # Platz des Besten in der Rangfolge des Verfahrens
    taus: dict                   # tau-b zum Nutzen
    regrets: dict                # Nutzenverlust der Wahl
    picks_perturbed: dict | None
    tau_matrix: np.ndarray       # Verfahren gegen Verfahren
    hits: int                    # wie viele Verfahren den Besten wählen
    distinct: int                # wie viele verschiedene Sieger
    all_agree: bool
    weighted_agree: bool
    agreed_right: bool | None    # der gemeinsame Sieger der sieben gewichteten ist der Beste (None: sie sind nicht einig)


def analyse(settings: Settings) -> Analysis:
    s = settings
    b = run_matrix(s.seed, s.n, s.m, s.rho, s.sigma, s.perturb)
    u, j = judge(b, s.decider)
    best = int(np.argmax(u))
    picks = {name: r["pick"] for name, r in j.items()}
    weighted_picks = {picks[name] for name in C.WEIGHTED}
    after = None if b.scores_perturbed is None else {name: sc.top1(v) for name, v in b.scores_perturbed.items()}
    return Analysis(s, b.x, b.w_hat, u, best, b.scores, picks, {k: r["rank"] for k, r in j.items()}, {k: r["tau"] for k, r in j.items()},
                    {k: r["regret"] for k, r in j.items()}, after, method_tau_matrix(b.scores), sum(p == best for p in picks.values()), len(set(picks.values())),
                    len(set(picks.values())) == 1, len(weighted_picks) == 1, (next(iter(weighted_picks)) == best) if len(weighted_picks) == 1 else None)


# ----------------------------------------------------------------------------- Experimente

def experiment_methods(deciders=C.DECIDERS, seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M) -> dict:
    """Experiment 1: Welche Verfahren treffen? Je Verfahren und Entscheider: Platz 1, Platz 1 unter den ersten drei, tau, Nutzenverlust (exakte Gewichte, unabhängige Kriterien)."""
    acc = {(d, name): {"hit": [], "top3": [], "tau": [], "regret": []} for d in deciders for name in C.METHODS}
    for seed in seeds:
        b = run_matrix(seed, n, m, 0.0, 0.0, 0.0)
        for d in deciders:
            u, j = judge(b, d)
            best = int(np.argmax(u))
            for name, r in j.items():
                a = acc[(d, name)]
                a["hit"].append(r["pick"] == best)
                a["top3"].append(r["rank"] <= 3)
                a["tau"].append(r["tau"])
                a["regret"].append(r["regret"])
    rows = [{"decider": d, "method": name, "hit": float(np.mean(v["hit"])), "top3": float(np.mean(v["top3"])), "tau": float(np.mean(v["tau"])),
             "regret": float(np.mean(v["regret"]))} for (d, name), v in acc.items()]
    return {"rows": rows, "n": n, "m": m, "matrices": len(seeds)}


def _config_grid(sigmas, rhos) -> list[tuple[float, float]]:
    """Alle (sigma, rho), die die Experimente 2 und 3 brauchen: jedes sigma bei rho = 0 und jedes rho bei sigma = 0 (ohne Doppelungen)."""
    cells = [(s, 0.0) for s in sigmas]
    return cells + [(0.0, r) for r in rhos if (0.0, r) not in cells]


def experiment_weights(sigmas=C.SIGMA_CHOICES, rhos=C.RHO_CHOICES, seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M, decider="guete_linear") -> dict:
    """Experiment 2: Was kosten die Gewichte? Platz 1 und tau je Verfahren bei Erhebungsrauschen sigma (rho = 0) und bei Kriterienkorrelation rho (exakte Gewichte)."""
    cells = []
    for sigma, rho in _config_grid(sigmas, rhos):
        hit = {name: [] for name in C.METHODS}
        tau = {name: [] for name in C.METHODS}
        for seed in seeds:
            b = run_matrix(seed, n, m, rho, sigma, 0.0)
            u, j = judge(b, decider)
            for name, r in j.items():
                hit[name].append(r["pick"] == int(np.argmax(u)))
                tau[name].append(r["tau"])
        cells.append({"sigma": sigma, "rho": rho, "hit": {k: float(np.mean(v)) for k, v in hit.items()}, "tau": {k: float(np.mean(v)) for k, v in tau.items()}})
    return {"cells": cells, "decider": decider, "n": n, "m": m, "matrices": len(seeds)}


def experiment_agreement(sigmas=C.SIGMA_CHOICES, rhos=C.RHO_CHOICES, seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M, decider="guete_linear") -> dict:
    """Experiment 3: Übereinstimmung ist kein Beleg. Wie oft wählen die sieben gewichteten (und alle zehn) Verfahren denselben Sieger, und wie oft ist er der Beste?"""
    cells = []
    for sigma, rho in _config_grid(sigmas, rhos):
        agree_w, right_w, agree_all, right_all, sum_ok, sum_ok_disagree, taus = [], [], [], [], [], [], []
        for seed in seeds:
            b = run_matrix(seed, n, m, rho, sigma, 0.0)
            u, j = judge(b, decider)
            best = int(np.argmax(u))
            picks = {name: r["pick"] for name, r in j.items()}
            wp = {picks[name] for name in C.WEIGHTED}
            ap = set(picks.values())
            agree_w.append(len(wp) == 1)
            agree_all.append(len(ap) == 1)
            sum_ok.append(picks["sum_nom"] == best)
            if len(wp) == 1:
                right_w.append(next(iter(wp)) == best)
            else:
                sum_ok_disagree.append(picks["sum_nom"] == best)
            if len(ap) == 1:
                right_all.append(next(iter(ap)) == best)
            taus.append(mean_off_diagonal(method_tau_matrix(b.scores)))
        cells.append({"sigma": sigma, "rho": rho, "weighted_agree": float(np.mean(agree_w)), "weighted_right": float(np.mean(right_w)) if right_w else None,
                      "all_agree": float(np.mean(agree_all)), "all_right": float(np.mean(right_all)) if right_all else None, "sum_right": float(np.mean(sum_ok)),
                      "sum_right_disagree": float(np.mean(sum_ok_disagree)) if sum_ok_disagree else None, "mean_tau": float(np.mean(taus))})
    return {"cells": cells, "decider": decider, "n": n, "m": m, "matrices": len(seeds)}


def experiment_stability(perturbs=C.PERTURB_CHOICES[1:], seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M) -> dict:
    """Experiment 4: Wie stabil? Bleibt der Sieger (und wie ähnlich bleibt die Rangfolge), wenn die Rohwerte mit lognormalem Rauschen gestört werden (exakte Gewichte)."""
    rows = []
    for level in perturbs:
        stay = {name: [] for name in C.METHODS}
        tau = {name: [] for name in C.METHODS}
        for seed in seeds:
            b = run_matrix(seed, n, m, 0.0, 0.0, level)
            for name in C.METHODS:
                stay[name].append(sc.top1(b.scores[name]) == sc.top1(b.scores_perturbed[name]))
                tau[name].append(sc.kendall_tau_b(b.scores[name], b.scores_perturbed[name]))
        rows.append({"perturb": level, "stay": {k: float(np.mean(v)) for k, v in stay.items()}, "tau": {k: float(np.mean(v)) for k, v in tau.items()}})
    return {"rows": rows, "n": n, "m": m, "matrices": len(seeds)}


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def _hit(res: dict, decider: str) -> dict:
    return {r["method"]: r for r in res["rows"] if r["decider"] == decider}


def verdict_methods(res: dict) -> tuple[str, str]:
    lin = _hit(res, "guete_linear")
    floor = _hit(res, "untergrenze")
    weighted_regret = [floor[name]["regret"] for name in C.WEIGHTED]
    best_hit = max(floor[name]["hit"] for name in C.METHODS)
    tops = " und ".join(C.METHOD_SHORT[name] for name in C.METHODS if floor[name]["hit"] == best_hit)
    text = (f"Bei exakt bekannten Gewichten und einem Entscheider mit linearem Güte-Nutzen trifft die gewichtete Summe den Besten in {pct(lin['sum_nom']['hit'])} % "
            f"(Nenn-Spannweite) und {pct(lin['sum_minmax']['hit'])} % (Min-Max), PROMETHEE II in {pct(lin['prom_v']['hit'])} %, TOPSIS in {pct(lin['topsis_vec']['hit'])} und "
            f"{pct(lin['topsis_nom']['hit'])} %, ELECTRE I in {pct(lin['electre']['hit'])} %, AHP in {pct(lin['ahp_dist']['hit'])} %; gleiche Gewichte {pct(lin['equal']['hit'])} %, "
            f"SMAA ohne Gewichtswissen {pct(lin['smaa']['hit'])} %, DEA-Kreuzeffizienz {pct(lin['dea_ce']['hit'])} %. Unterschiede unter etwa 5 Punkten sind bei "
            f"{res['matrices']} Matrizen nicht belastbar. Beim Entscheider mit Untergrenze trifft am häufigsten {tops} ({pct(best_hit)} %); der "
            f"Nutzenverlust der gewichteten Verfahren liegt zwischen {pct(min(weighted_regret))} und {pct(max(weighted_regret))} %.")
    return "info", text


def _cell(res: dict, sigma: float, rho: float) -> dict:
    return next(c for c in res["cells"] if c["sigma"] == sigma and c["rho"] == rho)


def hit_spread(cell: dict) -> float:
    """Spanne der Trefferquote zwischen bestem und schlechtestem Verfahren, in Anteilen."""
    return max(cell["hit"].values()) - min(cell["hit"].values())


def verdict_weights(res: dict) -> tuple[str, str]:
    exact = _cell(res, 0.0, 0.0)["hit"]
    mid = _cell(res, 0.6, 0.0)["hit"]
    high = _cell(res, 1.0, 0.0)["hit"]
    corr = _cell(res, 0.0, 0.5)
    text = (f"Mit Erhebungsrauschen 0,6 auf den Gewichten fällt die Summe von {pct(exact['sum_nom'])} auf {pct(mid['sum_nom'])} % und ELECTRE I von {pct(exact['electre'])} auf "
            f"{pct(mid['electre'])} %; bei Rauschen 1,0 sind es {pct(high['sum_nom'])} % (gleiche Gewichte, die keine Gewichte brauchen: {pct(high['equal'])} %). Bei korrelierten "
            f"Kriterien (ρ = 0,5) schrumpft die Spanne zwischen dem besten und dem schlechtesten Verfahren von {de(100 * hit_spread(_cell(res, 0.0, 0.0)), 1)} auf "
            f"{de(100 * hit_spread(corr), 1)} Punkte.")
    return "warning", text


def verdict_agreement(res: dict) -> tuple[str, str]:
    by_sigma = {s: _cell(res, s, 0.0) for s in C.SIGMA_CHOICES}
    shares = [c["weighted_agree"] for c in by_sigma.values()]
    corr = _cell(res, 0.0, 0.5)
    text = (f"Die sieben Verfahren, die dieselben Gewichte bekommen, wählen in {pct(min(shares))} bis {pct(max(shares))} % der Matrizen denselben Sieger. Der ist bei exakten "
            f"Gewichten in {pct(by_sigma[0.0]['weighted_right'])} % der Beste, bei Rauschen 0,6 in {pct(by_sigma[0.6]['weighted_right'])} %, bei 1,0 in "
            f"{pct(by_sigma[1.0]['weighted_right'])} %: Sie teilen die Eingabe und damit den Fehler, Übereinstimmung belegt nicht, dass der Sieger richtig ist. Alle zehn "
            f"wählen denselben Sieger in {pct(by_sigma[0.0]['all_agree'])} % der Matrizen (bei ρ = 0,5: {pct(corr['all_agree'])} %); das mittlere τ zwischen den Rangfolgen ist "
            f"{de(by_sigma[0.0]['mean_tau'], 2)} (bei ρ = 0,5: {de(corr['mean_tau'], 2)}).")
    return "warning", text


def verdict_stability(res: dict) -> tuple[str, str]:
    parts = []
    for row in sorted(res["rows"], key=lambda r: r["perturb"]):
        stay = row["stay"]
        top = max(C.METHODS, key=lambda name: stay[name])
        low = min(C.METHODS, key=lambda name: stay[name])
        parts.append(f"bei {pct(row['perturb'], 0)} % Rauschen auf den Rohwerten bleibt der Sieger bei {C.METHOD_SHORT[top]} in {pct(stay[top])} % der Matrizen, bei "
                     f"{C.METHOD_SHORT[low]} in {pct(stay[low])} %")
    return "info", "In dieser Welt: " + "; ".join(parts) + ". Ein stabiles Verfahren ist damit nicht richtiger, nur weniger empfindlich."


def verdict_case(a: Analysis) -> tuple[str, str]:
    names = [f"A{i + 1}" for i in range(a.settings.n)]
    best = names[a.best]
    if a.all_agree:
        win = names[next(iter(a.picks.values()))]
        if a.agreed_right:
            return "info", f"Alle zehn Verfahren wählen {win}, und das ist der Beste des Entscheiders. Einigkeit und Richtigkeit fallen hier zusammen."
        return "warning", f"Alle zehn Verfahren wählen {win}, aber der Beste des Entscheiders ist {best}: Einigkeit ist hier kein Beleg."
    if a.weighted_agree:
        win = names[a.picks["sum_nom"]]
        if a.agreed_right:
            return "info", f"Die sieben gewichteten Verfahren wählen {win}, den Besten des Entscheiders; SMAA, DEA und gleiche Gewichte sind anderer Meinung."
        return "warning", (f"Die sieben gewichteten Verfahren sind sich einig und wählen {win}, aber der Beste des Entscheiders ist {best}: Sie teilen dieselben Gewichte und damit "
                           "denselben Fehler.")
    return "info", f"{a.hits} von 10 Verfahren wählen den Besten des Entscheiders ({best}); es gibt {a.distinct} verschiedene Sieger."
