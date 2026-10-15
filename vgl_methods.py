"""Sieben gewichtete Verfahren und gleiche Gewichte als reine Funktionen (S, w) -> Wertungen; höher ist besser.

`S` sind die vorzeichenbehafteten Rohwerte (höher = besser), `w` die Gewichte. Jede Wertung hängt nur von der übergebenen Menge ab (keine eingefrorenen Konstanten):
Die Stabilität gegen gestörte Rohwerte rechnet das Verfahren auf den gestörten Werten neu."""

from __future__ import annotations

import numpy as np

import vgl_constants as C
import vgl_score as sc


def sum_nom(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gewichtete Summe auf der festen Nenn-Spannweite."""
    return sc.nominal_normalise(S) @ np.asarray(w, dtype=float)


def sum_minmax(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gewichtete Summe auf Min-Max der Stichprobe."""
    return sc.minmax_normalise(S) @ np.asarray(w, dtype=float)


def equal_weights(S: np.ndarray) -> np.ndarray:
    """Gleiche Gewichte auf der Nenn-Spannweite: kennt die Gewichte des Entscheiders nicht."""
    m = np.asarray(S).shape[1]
    return sum_nom(S, np.full(m, 1.0 / m))


def topsis_vec(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """TOPSIS klassisch: Vektor-Normierung, gewichtet; Ideal und Antiideal aus der Stichprobe; Abstände in der 2-Norm."""
    V = sc.vector_normalise(S) * np.asarray(w, dtype=float)
    dp = np.sqrt(((V - V.max(axis=0)) ** 2).sum(axis=1))
    dm = np.sqrt(((V - V.min(axis=0)) ** 2).sum(axis=1))
    tot = dp + dm
    return np.where(tot == 0, 0.5, dm / np.where(tot == 0, 1.0, tot))


def topsis_nom(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """TOPSIS auf der Nenn-Spannweite mit Nenn-Ideal (alle Werte 1) und Nenn-Antiideal (alle 0)."""
    w = np.asarray(w, dtype=float)
    V = sc.nominal_normalise(S) * w
    dp = np.sqrt(((V - w) ** 2).sum(axis=1))
    dm = np.sqrt((V ** 2).sum(axis=1))
    return dm / (dp + dm)                            # dp + dm >= Norm von w > 0 (Dreiecksungleichung): kein Teiler 0


def prom_v(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """PROMETHEE II mit V-Form (Schwelle p als Anteil der Spannweite) auf Min-Max; Nettofluss."""
    N = sc.minmax_normalise(S)
    n = len(N)
    D = N[:, None, :] - N[None, :, :]
    P = np.clip(D / C.PROMETHEE_P, 0.0, 1.0)
    pi = (P * np.asarray(w, dtype=float)).sum(axis=-1)
    np.fill_diagonal(pi, 0.0)
    return pi.sum(axis=1) / (n - 1) - pi.sum(axis=0) / (n - 1)


def electre(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ELECTRE I ohne Veto: a übertrifft b, wenn die Gewichte der Kriterien, in denen a mindestens so gut ist wie b, zusammen mindestens c* betragen; Copeland-Zahl."""
    N = sc.minmax_normalise(S)
    conc = ((N[:, None, :] >= N[None, :, :]) * np.asarray(w, dtype=float)).sum(axis=-1)
    R = conc >= C.ELECTRE_CSTAR - 1e-12
    np.fill_diagonal(R, False)
    return (R.sum(axis=1) - R.sum(axis=0)).astype(float)


def ahp_dist(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AHP verteilend: positive Werte je Kriterium (↓ als Kehrwert), Spalte durch ihre Summe, gewichtet summiert."""
    S = np.asarray(S, dtype=float)
    raw = np.abs(S)
    down = sc.directions(S.shape[1]) < 0
    v = raw.copy()
    v[:, down] = 1.0 / raw[:, down]                                                   # ↓-Rohwerte sind positiv (Euro, Minuten): kein Teiler 0
    return (v / v.sum(axis=0)) @ np.asarray(w, dtype=float)


WEIGHTED_FUNCTIONS = {"sum_nom": sum_nom, "sum_minmax": sum_minmax, "topsis_vec": topsis_vec, "topsis_nom": topsis_nom, "prom_v": prom_v, "electre": electre, "ahp_dist": ahp_dist}


def weighted_scores(S: np.ndarray, w: np.ndarray) -> dict[str, np.ndarray]:
    """Wertung der sieben gewichteten Verfahren mit den Gewichten `w`."""
    return {name: f(S, w) for name, f in WEIGHTED_FUNCTIONS.items()}
