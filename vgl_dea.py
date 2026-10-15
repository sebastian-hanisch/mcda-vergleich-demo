"""DEA (CCR, eingabeorientiert, Multiplikatorform) und Kreuzeffizienz. Kein Streamlit.

Je Alternative o:  max u·y_o  s.t.  v·x_o = 1,  u·y_k - v·x_k <= 0 für alle k,  u, v >= 0. Die Kreuzeffizienz bewertet jede Alternative mit den
Optimalgewichten aller anderen und mittelt (ein beliebiger Optimalgewichtsvektor des LPs, keine zweite Zielfunktion)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def multipliers(X: np.ndarray, Y: np.ndarray, o: int) -> tuple[float, np.ndarray, np.ndarray]:
    """(theta, u, v) der Alternative o. Die Eingaben sind positiv: das LP ist beschränkt und zulässig."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = X.shape
    s = Y.shape[1]
    c = np.concatenate([-Y[o], np.zeros(p)])
    A = np.hstack([Y, -X])
    Aeq = np.concatenate([np.zeros(s), X[o]])[None, :]
    res = linprog(c, A_ub=A, b_ub=np.zeros(n), A_eq=Aeq, b_eq=[1.0], method="highs")           # Voreinstellung der Grenzen: u, v >= 0
    return float(-res.fun), res.x[:s], res.x[s:]


def dea_ccr(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, list, list]:
    """(theta je Alternative, u-Liste, v-Liste)."""
    out = [multipliers(X, Y, o) for o in range(len(X))]
    return np.array([r[0] for r in out]), [r[1] for r in out], [r[2] for r in out]


def cross_matrix(X: np.ndarray, Y: np.ndarray, U: list, V: list) -> np.ndarray:
    """ce[k, j]: Effizienz von j mit den Gewichten von k, (u_k · y_j) / (v_k · x_j)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return np.array([(Y @ np.asarray(U[k])) / (X @ np.asarray(V[k])) for k in range(len(X))])


def cross_efficiency(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Kreuzeffizienz je Alternative (Mittel über alle Bewerter einschließlich ihrer selbst)."""
    _, U, V = dea_ccr(X, Y)
    return cross_matrix(X, Y, U, V).mean(axis=0)
