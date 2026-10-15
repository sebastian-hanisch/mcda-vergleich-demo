"""SMAA ohne Gewichtswissen: Gewichte gleichverteilt über dem Simplex (Dirichlet(1)), Rangmatrix, erwarteter Rang. Zufall aus SplitMix64 (plattformunabhängig)."""

from __future__ import annotations

import numpy as np

from entscheidungsmatrix_generator import SplitMix64


def sample_weights(ns: int, m: int, rng: SplitMix64) -> np.ndarray:
    """ns Gewichtsvektoren der Länge m, gleichverteilt über dem Simplex (normierte Exponentialverteilte)."""
    e = -np.log1p(-rng.uniform(ns * m)).reshape(ns, m)
    return e / e.sum(axis=1, keepdims=True)


def rank_matrix(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Rang 0 .. n-1 (0 = bester) jeder Alternative (Zeile) in jeder Ziehung (Spalte) nach der gewichteten Summe V @ w. Gleicher Nutzen: kleinere Nummer zuerst."""
    V = np.asarray(V, dtype=float)
    U = V @ np.asarray(W, dtype=float).T
    order = np.argsort(-U, axis=0, kind="stable")
    ranks = np.empty_like(order)
    ranks[order, np.arange(U.shape[1])[None, :]] = np.arange(V.shape[0])[:, None]
    return ranks


def expected_rank(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Erwarteter Rang (1 = bester) je Alternative über die Ziehungen."""
    return rank_matrix(V, W).mean(axis=1) + 1.0
