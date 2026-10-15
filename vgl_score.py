"""Nenn-Spannweite, Normierungen, Nutzen der Entscheider, Rauschen und Störung, Kendalls tau-b, Platz des Besten. Nur numpy; eigene Kopie der kleinen Helfer (kein Import über Repos).

Alle Wertungen stehen im vorzeichenbehafteten Raum, in dem HÖHER BESSER gilt."""

from __future__ import annotations

import numpy as np

import vgl_constants as C
from entscheidungsmatrix_generator import CRITERIA, SplitMix64, utility


def nominal_bounds(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nenn-Spannweite je Kriterium im vorzeichenbehafteten Raum: ↑ [lo, hi], ↓ [-hi, -lo]."""
    lo = np.array([CRITERIA[k][3] if CRITERIA[k][1] > 0 else -CRITERIA[k][4] for k in range(m)], dtype=float)
    hi = np.array([CRITERIA[k][4] if CRITERIA[k][1] > 0 else -CRITERIA[k][3] for k in range(m)], dtype=float)
    return lo, hi


def directions(m: int) -> np.ndarray:
    return np.array([CRITERIA[k][1] for k in range(m)], dtype=float)


def nominal_normalise(S: np.ndarray) -> np.ndarray:
    """Werte auf der festen Nenn-Spannweite [0, 1]: jede Alternative wird für sich bewertet."""
    S = np.asarray(S, dtype=float)
    lo, hi = nominal_bounds(S.shape[1])
    return (S - lo) / (hi - lo)


def minmax_normalise(S: np.ndarray) -> np.ndarray:
    """Min-Max der Stichprobe je Kriterium; ein Kriterium mit lauter gleichen Werten wird 0."""
    S = np.asarray(S, dtype=float)
    lo, hi = S.min(axis=0), S.max(axis=0)
    return (S - lo) / np.where(hi - lo == 0, 1.0, hi - lo)


def vector_normalise(S: np.ndarray) -> np.ndarray:
    """Vektor-Normierung (Division durch die 2-Norm der Spalte); eine Nullspalte bleibt 0."""
    S = np.asarray(S, dtype=float)
    norm = np.sqrt((S ** 2).sum(axis=0))
    return S / np.where(norm == 0, 1.0, norm)


def order_of(score: np.ndarray) -> np.ndarray:
    """Alternativen von der besten zur schlechtesten Wertung; Gleichstand (auf neun Stellen gerundet, LP-Rauschen): kleinere Nummer zuerst."""
    return np.lexsort((np.arange(len(score)), -np.round(np.asarray(score, dtype=float), C.TIE_DECIMALS)))


def top1(score: np.ndarray) -> int:
    return int(order_of(score)[0])


def rank_of_best(utilities: np.ndarray, score: np.ndarray) -> int:
    """Platz (1 = vorn), den die Alternative mit dem größten Nutzen in der Wertung `score` einnimmt."""
    return int(np.flatnonzero(order_of(score) == int(np.argmax(utilities)))[0]) + 1


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)


def allowed(utilities: np.ndarray) -> np.ndarray:
    """Alternativen, die der Entscheider nicht ausschließt (Untergrenze)."""
    return np.asarray(utilities) > C.EXCLUDED_BELOW


def regret(utilities: np.ndarray, pick: int) -> float:
    """Nutzenverlust der Wahl in Anteilen der Nutzenspannweite der erlaubten Alternativen; eine ausgeschlossene Wahl zählt 100 %."""
    u = np.asarray(utilities, dtype=float)
    ok = allowed(u)
    if not ok[pick]:
        return 1.0
    span = u[ok].max() - u[ok].min()
    return 0.0 if span == 0 else float((u[ok].max() - u[pick]) / span)


def decider_utility(kind: str, G: np.ndarray, signed: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Nutzen je Alternative für die fünf Entscheider (G Güte, signed Rohwerte mit Vorzeichen). Untergrenze: ein Kriterium unter Güte THETA_FLOOR gibt UNDER_FLOOR."""
    if kind == "roh_linear":
        return nominal_normalise(signed) @ np.asarray(w, dtype=float)
    forms = {"guete_linear": "linear", "guete_konkav": "konkav", "guete_cobb": "cobb", "untergrenze": "linear"}
    if kind not in forms:
        raise ValueError(f"unbekannter Entscheider: {kind}")
    u = utility(np.asarray(G, dtype=float), np.asarray(w, dtype=float), forms[kind])
    if kind == "untergrenze":
        u = np.where((np.asarray(G) >= C.THETA_FLOOR).all(axis=1), u, C.UNDER_FLOOR)
    return u


def noisy_weights(w: np.ndarray, sigma: float, rng: SplitMix64) -> np.ndarray:
    """Erhebungsrauschen: lognormal multiplikativ, danach auf Summe 1; sigma = 0 gibt die Gewichte unverändert zurück (und verbraucht keinen Zufall)."""
    w = np.asarray(w, dtype=float)
    if sigma == 0.0:
        return w
    v = w * np.exp(sigma * rng.normal(len(w)))
    return v / v.sum()


def perturb_raw(raw: np.ndarray, level: float, rng: SplitMix64) -> np.ndarray:
    """Störung der Rohwerte: lognormal multiplikativ (Vorzeichen und Richtung bleiben); level = 0 gibt die Werte unverändert zurück."""
    raw = np.asarray(raw, dtype=float)
    if level == 0.0:
        return raw
    return raw * np.exp(level * rng.normal(raw.size).reshape(raw.shape))


def split_inputs_outputs(raw: np.ndarray, directions_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eingaben = Kriterien mit Richtung -1 (weniger ist besser: Euro, Minuten), Ausgaben = Kriterien mit Richtung +1 (Punkte). Rohwerte ohne Vorzeichen."""
    raw = np.asarray(raw, dtype=float)
    d = np.asarray(directions_)
    return raw[:, d < 0], raw[:, d > 0]


def normalise_units(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Jede Spalte auf ihr Maximum (Einheiten): DEA mit CCR ändert sich dadurch nicht, die LPs werden besser konditioniert."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return X / X.max(axis=0), Y / Y.max(axis=0)
