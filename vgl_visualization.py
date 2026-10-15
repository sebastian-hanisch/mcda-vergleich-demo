"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import vgl_constants as C
from vgl_evaluation import Analysis, de, pct


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def methods_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.settings.n)
    rows = []
    for m in C.METHODS:
        row = {"Verfahren": C.METHOD_SHORT[m], "kennt die Gewichte": "ja" if C.KNOWS_WEIGHTS[m] else "nein", "Sieger": names[a.picks[m]],
               "trifft den Besten": "ja" if a.picks[m] == a.best else "nein", "Platz des Besten": a.ranks[m], "τ zum Nutzen": a.taus[m], "Nutzenverlust": a.regrets[m]}
        if a.picks_perturbed is not None:
            row["Sieger nach Störung"] = names[a.picks_perturbed[m]]
            row["bleibt"] = "ja" if a.picks_perturbed[m] == a.picks[m] else "nein"
        rows.append(row)
    return pd.DataFrame(rows)


def style_methods(df: pd.DataFrame):
    return df.style.format({"τ zum Nutzen": lambda v: de(v, 2), "Nutzenverlust": lambda v: f"{pct(v)} %"})


def build_utility(a: Analysis) -> go.Figure:
    """Nutzen je Alternative (sortiert); der Beste orange, Alternativen, die ein Verfahren wählt, mit der Anzahl der Verfahren beschriftet."""
    n = a.settings.n
    names = alt_names(n)
    shown = np.where(a.u > C.EXCLUDED_BELOW, a.u, np.nan)
    order = np.argsort(-np.where(np.isnan(shown), -np.inf, shown), kind="stable")
    votes = np.bincount(list(a.picks.values()), minlength=n)
    colors = [C.COLOR_BEST if i == a.best else (C.COLOR_PICK if votes[i] > 0 else C.COLOR_BAR) for i in order]
    fig = go.Figure(go.Bar(x=[names[i] for i in order], y=[0.0 if np.isnan(shown[i]) else float(shown[i]) for i in order], marker_color=colors,
                           text=[(f"{int(votes[i])} Verf." if votes[i] else "") + (" (ausgeschlossen)" if np.isnan(shown[i]) else "") for i in order], textposition="outside",
                           hovertemplate="%{x}: Nutzen %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title="Nutzen des Entscheiders (Orakel)")
    fig.update_xaxes(title="Alternative (orange: Beste des Entscheiders; grün: von mindestens einem Verfahren gewählt)")
    return _fixed(fig, 360)


def build_tau_matrix(a: Analysis) -> go.Figure:
    labels = [C.METHOD_SHORT[m] for m in C.METHODS]
    z = a.tau_matrix
    text = [[de(v, 2) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, text=text, texttemplate="%{text}", colorscale="Blues", zmin=-0.2, zmax=1.0, colorbar=dict(title="τ"),
                               hovertemplate="%{y} gegen %{x}: τ = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=labels, ticktext=labels)
    fig.update_xaxes(tickmode="array", tickvals=labels, ticktext=labels, tickangle=-40)
    return _fixed(fig, 520)


# ----------------------------------------------------------------------------- Experimente

def methods_hit_table(res: dict) -> pd.DataFrame:
    rows = {(r["decider"], r["method"]): r for r in res["rows"]}
    out = []
    for m in C.METHODS:
        row = {"Verfahren": C.METHOD_SHORT[m]}
        for d in C.DECIDERS:
            row[C.DECIDER_SHORT[d]] = f"{pct(rows[(d, m)]['hit'])} %"
        out.append(row)
    return pd.DataFrame(out)


def methods_detail_table(res: dict, decider: str = "guete_linear") -> pd.DataFrame:
    out = []
    for m in C.METHODS:
        r = next(x for x in res["rows"] if x["decider"] == decider and x["method"] == m)
        out.append({"Verfahren": C.METHOD_SHORT[m], "Platz 1": f"{pct(r['hit'])} %", "Platz 1 unter den ersten drei": f"{pct(r['top3'])} %", "τ zum Nutzen": de(r["tau"], 2),
                    "Nutzenverlust": f"{pct(r['regret'])} %"})
    return pd.DataFrame(out)


def build_methods(res: dict) -> go.Figure:
    fig = go.Figure()
    for idx, d in enumerate(C.DECIDERS):
        rows = {r["method"]: r for r in res["rows"] if r["decider"] == d}
        fig.add_trace(go.Bar(x=[C.METHOD_SHORT[m] for m in C.METHODS], y=[100 * rows[m]["hit"] for m in C.METHODS], name=C.DECIDER_SHORT[d], marker_color=C.COLOR_METHOD[idx]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    return _fixed(fig, 400)


def weights_table(res: dict) -> pd.DataFrame:
    out = []
    for c in sorted(res["cells"], key=lambda c: (c["rho"], c["sigma"])):
        row = {"Rauschen σ": de(c["sigma"], 1), "ρ": de(c["rho"], 1)}
        for m in C.METHODS:
            row[C.METHOD_SHORT[m]] = f"{pct(c['hit'][m])} %"
        out.append(row)
    return pd.DataFrame(out)


def build_weights(res: dict) -> go.Figure:
    cells = sorted((c for c in res["cells"] if c["rho"] == 0.0), key=lambda c: c["sigma"])
    fig = go.Figure()
    for idx, m in enumerate(C.METHODS):
        fig.add_trace(go.Scatter(x=[de(c["sigma"], 1) for c in cells], y=[100 * c["hit"][m] for c in cells], mode="lines+markers", name=C.METHOD_SHORT[m],
                                 line=dict(color=C.COLOR_METHOD[idx], width=3 if C.KNOWS_WEIGHTS[m] else 1.5, dash="solid" if C.KNOWS_WEIGHTS[m] else "dot")))
    fig.update_xaxes(title="Erhebungsrauschen σ der Gewichte (gepunktet: Verfahren ohne Gewichtswissen)", type="category")
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    return _fixed(fig, 400)


def agreement_table(res: dict) -> pd.DataFrame:
    out = []
    for c in sorted(res["cells"], key=lambda c: (c["rho"], c["sigma"])):
        out.append({"Rauschen σ": de(c["sigma"], 1), "ρ": de(c["rho"], 1), "sieben gewichtete einig": f"{pct(c['weighted_agree'])} %",
                    "davon der Beste": "-" if c["weighted_right"] is None else f"{pct(c['weighted_right'])} %",
                    "alle zehn einig": f"{pct(c['all_agree'])} %", "davon der Beste": "-" if c["all_right"] is None else f"{pct(c['all_right'])} %",
                    "Summe (Nenn) trifft": f"{pct(c['sum_right'])} %",
                    "… wenn die sieben nicht einig sind": "-" if c["sum_right_disagree"] is None else f"{pct(c['sum_right_disagree'])} %", "mittleres τ": de(c["mean_tau"], 2)})
    return pd.DataFrame(out)


def stability_table(res: dict) -> pd.DataFrame:
    out = []
    for m in C.METHODS:
        row = {"Verfahren": C.METHOD_SHORT[m]}
        for r in sorted(res["rows"], key=lambda r: r["perturb"]):
            row[f"Sieger bleibt ({pct(r['perturb'], 0)} %)"] = f"{pct(r['stay'][m])} %"
            row[f"τ vorher–nachher ({pct(r['perturb'], 0)} %)"] = de(r["tau"][m], 2)
        out.append(row)
    return pd.DataFrame(out)
